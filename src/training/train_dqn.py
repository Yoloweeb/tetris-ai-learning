from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.env.tetris_env import TetrisEnv
from src.training.features import make_feature_vector
from src.training.model import build_value_model

ReplayTransition = tuple[np.ndarray, float, list[np.ndarray], bool]


def build_candidates(env: TetrisEnv, actions: list[int], lookahead: bool = False, model=None) -> list[tuple[int, np.ndarray, float]]:
    # Build model-ready features for each legal action candidate.
    candidates: list[tuple[int, np.ndarray, float]] = []
    for action in actions:
        simulation = env.simulate_action(action)
        if simulation is None:
            continue
        simulated_state, feature_dict = simulation
        feature_vector = make_feature_vector(simulated_state, feature_dict)
        combined_value = 0.0
        if lookahead and model is not None:
            # Estimate one extra step so action ranking is less myopic.
            next_actions = env.get_valid_actions_from_state(simulated_state["board_raw"], int(simulated_state["current_piece"]))
            next_feature_vectors: list[np.ndarray] = []
            for next_action in next_actions:
                next_rotation = int(next_action) // env.board_width
                next_x = int(next_action) % env.board_width
                next_shape = env._get_shape_from_action_rotation(int(simulated_state["current_piece"]), next_rotation)
                next_board = np.array(simulated_state["board_raw"], copy=True)
                next_drop_y = env._find_drop_position(next_board, next_shape, next_x)
                if next_drop_y is None:
                    continue
                env._place_shape(next_board, next_shape, next_drop_y, next_x, int(simulated_state["current_piece"]) + 1)
                next_lines = env._clear_lines(next_board)
                next_binary = (next_board > 0).astype(np.int8)
                next_features = env.extract_board_features(next_binary)
                next_features["completed_lines"] = float(next_lines)
                next_features["landing_height"] = float(env._landing_height(next_shape, next_drop_y))
                next_features["eroded_piece_cells"] = float(next_lines * int(np.sum(next_shape > 0)))
                next_state = {"board": next_binary, "current_piece": int(simulated_state["next_piece"]), "next_piece": int(simulated_state["next_piece"])}
                next_feature_vectors.append(make_feature_vector(next_state, next_features))
            if next_feature_vectors:
                next_values = model.predict(np.array(next_feature_vectors, dtype=np.float32), verbose=0).reshape(-1)
                combined_value = float(np.max(next_values))
        candidates.append((int(action), feature_vector, combined_value))
    return candidates


def choose_action(
    model,
    candidates: list[tuple[int, np.ndarray, float]],
    epsilon: float,
) -> tuple[int, np.ndarray]:
    # Pick epsilon-greedy action using value + lookahead score.
    if not candidates:
        return 0, np.zeros(224, dtype=np.float32)

    if random.random() < epsilon:
        action, features, _ = random.choice(candidates)
        return action, features

    features = np.array([candidate[1] for candidate in candidates], dtype=np.float32)
    values = model.predict(features, verbose=0).reshape(-1)
    lookahead_values = np.array([candidate[2] for candidate in candidates], dtype=np.float32)
    scores = values + 0.5 * lookahead_values
    best_index = int(np.argmax(scores))
    best_action, best_features, _ = candidates[best_index]
    return best_action, best_features


def train_batch(
    model,
    target_model,
    replay_buffer: deque[ReplayTransition],
    batch_size: int,
    gamma: float,
) -> None:
    # Fit one TD batch sampled from replay memory.
    if len(replay_buffer) < batch_size:
        return

    minibatch = random.sample(replay_buffer, batch_size)
    current_features = np.array([item[0] for item in minibatch], dtype=np.float32)
    targets = np.zeros((batch_size, 1), dtype=np.float32)

    for idx, (_, reward, next_candidate_features, done) in enumerate(minibatch):
        if done or not next_candidate_features:
            target_value = float(reward)
        else:
            next_features = np.array(next_candidate_features, dtype=np.float32)
            next_values = target_model.predict(next_features, verbose=0).reshape(-1)
            target_value = float(reward) + gamma * float(np.max(next_values))
        targets[idx, 0] = target_value

    model.fit(current_features, targets, batch_size=batch_size, epochs=1, verbose=0)


def main() -> None:
    # --- Training configuration ---
    # Easy continued-training config block.
    resume_training = True
    starting_epsilon = 0.15
    epsilon_min = 0.01
    epsilon_decay = 0.998

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    episodes = 3000
    batch_size = 64
    gamma = 0.99
    epsilon = starting_epsilon
    target_sync_every = 10
    max_steps_per_episode = 500

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = model_dir / "tetris_value_best.keras"
    best_lines_checkpoint_path = model_dir / "tetris_value_best_lines.keras"
    latest_checkpoint_path = model_dir / "tetris_value_latest.keras"

    # --- Model setup ---
    env = TetrisEnv(seed=seed)

    resume_candidates = [best_lines_checkpoint_path, best_checkpoint_path, latest_checkpoint_path]
    resume_path = next((p for p in resume_candidates if p.exists()), None)

    if resume_training and resume_path is not None:
        loaded_model = keras.models.load_model(resume_path)
        if int(loaded_model.input_shape[-1]) == 224:
            model = loaded_model
            print(f"Loaded checkpoint for continued training: {resume_path}")
        else:
            model = build_value_model(input_dim=224)
            print(f"Checkpoint {resume_path} has incompatible input dim. Starting fresh model.")
    else:
        model = build_value_model(input_dim=224)
        if resume_training:
            print("No checkpoint found in preferred resume list. Starting fresh model.")

    target_model = build_value_model(input_dim=224)
    target_model.set_weights(model.get_weights())

    replay_buffer: deque[ReplayTransition] = deque(maxlen=20_000)
    reward_history: list[float] = []
    lines_history: list[int] = []
    steps_history: list[int] = []

    best_avg10_reward = float("-inf")
    best_avg10_lines = float("-inf")
    best_lines_single_episode = 0
    best_episode_stats: dict[str, float | int | str] = {}

    # --- Replay training loop ---
    for episode in range(1, episodes + 1):
        env.reset(seed=seed + episode)

        done = False
        total_reward = 0.0
        total_lines_cleared = 0
        steps_survived = 0
        episode_max_height = 0.0
        episode_holes = 0.0
        episode_bumpiness = 0.0
        end_reason = "natural"

        while not done and steps_survived < max_steps_per_episode:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                done = True
                end_reason = "no_valid_actions"
                break

            candidates = build_candidates(env, valid_actions, lookahead=True, model=model)
            if not candidates:
                done = True
                end_reason = "no_candidates"
                break

            action, selected_features = choose_action(model, candidates, epsilon)
            _, reward, done, info = env.step(action)

            if done:
                next_candidate_features: list[np.ndarray] = []
            else:
                next_actions = env.get_valid_actions()
                next_candidates = build_candidates(env, next_actions, lookahead=False, model=None)
                next_candidate_features = [candidate[1] for candidate in next_candidates]

            # Store transition with all legal next-state options.
            replay_buffer.append((selected_features, float(reward), next_candidate_features, bool(done)))
            train_batch(model, target_model, replay_buffer, batch_size, gamma)

            total_reward += float(reward)
            steps_survived += 1
            total_lines_cleared += int(info.get("lines_cleared", 0))
            episode_max_height = float(info.get("max_height", episode_max_height))
            episode_holes = float(info.get("holes", episode_holes))
            episode_bumpiness = float(info.get("bumpiness", episode_bumpiness))

        if not done and steps_survived >= max_steps_per_episode:
            end_reason = "step_cap"

        reward_history.append(total_reward)
        lines_history.append(total_lines_cleared)
        steps_history.append(steps_survived)

        avg10_reward = float(np.mean(reward_history[-10:]))
        avg10_lines = float(np.mean(lines_history[-10:]))
        avg10_steps = float(np.mean(steps_history[-10:]))
        best_lines_single_episode = max(best_lines_single_episode, total_lines_cleared)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % target_sync_every == 0:
            target_model.set_weights(model.get_weights())

        if episode % 25 == 0:
            model.save(latest_checkpoint_path)

        if avg10_reward > best_avg10_reward:
            best_avg10_reward = avg10_reward
            model.save(best_checkpoint_path)
            best_episode_stats = {
                "episode": episode,
                "reward": total_reward,
                "steps": steps_survived,
                "lines": total_lines_cleared,
                "max_height": episode_max_height,
                "holes": episode_holes,
                "bumpiness": episode_bumpiness,
                "avg10_reward": avg10_reward,
                "avg10_lines": avg10_lines,
                "avg10_steps": avg10_steps,
                "best_lines_single_episode": best_lines_single_episode,
                "epsilon": epsilon,
                "end_reason": end_reason,
            }

        if avg10_lines > best_avg10_lines:
            best_avg10_lines = avg10_lines
            model.save(best_lines_checkpoint_path)

        print(
            f"Episode {episode}/{episodes} | reward={total_reward:.2f} | epsilon={epsilon:.3f} "
            f"| steps={steps_survived} | total_lines={total_lines_cleared} "
            f"| avg10_reward={avg10_reward:.2f} | avg10_lines={avg10_lines:.2f} | avg10_steps={avg10_steps:.2f} "
            f"| best_lines_single={best_lines_single_episode} "
            f"| max_height={episode_max_height:.1f} | holes={episode_holes:.1f} | bumpiness={episode_bumpiness:.1f} "
            f"| end={end_reason}"
        )

    model.save(model_dir / "tetris_value_final.keras")
    model.save(latest_checkpoint_path)

    summary_path = model_dir / "tetris_value_best_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(best_episode_stats, f, indent=2)


if __name__ == "__main__":
    main()
