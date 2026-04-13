from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import random

import numpy as np
import tensorflow as tf

from src.env.tetris_env import TetrisEnv
from src.training.features import make_feature_vector
from src.training.model import build_value_model

ReplayTransition = tuple[np.ndarray, float, list[np.ndarray], bool]


def build_candidates(env: TetrisEnv, actions: list[int]) -> list[tuple[int, np.ndarray]]:
    candidates: list[tuple[int, np.ndarray]] = []
    for action in actions:
        simulation = env.simulate_action(action)
        if simulation is None:
            continue
        simulated_state, feature_dict = simulation
        feature_vector = make_feature_vector(simulated_state, feature_dict)
        candidates.append((int(action), feature_vector))
    return candidates


def choose_action(
    model,
    candidates: list[tuple[int, np.ndarray]],
    epsilon: float,
) -> tuple[int, np.ndarray]:
    if not candidates:
        return 0, np.zeros(220, dtype=np.float32)

    if random.random() < epsilon:
        return random.choice(candidates)

    features = np.array([candidate[1] for candidate in candidates], dtype=np.float32)
    values = model.predict(features, verbose=0).reshape(-1)
    best_index = int(np.argmax(values))
    return candidates[best_index]


def train_batch(
    model,
    target_model,
    replay_buffer: deque[ReplayTransition],
    batch_size: int,
    gamma: float,
) -> None:
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
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    episodes = 1000
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    target_sync_every = 10
    max_steps_per_episode = 500

    env = TetrisEnv(seed=seed)
    model = build_value_model(input_dim=220)
    target_model = build_value_model(input_dim=220)
    target_model.set_weights(model.get_weights())

    replay_buffer: deque[ReplayTransition] = deque(maxlen=20_000)
    reward_history: list[float] = []
    lines_history: list[int] = []

    best_avg10_reward = float("-inf")
    best_avg10_lines = float("-inf")
    best_episode_stats: dict[str, float | int | str] = {}

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

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

            candidates = build_candidates(env, valid_actions)
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
                next_candidates = build_candidates(env, next_actions)
                next_candidate_features = [candidate[1] for candidate in next_candidates]

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
        avg10_reward = float(np.mean(reward_history[-10:]))
        avg10_lines = float(np.mean(lines_history[-10:]))

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % target_sync_every == 0:
            target_model.set_weights(model.get_weights())

        if episode % 25 == 0:
            model.save(model_dir / "tetris_value_latest.keras")

        if avg10_reward > best_avg10_reward:
            best_avg10_reward = avg10_reward
            model.save(model_dir / "tetris_value_best.keras")
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
                "end_reason": end_reason,
            }

        if avg10_lines > best_avg10_lines:
            best_avg10_lines = avg10_lines
            model.save(model_dir / "tetris_value_best_lines.keras")

        print(
            f"Episode {episode}/{episodes} | reward={total_reward:.2f} | epsilon={epsilon:.3f} "
            f"| steps={steps_survived} | total_lines={total_lines_cleared} | max_height={episode_max_height:.1f} "
            f"| holes={episode_holes:.1f} | bumpiness={episode_bumpiness:.1f} "
            f"| avg10_reward={avg10_reward:.2f} | avg10_lines={avg10_lines:.2f} | end={end_reason}"
        )

    model.save(model_dir / "tetris_value_final.keras")
    model.save(model_dir / "tetris_value_latest.keras")

    summary_path = model_dir / "tetris_value_best_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(best_episode_stats, f, indent=2)


if __name__ == "__main__":
    main()
