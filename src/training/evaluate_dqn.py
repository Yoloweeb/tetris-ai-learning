from __future__ import annotations

from pathlib import Path

import numpy as np
from tensorflow import keras

from src.env.tetris_env import TetrisEnv
from src.training.features import make_feature_vector


def choose_greedy_action(model: keras.Model, env: TetrisEnv, valid_actions: list[int]) -> int:
    if not valid_actions:
        return 0

    candidate_actions: list[int] = []
    feature_vectors: list[np.ndarray] = []

    for action in valid_actions:
        simulation = env.simulate_action(action)
        if simulation is None:
            continue
        simulated_state, feature_dict = simulation
        feature_vectors.append(make_feature_vector(simulated_state, feature_dict))
        candidate_actions.append(int(action))

    if not candidate_actions:
        return 0

    features = np.array(feature_vectors, dtype=np.float32)
    values = model.predict(features, verbose=0).reshape(-1)
    best_idx = int(np.argmax(values))
    return candidate_actions[best_idx]


def main() -> None:
    episodes = 100
    max_steps_per_episode = 500

    model_candidates = [
        Path("models/tetris_value_best_lines.keras"),
        Path("models/tetris_value_best.keras"),
        Path("models/tetris_value_latest.keras"),
    ]
    model_path = next((path for path in model_candidates if path.exists()), None)
    if model_path is None:
        raise FileNotFoundError("No model found at best_lines, best, or latest checkpoints")

    model = keras.models.load_model(model_path)
    env = TetrisEnv(seed=777)

    rewards: list[float] = []
    lines_cleared_history: list[int] = []
    steps_history: list[int] = []

    for episode in range(1, episodes + 1):
        env.reset(seed=777 + episode)
        done = False

        total_reward = 0.0
        total_lines_cleared = 0
        steps_survived = 0

        while not done and steps_survived < max_steps_per_episode:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break

            action = choose_greedy_action(model, env, valid_actions)
            _, reward, done, info = env.step(action)

            total_reward += float(reward)
            total_lines_cleared += int(info.get("lines_cleared", 0))
            steps_survived += 1

        rewards.append(total_reward)
        lines_cleared_history.append(total_lines_cleared)
        steps_history.append(steps_survived)

        print(
            f"Episode {episode}/{episodes} | reward={total_reward:.2f} "
            f"| lines={total_lines_cleared} | steps={steps_survived}"
        )

    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    avg_lines = float(np.mean(lines_cleared_history)) if lines_cleared_history else 0.0
    avg_steps = float(np.mean(steps_history)) if steps_history else 0.0
    max_lines = int(np.max(lines_cleared_history)) if lines_cleared_history else 0
    pct_at_least_1_line = 100.0 * float(np.mean([lines >= 1 for lines in lines_cleared_history])) if lines_cleared_history else 0.0
    pct_at_least_2_lines = 100.0 * float(np.mean([lines >= 2 for lines in lines_cleared_history])) if lines_cleared_history else 0.0

    print("\nEvaluation Summary")
    print(f"episodes={episodes}")
    print(f"average_reward={avg_reward:.3f}")
    print(f"average_lines_cleared={avg_lines:.3f}")
    print(f"average_steps_survived={avg_steps:.3f}")
    print(f"max_lines_single_episode={max_lines}")
    print(f"pct_episodes_at_least_1_line={pct_at_least_1_line:.2f}%")
    print(f"pct_episodes_at_least_2_lines={pct_at_least_2_lines:.2f}%")


if __name__ == "__main__":
    main()
