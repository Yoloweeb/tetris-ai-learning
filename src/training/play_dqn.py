from __future__ import annotations

from pathlib import Path

import numpy as np
from tensorflow import keras

from src.env.tetris_env import TetrisEnv
from src.training.features import make_feature_vector


def board_to_text(board: np.ndarray) -> str:
    return "\n".join("".join("#" if cell > 0 else "." for cell in row) for row in board)


def choose_greedy_action(model: keras.Model, env: TetrisEnv, valid_actions: list[int]) -> tuple[int, float]:
    if not valid_actions:
        return 0, 0.0

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
        return 0, 0.0

    features = np.array(feature_vectors, dtype=np.float32)
    values = model.predict(features, verbose=0).reshape(-1)
    best_idx = int(np.argmax(values))
    return candidate_actions[best_idx], float(values[best_idx])


def main() -> None:
    model_path = Path("models/tetris_value_best.keras")
    if not model_path.exists():
        model_path = Path("models/tetris_value_latest.keras")
    if not model_path.exists():
        raise FileNotFoundError("No saved model found at models/tetris_value_best.keras or models/tetris_value_latest.keras")

    model = keras.models.load_model(model_path)
    print(f"Loaded model: {model_path}")

    env = TetrisEnv(seed=123)
    state = env.reset(seed=123)
    done = False

    while not done:
        valid_actions = env.get_valid_actions()
        action, predicted_value = choose_greedy_action(model, env, valid_actions)
        state, reward, done, info = env.step(action)

        print(
            f"chosen_action={action} predicted_value={predicted_value:.4f} "
            f"reward={reward:.3f} lines_cleared={int(info.get('lines_cleared', 0))} done={done}"
        )
        print(board_to_text(np.asarray(state["board"], dtype=np.int8)))
        print("-" * 20)


if __name__ == "__main__":
    main()
