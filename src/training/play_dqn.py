from __future__ import annotations

from pathlib import Path

import numpy as np
from tensorflow import keras

from src.env.tetris_env import TetrisEnv
from src.training.state_utils import preprocess_state


def board_to_text(board: np.ndarray) -> str:
    return "\n".join("".join("#" if cell > 0 else "." for cell in row) for row in board)


def choose_greedy_action(model: keras.Model, state: dict[str, object], valid_actions: list[int]) -> int:
    if not valid_actions:
        return 0

    board, current_piece, next_piece = preprocess_state(state)
    q_values = model.predict(
        [board[np.newaxis, ...], current_piece[np.newaxis, ...], next_piece[np.newaxis, ...]], verbose=0
    )[0]
    masked_q_values = np.full_like(q_values, -1e9, dtype=np.float32)
    masked_q_values[valid_actions] = q_values[valid_actions]
    return int(np.argmax(masked_q_values))


def main() -> None:
    model_path = Path("models/tetris_dqn.keras")
    if not model_path.exists():
        raise FileNotFoundError("models/tetris_dqn.keras not found")

    model = keras.models.load_model(model_path)
    env = TetrisEnv(seed=123)

    state = env.reset(seed=123)
    done = False

    while not done:
        valid_actions = env.get_valid_actions()
        action = choose_greedy_action(model, state, valid_actions)
        next_state, reward, done, info = env.step(action)
        print(
            f"action={action} reward={reward:.2f} lines_cleared={info.get('lines_cleared', 0)} done={done}"
        )
        print(board_to_text(next_state["board"]))
        print("-" * 20)
        state = next_state


if __name__ == "__main__":
    main()
