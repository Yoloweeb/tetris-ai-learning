from __future__ import annotations

import numpy as np


def piece_to_one_hot(piece_id: int) -> np.ndarray:
    one_hot = np.zeros(7, dtype=np.float32)
    one_hot[int(piece_id)] = 1.0
    return one_hot


def preprocess_state(state: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    board = np.asarray(state["board"], dtype=np.float32)[..., np.newaxis]
    current_piece = piece_to_one_hot(int(state["current_piece"]))
    next_piece = piece_to_one_hot(int(state["next_piece"]))
    return board, current_piece, next_piece
