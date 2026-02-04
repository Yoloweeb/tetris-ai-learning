from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


PIECE_NAMES = ("I", "O", "T", "S", "Z", "J", "L")
PIECE_TO_INDEX = {name: idx for idx, name in enumerate(PIECE_NAMES)}


def _piece_index(piece: Optional[object]) -> Optional[int]:
    if piece is None:
        return None
    if isinstance(piece, (int, np.integer)):
        return int(piece)
    if isinstance(piece, str):
        return PIECE_TO_INDEX.get(piece)
    return None


def _one_hot(index: Optional[int], size: int = 7) -> np.ndarray:
    vec = np.zeros(size, dtype=np.int8)
    if index is None:
        return vec
    if 0 <= index < size:
        vec[index] = 1
    return vec


def encode_state(
    board: Iterable[Iterable[int]],
    current_piece: object,
    next_piece: Optional[object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    board_array = np.asarray(board, dtype=np.int8)
    if board_array.shape != (20, 10):
        raise ValueError(f"Board must be shape (20, 10); got {board_array.shape}")
    current_index = _piece_index(current_piece)
    next_index = _piece_index(next_piece)
    return board_array, _one_hot(current_index), _one_hot(next_index)
