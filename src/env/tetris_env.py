from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# 7 tetrominoes in standard Tetris order: I, O, T, S, Z, J, L
BASE_TETROMINOES: dict[int, np.ndarray] = {
    0: np.array([[1, 1, 1, 1]], dtype=np.int8),
    1: np.array([[1, 1], [1, 1]], dtype=np.int8),
    2: np.array([[0, 1, 0], [1, 1, 1]], dtype=np.int8),
    3: np.array([[0, 1, 1], [1, 1, 0]], dtype=np.int8),
    4: np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8),
    5: np.array([[1, 0, 0], [1, 1, 1]], dtype=np.int8),
    6: np.array([[0, 0, 1], [1, 1, 1]], dtype=np.int8),
}


@dataclass
class TetrisEnv:
    board_height: int = 20
    board_width: int = 10
    seed: int = 0

    _rng: np.random.Generator = field(init=False, repr=False)
    _board: np.ndarray = field(init=False, repr=False)
    _current_piece: int = field(init=False, repr=False)
    _next_piece: int = field(init=False, repr=False)
    _done: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._board = np.zeros((self.board_height, self.board_width), dtype=np.int8)
        self._current_piece = 0
        self._next_piece = 0
        self._done = False

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._board.fill(0)
        self._current_piece = int(self._rng.integers(0, 7))
        self._next_piece = int(self._rng.integers(0, 7))
        self._done = False
        return self._get_state()

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self._done:
            return self._get_state(), -10.0, True, {"reason": "game_over"}

        rotation = int(action) // self.board_width
        x = int(action) % self.board_width

        shape = self._get_rotated_shape(self._current_piece, rotation)
        drop_y = self._find_drop_position(self._board, shape, x)

        if drop_y is None:
            self._done = True
            return self._get_state(), -10.0, True, {
                "lines_cleared": 0,
                "rotation": rotation,
                "x": x,
                "reason": "invalid_action",
            }

        before = self._board.copy()
        self._place_shape(self._board, shape, drop_y, x, self._current_piece + 1)
        lines_cleared = self._clear_lines(self._board)

        self._current_piece = self._next_piece
        self._next_piece = int(self._rng.integers(0, 7))

        reached_top = bool(np.any(self._board[0] > 0))
        self._done = reached_top

        reward = float(lines_cleared**2)
        if np.array_equal(before, self._board):
            reward -= 0.1
        if self._done:
            reward -= 10.0

        info = {
            "lines_cleared": lines_cleared,
            "rotation": rotation,
            "x": x,
            "reached_top": reached_top,
        }
        return self._get_state(), reward, self._done, info

    def get_valid_actions(self) -> list[int]:
        if self._done:
            return []

        valid: list[int] = []
        for rotation in range(4):
            shape = self._get_rotated_shape(self._current_piece, rotation)
            for x in range(self.board_width):
                if self._find_drop_position(self._board, shape, x) is not None:
                    valid.append(rotation * self.board_width + x)
        return valid

    def _get_state(self) -> dict[str, Any]:
        return {
            "board": self._board.copy(),
            "current_piece": self._current_piece,
            "next_piece": self._next_piece,
        }

    def _get_rotated_shape(self, piece: int, rotation: int) -> np.ndarray:
        shape = BASE_TETROMINOES[int(piece)]
        return np.rot90(shape, k=rotation % 4).astype(np.int8)

    def _find_drop_position(self, board: np.ndarray, shape: np.ndarray, x: int) -> int | None:
        shape_h, shape_w = shape.shape
        if x < 0 or (x + shape_w) > self.board_width:
            return None

        last_valid_y: int | None = None
        max_y = self.board_height - shape_h
        for y in range(max_y + 1):
            if self._collides(board, shape, y, x):
                break
            last_valid_y = y
        return last_valid_y

    def _collides(self, board: np.ndarray, shape: np.ndarray, y: int, x: int) -> bool:
        shape_h, shape_w = shape.shape
        if y < 0 or x < 0 or (y + shape_h) > self.board_height or (x + shape_w) > self.board_width:
            return True

        board_window = board[y : y + shape_h, x : x + shape_w]
        return bool(np.any((shape > 0) & (board_window > 0)))

    def _place_shape(self, board: np.ndarray, shape: np.ndarray, y: int, x: int, value: int) -> None:
        shape_h, shape_w = shape.shape
        window = board[y : y + shape_h, x : x + shape_w]
        window[shape > 0] = value

    def _clear_lines(self, board: np.ndarray) -> int:
        full_rows = np.where(np.all(board > 0, axis=1))[0]
        lines_cleared = int(full_rows.size)
        if lines_cleared == 0:
            return 0

        board[:] = np.delete(board, full_rows, axis=0)
        new_rows = np.zeros((lines_cleared, self.board_width), dtype=np.int8)
        board[:] = np.vstack([new_rows, board])
        return lines_cleared
