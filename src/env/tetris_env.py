from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


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
        self._board = np.zeros((self.board_height, self.board_width), dtype=np.int8)
        self._current_piece = int(self._rng.integers(0, 7))
        self._next_piece = int(self._rng.integers(0, 7))
        self._done = False
        return self._get_state()

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self._done:
            return self._get_state(), -10.0, True, {"reason": "game_over", "lines_cleared": 0}

        rotation = int(action) // self.board_width
        x = int(action) % self.board_width

        shape = self._get_shape_from_action_rotation(self._current_piece, rotation)
        drop_y = self._find_drop_position(self._board, shape, x)

        if drop_y is None:
            self._done = True
            max_height = self._max_height(self._board)
            holes = self._count_holes(self._board)
            bumpiness = self._compute_bumpiness(self._column_heights(self._board))
            return self._get_state(), -10.0, True, {
                "reason": "invalid_action",
                "rotation": rotation,
                "x": x,
                "lines_cleared": 0,
                "max_height": max_height,
                "holes": holes,
                "bumpiness": bumpiness,
            }

        self._place_shape(self._board, shape, drop_y, x, self._current_piece + 1)
        lines_cleared = self._clear_lines(self._board)
        heights = self._column_heights(self._board)
        max_height = self._max_height(self._board)
        holes = self._count_holes(self._board)
        bumpiness = self._compute_bumpiness(heights)

        self._current_piece = self._next_piece
        self._next_piece = int(self._rng.integers(0, 7))

        self._done = bool(np.any(self._board[0] > 0))

        line_clear_reward = 0.0
        if lines_cleared == 1:
            line_clear_reward = 10.0
        elif lines_cleared == 2:
            line_clear_reward = 25.0
        elif lines_cleared == 3:
            line_clear_reward = 45.0
        elif lines_cleared >= 4:
            line_clear_reward = 80.0

        reward = line_clear_reward
        reward += 0.05
        reward -= 0.01 * float(max_height)
        reward -= 0.02 * float(holes)
        reward -= 0.005 * float(bumpiness)
        if self._done:
            reward -= 10.0

        return self._get_state(), reward, self._done, {
            "rotation": rotation,
            "x": x,
            "lines_cleared": lines_cleared,
            "max_height": max_height,
            "holes": holes,
            "bumpiness": bumpiness,
        }

    def get_valid_actions(self) -> list[int]:
        if self._done:
            return []
        return self.get_valid_actions_from_state(self._board, self._current_piece)

    def get_valid_actions_from_state(self, board: np.ndarray, current_piece: int) -> list[int]:
        board_int = np.asarray(board, dtype=np.int8)
        valid_actions: list[int] = []
        unique_rotations = self._get_unique_rotations(current_piece)

        for rotation in range(min(4, len(unique_rotations))):
            shape = unique_rotations[rotation]
            for x in range(self.board_width):
                if self._find_drop_position(board_int, shape, x) is not None:
                    valid_actions.append(rotation * self.board_width + x)
        return valid_actions

    def _get_state(self) -> dict[str, Any]:
        return {
            "board": (self._board > 0).astype(np.int8),
            "current_piece": self._current_piece,
            "next_piece": self._next_piece,
        }

    def _get_unique_rotations(self, piece: int) -> list[np.ndarray]:
        base_shape = BASE_TETROMINOES[int(piece)]
        unique: list[np.ndarray] = []
        seen: set[bytes] = set()

        for rotation in range(4):
            shape = np.rot90(base_shape, k=rotation % 4).astype(np.int8)
            key = shape.tobytes()
            if key not in seen:
                seen.add(key)
                unique.append(shape)
        return unique

    def _get_shape_from_action_rotation(self, piece: int, rotation: int) -> np.ndarray:
        unique_rotations = self._get_unique_rotations(piece)
        index = int(rotation) % len(unique_rotations)
        return unique_rotations[index]

    def _find_drop_position(self, board: np.ndarray, shape: np.ndarray, x: int) -> int | None:
        shape_h, shape_w = shape.shape
        if x < 0 or x + shape_w > self.board_width:
            return None

        last_valid_y: int | None = None
        for y in range(self.board_height - shape_h + 1):
            if self._collides(board, shape, y, x):
                break
            last_valid_y = y
        return last_valid_y

    def _collides(self, board: np.ndarray, shape: np.ndarray, y: int, x: int) -> bool:
        shape_h, shape_w = shape.shape
        if y < 0 or x < 0 or y + shape_h > self.board_height or x + shape_w > self.board_width:
            return True
        board_slice = board[y : y + shape_h, x : x + shape_w]
        return bool(np.any((shape > 0) & (board_slice > 0)))

    def _place_shape(self, board: np.ndarray, shape: np.ndarray, y: int, x: int, value: int) -> None:
        shape_h, shape_w = shape.shape
        board_slice = board[y : y + shape_h, x : x + shape_w]
        board_slice[shape > 0] = value

    def _clear_lines(self, board: np.ndarray) -> int:
        full_rows_mask = np.all(board > 0, axis=1)
        lines_cleared = int(np.sum(full_rows_mask))
        if lines_cleared == 0:
            return 0

        remaining_rows = board[~full_rows_mask]
        new_rows = np.zeros((lines_cleared, self.board_width), dtype=np.int8)
        rebuilt = np.vstack((new_rows, remaining_rows))
        board[:, :] = rebuilt[: self.board_height, : self.board_width]
        return lines_cleared

    def _column_heights(self, board: np.ndarray) -> np.ndarray:
        heights = np.zeros(self.board_width, dtype=np.int32)
        for col in range(self.board_width):
            filled_rows = np.where(board[:, col] > 0)[0]
            if filled_rows.size > 0:
                heights[col] = self.board_height - int(filled_rows[0])
        return heights

    def _compute_bumpiness(self, heights: np.ndarray) -> int:
        if heights.size <= 1:
            return 0
        diffs = np.abs(np.diff(heights))
        return int(np.sum(diffs))

    def _max_height(self, board: np.ndarray) -> int:
        heights = self._column_heights(board)
        return int(np.max(heights)) if heights.size > 0 else 0

    def _count_holes(self, board: np.ndarray) -> int:
        holes = 0
        for col in range(self.board_width):
            column = board[:, col]
            filled_rows = np.where(column > 0)[0]
            if filled_rows.size == 0:
                continue
            top = int(filled_rows[0])
            holes += int(np.sum(column[top:] == 0))
        return holes
