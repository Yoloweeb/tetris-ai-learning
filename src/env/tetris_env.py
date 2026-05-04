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
        # Initialize deterministic RNG and empty board storage.
        self._rng = np.random.default_rng(self.seed)
        self._board = np.zeros((self.board_height, self.board_width), dtype=np.int8)
        self._current_piece = 0
        self._next_piece = 0
        self._done = False

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        # Reset board and piece queue for a fresh episode.
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._board = np.zeros((self.board_height, self.board_width), dtype=np.int8)
        self._current_piece = int(self._rng.integers(0, 7))
        self._next_piece = int(self._rng.integers(0, 7))
        self._done = False
        return self._get_state()

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        # Execute one placement and return shaped learning feedback.
        if self._done:
            metrics = self.extract_board_features((self._board > 0).astype(np.int8))
            return self._get_state(), -10.0, True, {"reason": "game_over", **metrics}

        sim_result = self.simulate_action(action)
        if sim_result is None:
            self._done = True
            metrics = self.extract_board_features((self._board > 0).astype(np.int8))
            return self._get_state(), -10.0, True, {"reason": "invalid_action", **metrics}

        simulated_state, metrics = sim_result
        self._board = np.asarray(simulated_state["board_raw"], dtype=np.int8)
        self._current_piece = int(self._next_piece)
        self._next_piece = int(self._rng.integers(0, 7))
        self._done = bool(np.any(self._board[0] > 0))

        # Reward schedule strongly favors multi-line clears.
        lines_cleared = int(metrics["completed_lines"])
        line_clear_reward = 0.0
        if lines_cleared == 1:
            line_clear_reward = 20.0
        elif lines_cleared == 2:
            line_clear_reward = 50.0
        elif lines_cleared == 3:
            line_clear_reward = 100.0
        elif lines_cleared >= 4:
            line_clear_reward = 200.0

        # Penalize unstable boards so the agent avoids risky stacks.
        reward = line_clear_reward
        reward += 0.02
        reward -= 0.005 * float(metrics["max_height"])
        reward -= 0.01 * float(metrics["holes"])
        reward -= 0.003 * float(metrics["bumpiness"])
        if self._done:
            reward -= 10.0

        info = {
            "lines_cleared": lines_cleared,
            "max_height": float(metrics["max_height"]),
            "aggregate_height": float(metrics["aggregate_height"]),
            "holes": float(metrics["holes"]),
            "bumpiness": float(metrics["bumpiness"]),
            "well_sum": float(metrics["well_sum"]),
        }
        return self._get_state(), float(reward), self._done, info

    def get_valid_actions(self) -> list[int]:
        # Return legal actions for the live board state.
        if self._done:
            return []
        return self.get_valid_actions_from_state(self._board, self._current_piece)

    def get_valid_actions_from_state(self, board: np.ndarray, current_piece: int) -> list[int]:
        # Enumerate rotation/column pairs that can be dropped safely.
        board_int = np.asarray(board, dtype=np.int8)
        valid_actions: list[int] = []
        unique_rotations = self._get_unique_rotations(current_piece)

        for rotation, shape in enumerate(unique_rotations):
            for x in range(self.board_width):
                if self._find_drop_position(board_int, shape, x) is not None:
                    valid_actions.append(rotation * self.board_width + x)
        return valid_actions

    def simulate_action(self, action: int) -> tuple[dict[str, Any], dict[str, float]] | None:
        # Preview an action without mutating the real environment state.
        if self._done:
            return None

        rotation = int(action) // self.board_width
        x = int(action) % self.board_width
        shape = self._get_shape_from_action_rotation(self._current_piece, rotation)

        board_copy = np.array(self._board, copy=True)
        drop_y = self._find_drop_position(board_copy, shape, x)
        if drop_y is None:
            return None

        self._place_shape(board_copy, shape, drop_y, x, self._current_piece + 1)
        lines_cleared = self._clear_lines(board_copy)

        binary_board = (board_copy > 0).astype(np.int8)
        features = self.extract_board_features(binary_board)
        features["completed_lines"] = float(lines_cleared)
        features["landing_height"] = float(self._landing_height(shape, drop_y))
        features["eroded_piece_cells"] = float(lines_cleared * int(np.sum(shape > 0)))

        simulated_state: dict[str, Any] = {
            "board": binary_board,
            "board_raw": board_copy,
            "current_piece": int(self._next_piece),
            "next_piece": int(self._sample_hypothetical_next_piece()),
        }
        return simulated_state, features

    def extract_board_features(self, board: np.ndarray) -> dict[str, float]:
        # Compute handcrafted board metrics used by reward and value model.
        board_int = np.asarray(board, dtype=np.int8)
        heights = self._column_heights(board_int)
        return {
            "max_height": float(np.max(heights) if heights.size > 0 else 0.0),
            "aggregate_height": float(np.sum(heights)),
            "holes": float(self._count_holes(board_int)),
            "bumpiness": float(self._compute_bumpiness(heights)),
            "completed_lines": float(np.sum(np.all(board_int > 0, axis=1))),
            "well_sum": float(self._well_sum(heights)),
            "row_transitions": float(self._row_transitions(board_int)),
            "column_transitions": float(self._column_transitions(board_int)),
            "landing_height": 0.0,
            "eroded_piece_cells": 0.0,
        }

    def _get_state(self) -> dict[str, Any]:
        return {
            "board": (self._board > 0).astype(np.int8),
            "current_piece": int(self._current_piece),
            "next_piece": int(self._next_piece),
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
        return int(np.sum(np.abs(np.diff(heights))))

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

    def _well_sum(self, heights: np.ndarray) -> int:
        if heights.size == 0:
            return 0

        total = 0
        for col in range(self.board_width):
            left_height = heights[col - 1] if col > 0 else self.board_height
            right_height = heights[col + 1] if col < self.board_width - 1 else self.board_height
            min_neighbor = min(left_height, right_height)
            depth = max(0, int(min_neighbor - heights[col]))
            total += depth
        return int(total)

    def _row_transitions(self, board: np.ndarray) -> int:
        transitions = 0
        for r in range(self.board_height):
            prev_filled = 1
            for c in range(self.board_width):
                filled = 1 if board[r, c] > 0 else 0
                if filled != prev_filled:
                    transitions += 1
                prev_filled = filled
            if prev_filled == 0:
                transitions += 1
        return int(transitions)

    def _column_transitions(self, board: np.ndarray) -> int:
        transitions = 0
        for c in range(self.board_width):
            prev_filled = 1
            for r in range(self.board_height):
                filled = 1 if board[r, c] > 0 else 0
                if filled != prev_filled:
                    transitions += 1
                prev_filled = filled
            if prev_filled == 0:
                transitions += 1
        return int(transitions)

    def _landing_height(self, shape: np.ndarray, drop_y: int) -> float:
        shape_h = shape.shape[0]
        return float(self.board_height - (drop_y + (shape_h / 2.0)))

    def _sample_hypothetical_next_piece(self) -> int:
        bit_generator = np.random.PCG64()
        bit_generator.state = self._rng.bit_generator.state
        hypothetical_rng = np.random.Generator(bit_generator)
        return int(hypothetical_rng.integers(0, 7))
