from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from src.utils.encoding import encode_state


@dataclass
class TetrisEnv:
    board_height: int = 20
    board_width: int = 10
    rotations: int = 4

    def __post_init__(self) -> None:
        self._rng: np.random.Generator = np.random.default_rng()
        self._board = np.zeros((self.board_height, self.board_width), dtype=np.int8)
        self._current_piece: int = 0
        self._next_piece: int = 0
        self._done: bool = False

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._board = np.zeros((self.board_height, self.board_width), dtype=np.int8)
        self._current_piece = int(self._rng.integers(0, 7))
        self._next_piece = int(self._rng.integers(0, 7))
        self._done = False
        return encode_state(self._board, self._current_piece, self._next_piece)

    def step(self, action):
        if self._done:
            raise RuntimeError("Environment is done. Call reset() to start a new episode.")
        rotation_index, target_x = self._parse_action(action)
        if rotation_index >= self.rotations or rotation_index < 0:
            raise ValueError("Rotation index out of range.")
        if target_x < 0 or target_x >= self.board_width:
            raise ValueError("Target x out of range.")
        reward = 0.0
        info = {
            "rotation_index": rotation_index,
            "target_x": target_x,
            "board": self._board.copy(),
        }
        self._current_piece = self._next_piece
        self._next_piece = int(self._rng.integers(0, 7))
        next_state = encode_state(self._board, self._current_piece, self._next_piece)
        return next_state, reward, self._done, info

    def get_valid_actions(self, state: Optional[Iterable] = None) -> list[int]:
        return [rot * self.board_width + x for rot in range(self.rotations) for x in range(self.board_width)]

    def _parse_action(self, action) -> tuple[int, int]:
        if isinstance(action, tuple) and len(action) == 2:
            return int(action[0]), int(action[1])
        if isinstance(action, (int, np.integer)):
            rotation_index = int(action) // self.board_width
            target_x = int(action) % self.board_width
            return rotation_index, target_x
        raise ValueError("Action must be an int or (rotation_index, target_x) tuple.")
