from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Iterable, Optional

import numpy as np

from src.utils.encoding import PIECE_TO_INDEX, encode_state


@dataclass
class TetrisEnv:
    board_height: int = 20
    board_width: int = 10
    rotations: int = 4

    _rng: np.random.Generator = field(init=False, repr=False)
    _game: Any = field(init=False, default=None, repr=False)
    _board: np.ndarray = field(init=False, repr=False)
    _done: bool = field(init=False, default=False, repr=False)
    _current_piece_obj: Any = field(init=False, default=None, repr=False)
    _next_piece_obj: Any = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng()
        self._board = np.zeros((self.board_height, self.board_width), dtype=np.int8)
        self._done = False

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._ensure_external_path()
        self._game = self._create_game_instance(seed)
        self._call_first(self._game, ("reset", "new_game", "start", "init"), default=None)

        self._sync_from_engine()
        if self._current_piece_obj is None:
            self._current_piece_obj = self._get_attr(
                self._game,
                (
                    "current_tetromino",
                    "tetromino",
                    "current_piece",
                    "piece",
                    "active_tetromino",
                ),
            )
        if self._next_piece_obj is None:
            self._next_piece_obj = self._get_attr(
                self._game,
                (
                    "next_tetromino",
                    "next_piece",
                    "preview",
                    "next",
                ),
            )

        self._done = bool(self._get_attr(self._game, ("game_over", "is_over", "done"), default=False))
        return self._encode_current_state()

    def step(self, action):
        if self._done:
            raise RuntimeError("Environment is done. Call reset() to start a new episode.")

        rotation_index, target_x = self._parse_action(action)
        board_before = self._board.copy()

        piece = self._current_piece_obj or self._get_attr(
            self._game,
            ("current_tetromino", "tetromino", "current_piece", "piece", "active_tetromino"),
        )
        rotations = self._extract_rotations(piece)
        if not rotations:
            raise RuntimeError("Unable to read tetromino rotation matrices from python-tetris engine.")

        chosen_rotation = int(rotation_index) % len(rotations)
        shape = rotations[chosen_rotation]
        drop_y = self._find_drop_y(self._board, shape, int(target_x))
        if drop_y is None:
            raise ValueError(f"Invalid placement for action={action}: cannot place at x={target_x}.")

        placed_board = self._place_shape(self._board.copy(), shape, drop_y, int(target_x), self._piece_id(piece) + 1)
        placed_board, cleared_lines = self._clear_lines(placed_board)
        self._board = placed_board.astype(np.int8, copy=False)

        self._sync_board_to_engine()
        self._advance_engine_piece(chosen_rotation, int(target_x))
        self._sync_from_engine(prefer_local_board=True)

        if self._current_piece_obj is None:
            self._current_piece_obj = self._next_piece_obj
        self._next_piece_obj = self._get_attr(
            self._game,
            ("next_tetromino", "next_piece", "preview", "next"),
            default=self._next_piece_obj,
        )

        invalid_next = not bool(self.get_valid_actions())
        top_out = bool(np.any(self._board[0] > 0))
        engine_done = bool(self._get_attr(self._game, ("game_over", "is_over", "done"), default=False))
        self._done = bool(engine_done or top_out or invalid_next)

        reward = float(cleared_lines * cleared_lines)
        if np.array_equal(board_before, self._board):
            reward -= 0.1

        info = {
            "rotation_index": chosen_rotation,
            "target_x": int(target_x),
            "lines_cleared": int(cleared_lines),
            "board": self._board.copy(),
            "engine_game": self._game,
            "matrix": self._get_attr(self._game, ("matrix", "_matrix", "board", "_board")),
            "current_tetromino": self._current_piece_obj,
            "next_tetromino": self._next_piece_obj,
        }

        return self._encode_current_state(), reward, self._done, info

    def get_valid_actions(self, state: Optional[Iterable] = None) -> list[int]:
        del state
        if self._done:
            return []

        piece = self._current_piece_obj or self._get_attr(
            self._game,
            ("current_tetromino", "tetromino", "current_piece", "piece", "active_tetromino"),
        )
        rotations = self._extract_rotations(piece)
        if not rotations:
            return []

        valid: list[int] = []
        for rot_idx, shape in enumerate(rotations):
            for x in range(self.board_width):
                if self._find_drop_y(self._board, shape, x) is not None:
                    valid.append(rot_idx * self.board_width + x)
        return valid

    def _parse_action(self, action) -> tuple[int, int]:
        if isinstance(action, tuple) and len(action) == 2:
            return int(action[0]), int(action[1])
        if isinstance(action, (int, np.integer)):
            action_int = int(action)
            return action_int // self.board_width, action_int % self.board_width
        raise ValueError("Action must be an int or (rotation_index, target_x) tuple.")

    def _ensure_external_path(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        external_root = repo_root / "external" / "python-tetris"
        if not external_root.exists():
            raise RuntimeError(f"Missing external/python-tetris directory: {external_root}")
        if str(external_root) not in sys.path:
            sys.path.insert(0, str(external_root))

    def _create_game_instance(self, seed: Optional[int]) -> Any:
        constructors: list[tuple[str, str]] = [
            ("pytetris.game", "Game"),
            ("pytetris.tetris", "Tetris"),
            ("pytetris", "Game"),
            ("pytetris", "Tetris"),
        ]
        last_error: Exception | None = None
        for module_name, class_name in constructors:
            try:
                module = __import__(module_name, fromlist=[class_name])
                cls = getattr(module, class_name)
                for kwargs in ({}, {"seed": seed}, {"height": self.board_height, "width": self.board_width}):
                    try:
                        return cls(**{k: v for k, v in kwargs.items() if v is not None})
                    except TypeError:
                        continue
            except Exception as exc:
                last_error = exc

        try:
            import pytetris  # type: ignore

            for attr in dir(pytetris):
                candidate = getattr(pytetris, attr)
                if isinstance(candidate, type):
                    try:
                        return candidate()
                    except Exception:
                        continue
        except Exception as exc:
            last_error = exc

        raise RuntimeError("Unable to instantiate python-tetris game object.") from last_error

    def _sync_from_engine(self, *, prefer_local_board: bool = False) -> None:
        engine_board = self._extract_board_from_engine()
        if engine_board is not None and not prefer_local_board:
            self._board = engine_board

        self._current_piece_obj = self._get_attr(
            self._game,
            ("current_tetromino", "tetromino", "current_piece", "piece", "active_tetromino"),
        )
        self._next_piece_obj = self._get_attr(
            self._game,
            ("next_tetromino", "next_piece", "preview", "next"),
        )

    def _extract_board_from_engine(self) -> np.ndarray | None:
        raw = self._get_attr(self._game, ("matrix", "_matrix", "board", "_board"))
        if raw is None:
            return None

        if hasattr(raw, "matrix"):
            raw = getattr(raw, "matrix")
        elif hasattr(raw, "grid"):
            raw = getattr(raw, "grid")
        elif hasattr(raw, "board"):
            raw = getattr(raw, "board")

        arr = np.asarray(raw, dtype=np.int8)
        if arr.shape == (self.board_height, self.board_width):
            return arr.copy()
        return None

    def _sync_board_to_engine(self) -> None:
        raw = self._get_attr(self._game, ("matrix", "_matrix", "board", "_board"))
        if raw is None:
            return

        if hasattr(raw, "matrix"):
            target = getattr(raw, "matrix")
            try:
                target[:] = self._board.tolist()
                return
            except Exception:
                pass
        try:
            raw[:] = self._board.tolist()
        except Exception:
            pass

    def _extract_rotations(self, piece_obj: Any) -> list[np.ndarray]:
        if piece_obj is None:
            return []

        raw = self._get_attr(
            piece_obj,
            (
                "rotations",
                "rotation_matrices",
                "matrices",
                "states",
                "all_rotations",
                "shapes",
            ),
        )

        if raw is None and isinstance(piece_obj, (list, tuple)):
            raw = piece_obj

        if raw is None:
            matrix = self._get_attr(piece_obj, ("matrix", "shape", "cells"))
            if matrix is not None:
                raw = [matrix]

        if raw is None:
            return []

        rotations: list[np.ndarray] = []
        for m in list(raw):
            arr = np.asarray(m, dtype=np.int8)
            if arr.ndim != 2:
                continue
            if arr.size == 0:
                continue
            if np.max(arr) > 1:
                arr = (arr > 0).astype(np.int8)
            rotations.append(arr)
        return rotations

    def _piece_id(self, piece_obj: Any) -> int:
        if piece_obj is None:
            return 0
        if isinstance(piece_obj, (int, np.integer)):
            return int(piece_obj) % 7

        name = self._get_attr(piece_obj, ("name", "kind", "type", "id"))
        if isinstance(name, str):
            upper = name.upper()
            if upper in PIECE_TO_INDEX:
                return PIECE_TO_INDEX[upper]

        if isinstance(name, (int, np.integer)):
            return int(name) % 7

        return 0

    def _find_drop_y(self, board: np.ndarray, shape: np.ndarray, x: int) -> int | None:
        h, w = shape.shape
        if x < 0 or x + w > self.board_width:
            return None

        last_valid: int | None = None
        for y in range(-h, self.board_height + 1):
            if self._collides(board, shape, y, x):
                break
            last_valid = y
        if last_valid is None:
            return None
        return last_valid

    def _collides(self, board: np.ndarray, shape: np.ndarray, y: int, x: int) -> bool:
        for r in range(shape.shape[0]):
            for c in range(shape.shape[1]):
                if shape[r, c] == 0:
                    continue
                by = y + r
                bx = x + c
                if bx < 0 or bx >= self.board_width:
                    return True
                if by >= self.board_height:
                    return True
                if by >= 0 and board[by, bx] != 0:
                    return True
        return False

    def _place_shape(self, board: np.ndarray, shape: np.ndarray, y: int, x: int, piece_value: int) -> np.ndarray:
        out = board.copy()
        for r in range(shape.shape[0]):
            for c in range(shape.shape[1]):
                if shape[r, c] == 0:
                    continue
                by = y + r
                bx = x + c
                if 0 <= by < self.board_height and 0 <= bx < self.board_width:
                    out[by, bx] = piece_value
        return out

    def _clear_lines(self, board: np.ndarray) -> tuple[np.ndarray, int]:
        full = np.all(board > 0, axis=1)
        cleared = int(np.sum(full))
        if cleared == 0:
            return board, 0

        remaining = board[~full]
        refill = np.zeros((cleared, self.board_width), dtype=np.int8)
        out = np.vstack([refill, remaining])
        return out, cleared

    def _advance_engine_piece(self, rotation_index: int, target_x: int) -> None:
        applied = self._call_first(
            self._game,
            (
                "place",
                "hard_drop",
                "drop",
                "lock",
                "step",
                "tick",
            ),
            args=(rotation_index, target_x),
            default=None,
        )
        if applied is None:
            self._call_first(self._game, ("spawn", "new_piece", "next_piece"), default=None)

    def _encode_current_state(self):
        return encode_state(self._board, self._piece_id(self._current_piece_obj), self._piece_id(self._next_piece_obj))

    @staticmethod
    def _get_attr(obj: Any, names: tuple[str, ...], default: Any = None) -> Any:
        for name in names:
            if hasattr(obj, name):
                value = getattr(obj, name)
                if value is not None:
                    return value
        return default

    @staticmethod
    def _call_first(obj: Any, names: tuple[str, ...], args: tuple[Any, ...] = (), default: Any = None) -> Any:
        for name in names:
            fn = getattr(obj, name, None)
            if callable(fn):
                for call_args in (args, (), args[:1]):
                    try:
                        return fn(*call_args)
                    except TypeError:
                        continue
                    except Exception:
                        return default
        return default
