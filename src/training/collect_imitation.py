from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.env.tetris_env import TetrisEnv

DEFAULT_EPISODES = 50
DEFAULT_OUTPUT_PATH = "data/imitation_dataset.npz"
DEFAULT_PROGRESS_EVERY = 5
DEFAULT_MAX_STEPS_PER_EPISODE = 200


def _add_external_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    external_root = repo_root / "external" / "python-tetris"
    if str(external_root) not in sys.path:
        sys.path.insert(0, str(external_root))
    return external_root


def _load_expert_policy():
    external_root = _add_external_path()
    if not external_root.exists():
        raise RuntimeError(f"Missing external expert repository: {external_root}")

    try:
        from pytetris.ai import pierre_dellacherie
    except Exception as exc:
        raise RuntimeError(
            f"Failed to import 'pierre_dellacherie' from pytetris.ai using {external_root}"
        ) from exc

    return pierre_dellacherie


def _extract_state(state: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    board: Any = None
    current_piece: Any = None
    next_piece: Any = None

    if isinstance(state, (tuple, list)) and len(state) == 3:
        board, current_piece, next_piece = state
    elif isinstance(state, dict):
        board = state.get("board")
        current_piece = state.get("current_piece", state.get("current"))
        next_piece = state.get("next_piece", state.get("next"))
    else:
        board = getattr(state, "board", None)
        current_piece = getattr(state, "current_piece", getattr(state, "current", None))
        next_piece = getattr(state, "next_piece", getattr(state, "next", None))

    if board is None or current_piece is None or next_piece is None:
        return None

    board_arr = np.asarray(board, dtype=np.int8)
    current_arr = np.asarray(current_piece, dtype=np.int8)
    next_arr = np.asarray(next_piece, dtype=np.int8)

    if board_arr.shape != (20, 10):
        return None
    if current_arr.shape != (7,):
        return None
    if next_arr.shape != (7,):
        return None

    return board_arr, current_arr, next_arr


def _get_attr(obj: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return None


def _extract_rotations(piece_obj: Any) -> Any:
    if piece_obj is None:
        return None
    if isinstance(piece_obj, (list, tuple)) and piece_obj:
        return piece_obj
    return _get_attr(
        piece_obj,
        (
            "rotations",
            "rotation_matrices",
            "matrices",
            "states",
            "all_rotations",
        ),
    )


def _extract_expert_inputs(env: TetrisEnv) -> tuple[Any, Any] | None:
    game_obj = _get_attr(env, ("_game", "game", "_engine", "engine"))

    matrix = None
    current_piece_obj = None
    if game_obj is not None:
        matrix = _get_attr(game_obj, ("matrix", "_matrix", "board", "_board"))
        current_piece_obj = _get_attr(
            game_obj,
            (
                "current_tetromino",
                "tetromino",
                "current_piece",
                "piece",
                "_current_piece",
            ),
        )

    if matrix is None:
        matrix = _get_attr(env, ("_matrix", "matrix", "game_matrix", "_board", "board"))

    if current_piece_obj is None:
        current_piece_obj = _get_attr(
            env,
            (
                "_current_tetromino",
                "current_tetromino",
                "_piece",
                "piece",
                "_current_piece_obj",
            ),
        )

    rotations = _extract_rotations(current_piece_obj)

    if matrix is None or rotations is None:
        return None

    return matrix, rotations


def _extract_rotation_x(expert_result: Any) -> tuple[int, int] | None:
    if isinstance(expert_result, dict):
        if "rotation" in expert_result and (
            "x" in expert_result or "target_x" in expert_result or "column" in expert_result
        ):
            x_val = expert_result.get("x", expert_result.get("target_x", expert_result.get("column")))
            return int(expert_result["rotation"]), int(x_val)
        if "rotation_index" in expert_result and (
            "x" in expert_result or "target_x" in expert_result or "column" in expert_result
        ):
            x_val = expert_result.get("x", expert_result.get("target_x", expert_result.get("column")))
            return int(expert_result["rotation_index"]), int(x_val)
        for nested_key in ("placement", "best", "move", "result"):
            if nested_key in expert_result:
                nested = _extract_rotation_x(expert_result[nested_key])
                if nested is not None:
                    return nested

    if isinstance(expert_result, (tuple, list)) and len(expert_result) >= 2:
        try:
            return int(expert_result[0]), int(expert_result[1])
        except (TypeError, ValueError):
            return None

    rot = _get_attr(expert_result, ("rotation", "rotation_index", "rot"))
    x_val = _get_attr(expert_result, ("x", "target_x", "column"))
    if rot is not None and x_val is not None:
        try:
            return int(rot), int(x_val)
        except (TypeError, ValueError):
            return None

    return None


def _map_expert_result_to_valid(env: TetrisEnv, state: Any, expert_result: Any) -> int | None:
    valid_actions = [int(a) for a in env.get_valid_actions(state)]
    if not valid_actions:
        return None
    valid_set = set(valid_actions)

    if isinstance(expert_result, (int, np.integer)):
        action = int(expert_result)
        if action in valid_set:
            return action

    parsed = _extract_rotation_x(expert_result)
    if parsed is None:
        return None

    if hasattr(env, "_parse_action"):
        target_rotation, target_x = parsed
        for action in valid_actions:
            try:
                rotation, x_pos = env._parse_action(action)
            except Exception:
                continue
            if int(rotation) == target_rotation and int(x_pos) == target_x:
                return int(action)

    return None


def collect_dataset(
    episodes: int,
    output_path: str,
    progress_every: int,
    max_steps_per_episode: int,
) -> None:
    env = TetrisEnv()
    expert_policy = _load_expert_policy()

    x_board: list[np.ndarray] = []
    x_current: list[np.ndarray] = []
    x_next: list[np.ndarray] = []
    y_action: list[int] = []

    skipped_invalid_states = 0
    skipped_missing_expert_inputs = 0
    skipped_invalid_expert_results = 0

    for episode_idx in range(episodes):
        state = env.reset(seed=episode_idx)

        for _ in range(max_steps_per_episode):
            parsed_state = _extract_state(state)
            if parsed_state is None:
                skipped_invalid_states += 1
                break

            board, current_piece, next_piece = parsed_state
            valid_actions = [int(a) for a in env.get_valid_actions(state)]
            if not valid_actions:
                break

            expert_inputs = _extract_expert_inputs(env)
            if expert_inputs is None:
                skipped_missing_expert_inputs += 1
                fallback = valid_actions[0]
                state, _, done, _ = env.step(fallback)
                if done:
                    break
                continue

            raw_matrix, current_rotations = expert_inputs

            try:
                results = expert_policy(raw_matrix, current_rotations)
            except Exception:
                skipped_invalid_expert_results += 1
                fallback = valid_actions[0]
                state, _, done, _ = env.step(fallback)
                if done:
                    break
                continue

            if not isinstance(results, (list, tuple)) or len(results) == 0:
                skipped_invalid_expert_results += 1
                fallback = valid_actions[0]
                state, _, done, _ = env.step(fallback)
                if done:
                    break
                continue

            expert_best_result = results[0]
            action = _map_expert_result_to_valid(env, state, expert_best_result)
            if action is None:
                skipped_invalid_expert_results += 1
                fallback = valid_actions[0]
                state, _, done, _ = env.step(fallback)
                if done:
                    break
                continue

            x_board.append(board)
            x_current.append(current_piece)
            x_next.append(next_piece)
            y_action.append(action)

            state, _, done, _ = env.step(action)
            if done:
                break

        if (episode_idx + 1) % max(1, progress_every) == 0 or (episode_idx + 1) == episodes:
            print(
                f"Progress: episode {episode_idx + 1}/{episodes}, "
                f"samples={len(y_action)}, "
                f"skipped_invalid_states={skipped_invalid_states}, "
                f"skipped_missing_expert_inputs={skipped_missing_expert_inputs}, "
                f"skipped_invalid_expert_results={skipped_invalid_expert_results}"
            )

    if x_board:
        arr_board = np.stack(x_board).astype(np.int8, copy=False)
        arr_current = np.stack(x_current).astype(np.int8, copy=False)
        arr_next = np.stack(x_next).astype(np.int8, copy=False)
        arr_action = np.asarray(y_action, dtype=np.int32)
    else:
        arr_board = np.empty((0, 20, 10), dtype=np.int8)
        arr_current = np.empty((0, 7), dtype=np.int8)
        arr_next = np.empty((0, 7), dtype=np.int8)
        arr_action = np.empty((0,), dtype=np.int32)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, X_board=arr_board, X_current=arr_current, X_next=arr_next, y_action=arr_action)

    print("Collection complete")
    print(f"episodes: {episodes}")
    print(f"collected_samples: {arr_action.shape[0]}")
    print(f"skipped_invalid_states: {skipped_invalid_states}")
    print(f"skipped_missing_expert_inputs: {skipped_missing_expert_inputs}")
    print(f"skipped_invalid_expert_results: {skipped_invalid_expert_results}")
    print(f"X_board: {arr_board.shape}, dtype={arr_board.dtype}")
    print(f"X_current: {arr_current.shape}, dtype={arr_current.dtype}")
    print(f"X_next: {arr_next.shape}, dtype={arr_next.dtype}")
    print(f"y_action: {arr_action.shape}, dtype={arr_action.dtype}")
    print(f"saved_to: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect imitation-learning samples for Tetris.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
    parser.add_argument("--max-steps-per-episode", type=int, default=DEFAULT_MAX_STEPS_PER_EPISODE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collect_dataset(
        episodes=args.episodes,
        output_path=args.output,
        progress_every=args.progress_every,
        max_steps_per_episode=args.max_steps_per_episode,
    )


if __name__ == "__main__":
    main()
