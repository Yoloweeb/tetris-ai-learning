from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from src.env.tetris_env import TetrisEnv

DEFAULT_EPISODES = 50
DEFAULT_OUTPUT_PATH = "data/imitation_dataset.npz"
DEFAULT_PROGRESS_EVERY = 5
DEFAULT_MAX_STEPS_PER_EPISODE = 200


def _add_external_tetris_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    external_path = repo_root / "external" / "python-tetris"
    if external_path.exists() and str(external_path) not in sys.path:
        sys.path.insert(0, str(external_path))
    return external_path


def _call_with_supported_args(fn: Callable, **kwargs):
    signature = inspect.signature(fn)
    filtered = {name: value for name, value in kwargs.items() if name in signature.parameters}
    if filtered:
        return fn(**filtered)
    return fn()


def _find_callable(module) -> Callable | None:
    candidate_names = (
        "choose_action",
        "select_action",
        "best_action",
        "get_best_action",
        "act",
    )
    for name in candidate_names:
        candidate = getattr(module, name, None)
        if callable(candidate):
            return candidate

    candidate_class_names = (
        "HeuristicAI",
        "HeuristicAgent",
        "HeuristicBot",
        "Expert",
        "TetrisAI",
    )
    for class_name in candidate_class_names:
        cls = getattr(module, class_name, None)
        if cls is None:
            continue
        try:
            instance = cls()
        except Exception:
            continue
        for method_name in candidate_names:
            method = getattr(instance, method_name, None)
            if callable(method):
                return method
    return None


def _load_expert_policy() -> Callable:
    external_path = _add_external_tetris_path()
    module_candidates = (
        "heuristic",
        "heuristic_ai",
        "agent",
        "ai",
        "bot",
        "expert",
        "tetris",
    )

    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        expert_callable = _find_callable(module)
        if expert_callable is not None:
            return expert_callable

    raise RuntimeError(
        "Could not load heuristic expert policy from external/python-tetris. "
        f"Checked directory: {external_path}"
    )


def _coerce_action(raw_action, valid_actions: Iterable[int]) -> int | None:
    valid_set = set(int(a) for a in valid_actions)

    if isinstance(raw_action, (int, np.integer)):
        action = int(raw_action)
        return action if action in valid_set else None

    if isinstance(raw_action, tuple) and len(raw_action) == 2:
        try:
            rotation_index = int(raw_action[0])
            target_x = int(raw_action[1])
            action = rotation_index * 10 + target_x
        except Exception:
            return None
        return action if action in valid_set else None

    return None


def collect_dataset(
    episodes: int,
    output_path: str,
    progress_every: int = DEFAULT_PROGRESS_EVERY,
    max_steps_per_episode: int = DEFAULT_MAX_STEPS_PER_EPISODE,
) -> None:
    env = TetrisEnv()
    expert_policy = _load_expert_policy()

    boards: list[np.ndarray] = []
    currents: list[np.ndarray] = []
    nexts: list[np.ndarray] = []
    actions: list[int] = []

    skipped_invalid_state = 0
    skipped_invalid_action = 0

    for episode in range(episodes):
        state = env.reset(seed=episode)

        for _ in range(max_steps_per_episode):
            board, current_piece, next_piece = state

            if board.shape != (20, 10) or current_piece.shape != (7,) or next_piece.shape != (7,):
                skipped_invalid_state += 1
                valid_actions = env.get_valid_actions(state)
                if not valid_actions:
                    break
                state, _, done, _ = env.step(valid_actions[0])
                if done:
                    break
                continue

            valid_actions = env.get_valid_actions(state)
            if not valid_actions:
                break

            try:
                raw_action = _call_with_supported_args(
                    expert_policy,
                    board=board,
                    current_piece=current_piece,
                    next_piece=next_piece,
                    state=state,
                    env=env,
                    valid_actions=valid_actions,
                )
            except Exception:
                skipped_invalid_action += 1
                raw_action = None

            action = _coerce_action(raw_action, valid_actions)
            if action is None:
                skipped_invalid_action += 1
                fallback_action = valid_actions[0]
                state, _, done, _ = env.step(fallback_action)
                if done:
                    break
                continue

            boards.append(board.astype(np.int8, copy=False))
            currents.append(current_piece.astype(np.int8, copy=False))
            nexts.append(next_piece.astype(np.int8, copy=False))
            actions.append(action)

            state, _, done, _ = env.step(action)
            if done:
                break

        if (episode + 1) % max(1, progress_every) == 0 or episode + 1 == episodes:
            print(
                f"Episode {episode + 1}/{episodes} | samples={len(actions)} "
                f"| skipped_state={skipped_invalid_state} | skipped_action={skipped_invalid_action}"
            )

    if boards:
        x_board = np.stack(boards).astype(np.int8)
        x_current = np.stack(currents).astype(np.int8)
        x_next = np.stack(nexts).astype(np.int8)
        y_action = np.asarray(actions, dtype=np.int64)
    else:
        x_board = np.empty((0, 20, 10), dtype=np.int8)
        x_current = np.empty((0, 7), dtype=np.int8)
        x_next = np.empty((0, 7), dtype=np.int8)
        y_action = np.empty((0,), dtype=np.int64)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, X_board=x_board, X_current=x_current, X_next=x_next, y_action=y_action)

    print(f"Saved dataset to {output}")
    print(f"X_board shape: {x_board.shape}, dtype: {x_board.dtype}")
    print(f"X_current shape: {x_current.shape}, dtype: {x_current.dtype}")
    print(f"X_next shape: {x_next.shape}, dtype: {x_next.dtype}")
    print(f"y_action shape: {y_action.shape}, dtype: {y_action.dtype}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect imitation-learning data for Tetris.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES, help="Number of episodes to run.")
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to output .npz dataset file.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help="Print progress every N episodes.",
    )
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=DEFAULT_MAX_STEPS_PER_EPISODE,
        help="Maximum number of decision steps per episode.",
    )
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
