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

def _load_expert_policy():
    repo_root = Path(__file__).resolve().parents[2]
    external_path = repo_root / "external" / "python-tetris"
    if str(external_path) not in sys.path:
        sys.path.insert(0, str(external_path))
        
    from pytetris.ai import pierre_dellacherie
    return pierre_dellacherie


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
        module = __import__(EXPERT_MODULE, fromlist=[EXPERT_FUNCTION])
    except Exception as exc:
        raise RuntimeError(
            f"Failed importing expert module '{EXPERT_MODULE}' from {external_root}"
        ) from exc

    policy = getattr(module, EXPERT_FUNCTION, None)
    if not callable(policy):
        raise RuntimeError(
            f"Expert function '{EXPERT_FUNCTION}' is not callable in module '{EXPERT_MODULE}'"
        )
    return policy


def _extract_state(state: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not isinstance(state, tuple) or len(state) != 3:
        return None

    board, current_piece, next_piece = state
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


def _map_expert_action_to_valid(
    env: TetrisEnv, state: tuple[np.ndarray, np.ndarray, np.ndarray], expert_action: Any
) -> int | None:
    valid_actions = env.get_valid_actions(state)
    valid_set = {int(a) for a in valid_actions}

    if isinstance(expert_action, (int, np.integer)):
        action = int(expert_action)
        return action if action in valid_set else None

    if isinstance(expert_action, tuple) and len(expert_action) == 2:
        try:
            expert_rotation = int(expert_action[0])
            expert_target_x = int(expert_action[1])
        except (TypeError, ValueError):
            return None

        for candidate in valid_actions:
            rotation, target_x = env._parse_action(candidate)
            if rotation == expert_rotation and target_x == expert_target_x:
                return int(candidate)

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
    skipped_invalid_expert_actions = 0

    for episode_idx in range(episodes):
        state = env.reset(seed=episode_idx)

        for _ in range(max_steps_per_episode):
            parsed_state = _extract_state(state)
            if parsed_state is None:
                skipped_invalid_states += 1
                break

            board, current_piece, next_piece = parsed_state
            valid_actions = env.get_valid_actions(state)
            if not valid_actions:
                break

            try:
                expert_action = expert_policy(state=state, valid_actions=valid_actions)
            except Exception:
                skipped_invalid_expert_actions += 1
                fallback = int(valid_actions[0])
                state, _, done, _ = env.step(fallback)
                if done:
                    break
                continue

            action = _map_expert_action_to_valid(env, state, expert_action)
            if action is None:
                skipped_invalid_expert_actions += 1
                fallback = int(valid_actions[0])
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
                f"skipped_states={skipped_invalid_states}, "
                f"skipped_expert_actions={skipped_invalid_expert_actions}"
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
    print(f"skipped_invalid_expert_actions: {skipped_invalid_expert_actions}")
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
