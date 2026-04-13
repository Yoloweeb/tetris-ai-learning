from __future__ import annotations

from collections import deque
from pathlib import Path
import random

import numpy as np
import tensorflow as tf

from src.env.tetris_env import TetrisEnv
from src.training.model import build_dqn_model
from src.training.state_utils import preprocess_state


ReplayTransition = tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    bool,
]


def choose_action(
    model,
    board: np.ndarray,
    current_piece: np.ndarray,
    next_piece: np.ndarray,
    valid_actions: list[int],
    epsilon: float,
) -> int:
    if not valid_actions:
        return 0

    if random.random() < epsilon:
        return int(random.choice(valid_actions))

    q_values = model.predict(
        [board[np.newaxis, ...], current_piece[np.newaxis, ...], next_piece[np.newaxis, ...]], verbose=0
    )[0]
    masked_q_values = np.full_like(q_values, -1e9, dtype=np.float32)
    masked_q_values[valid_actions] = q_values[valid_actions]
    return int(np.argmax(masked_q_values))


def train_batch(
    model,
    target_model,
    env: TetrisEnv,
    replay_buffer: deque[ReplayTransition],
    batch_size: int,
    gamma: float,
) -> None:
    if len(replay_buffer) < batch_size:
        return

    minibatch = random.sample(replay_buffer, batch_size)

    state_boards = np.array([item[0] for item in minibatch], dtype=np.float32)
    state_currents = np.array([item[1] for item in minibatch], dtype=np.float32)
    state_nexts = np.array([item[2] for item in minibatch], dtype=np.float32)

    next_boards = np.array([item[5] for item in minibatch], dtype=np.float32)
    next_currents = np.array([item[6] for item in minibatch], dtype=np.float32)
    next_nexts = np.array([item[7] for item in minibatch], dtype=np.float32)

    current_q_values = model.predict([state_boards, state_currents, state_nexts], verbose=0)
    next_q_values = target_model.predict([next_boards, next_currents, next_nexts], verbose=0)

    targets = np.array(current_q_values, copy=True)

    for i, transition in enumerate(minibatch):
        _, _, _, action, reward, next_board, next_current, _, done = transition
        if done:
            target_value = reward
        else:
            next_board_2d = np.asarray(next_board[..., 0], dtype=np.int8)
            next_piece_id = int(np.argmax(next_current))
            valid_next_actions = env.get_valid_actions_from_state(next_board_2d, next_piece_id)
            if valid_next_actions:
                max_next_q = float(np.max(next_q_values[i, valid_next_actions]))
            else:
                max_next_q = 0.0
            target_value = reward + gamma * max_next_q
        targets[i, action] = target_value

    model.fit([state_boards, state_currents, state_nexts], targets, batch_size=batch_size, epochs=1, verbose=0)


def main() -> None:
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    episodes = 1000
    batch_size = 32
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    target_sync_every = 10
    max_steps_per_episode = 500

    env = TetrisEnv(seed=seed)
    model = build_dqn_model(number_of_actions=40)
    target_model = build_dqn_model(number_of_actions=40)
    target_model.set_weights(model.get_weights())

    replay_buffer: deque[ReplayTransition] = deque(maxlen=10_000)
    reward_history: list[float] = []
    best_avg_reward_last_10 = float("-inf")

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    for episode in range(1, episodes + 1):
        state = env.reset(seed=seed + episode)
        board, current_piece, next_piece = preprocess_state(state)

        done = False
        total_reward = 0.0
        steps_survived = 0
        total_lines_cleared = 0
        episode_max_height = 0
        episode_holes = 0
        episode_bumpiness = 0
        ended_by_step_cap = False

        while not done and steps_survived < max_steps_per_episode:
            valid_actions = env.get_valid_actions()
            action = choose_action(model, board, current_piece, next_piece, valid_actions, epsilon)

            next_state, reward, done, info = env.step(action)
            next_board, next_current_piece, next_next_piece = preprocess_state(next_state)

            replay_buffer.append(
                (
                    board,
                    current_piece,
                    next_piece,
                    action,
                    float(reward),
                    next_board,
                    next_current_piece,
                    next_next_piece,
                    bool(done),
                )
            )

            train_batch(model, target_model, env, replay_buffer, batch_size, gamma)

            board, current_piece, next_piece = next_board, next_current_piece, next_next_piece
            total_reward += float(reward)
            steps_survived += 1
            total_lines_cleared += int(info.get("lines_cleared", 0))
            episode_max_height = int(info.get("max_height", episode_max_height))
            episode_holes = int(info.get("holes", episode_holes))
            episode_bumpiness = int(info.get("bumpiness", episode_bumpiness))

        if not done and steps_survived >= max_steps_per_episode:
            ended_by_step_cap = True

        reward_history.append(total_reward)
        avg_reward_last_10 = float(np.mean(reward_history[-10:]))

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % target_sync_every == 0:
            target_model.set_weights(model.get_weights())

        if episode % 25 == 0:
            model.save(model_dir / "tetris_dqn_latest.keras")

        if avg_reward_last_10 > best_avg_reward_last_10:
            best_avg_reward_last_10 = avg_reward_last_10
            model.save(model_dir / "tetris_dqn_best.keras")

        end_reason = "step_cap" if ended_by_step_cap else "natural"
        print(
            f"Episode {episode}/{episodes} | reward={total_reward:.2f} | epsilon={epsilon:.3f} "
            f"| steps={steps_survived} | lines={total_lines_cleared} | max_height={episode_max_height} "
            f"| holes={episode_holes} | bumpiness={episode_bumpiness} | avg10={avg_reward_last_10:.2f} | end={end_reason}"
        )

    model.save(model_dir / "tetris_dqn.keras")
    model.save(model_dir / "tetris_dqn_latest.keras")


if __name__ == "__main__":
    main()
