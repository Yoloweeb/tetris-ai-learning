from __future__ import annotations

from collections import deque
from pathlib import Path
import random

import numpy as np

from src.env.tetris_env import TetrisEnv
from src.training.model import build_dqn_model


def preprocess_state(state: dict[str, object]) -> np.ndarray:
    board = np.asarray(state["board"], dtype=np.float32)
    return board[..., np.newaxis]


def choose_action(model, state_tensor: np.ndarray, valid_actions: list[int], epsilon: float) -> int:
    if not valid_actions:
        return 0

    if random.random() < epsilon:
        return int(random.choice(valid_actions))

    q_values = model.predict(state_tensor[np.newaxis, ...], verbose=0)[0]
    masked_q = np.full_like(q_values, -1e9, dtype=np.float32)
    masked_q[valid_actions] = q_values[valid_actions]
    return int(np.argmax(masked_q))


def train_batch(model, replay_buffer: deque, batch_size: int, gamma: float) -> None:
    if len(replay_buffer) < batch_size:
        return

    minibatch = random.sample(replay_buffer, batch_size)
    states = np.array([item[0] for item in minibatch], dtype=np.float32)
    next_states = np.array([item[3] for item in minibatch], dtype=np.float32)

    q_values = model.predict(states, verbose=0)
    next_q_values = model.predict(next_states, verbose=0)

    for i, (state, action, reward, next_state, done) in enumerate(minibatch):
        del state, next_state
        target = reward if done else reward + gamma * float(np.max(next_q_values[i]))
        q_values[i, action] = target

    model.fit(states, q_values, epochs=1, verbose=0)


def main() -> None:
    episodes = 20
    batch_size = 32
    gamma = 0.99

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    env = TetrisEnv(seed=0)
    model = build_dqn_model(number_of_actions=40)
    replay_buffer: deque = deque(maxlen=10_000)

    for episode in range(episodes):
        state = env.reset(seed=episode)
        state_tensor = preprocess_state(state)
        done = False
        total_reward = 0.0

        while not done:
            valid_actions = env.get_valid_actions()
            action = choose_action(model, state_tensor, valid_actions, epsilon)
            next_state, reward, done, _ = env.step(action)
            next_state_tensor = preprocess_state(next_state)

            replay_buffer.append((state_tensor, action, reward, next_state_tensor, done))
            train_batch(model, replay_buffer, batch_size, gamma)

            state_tensor = next_state_tensor
            total_reward += reward

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"Episode {episode + 1}/{episodes} reward={total_reward:.2f} epsilon={epsilon:.3f}")

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / "tetris_dqn.keras")
    print("Saved model to models/tetris_dqn.keras")


if __name__ == "__main__":
    main()
