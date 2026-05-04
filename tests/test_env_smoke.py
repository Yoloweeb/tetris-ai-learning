import numpy as np

from src.env.tetris_env import TetrisEnv


def test_env_smoke():
    env = TetrisEnv()
    state = env.reset(seed=123)

    assert state["board"].shape == (20, 10)
    assert isinstance(state["current_piece"], int)
    assert isinstance(state["next_piece"], int)

    for _ in range(10):
        valid_actions = env.get_valid_actions()
        assert isinstance(valid_actions, list)

        if not valid_actions:
            break

        action = valid_actions[0]
        next_state, reward, done, info = env.step(action)

        assert next_state["board"].shape == (20, 10)
        assert isinstance(next_state["current_piece"], int)
        assert isinstance(next_state["next_piece"], int)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

        state = next_state
        if done:
            break
