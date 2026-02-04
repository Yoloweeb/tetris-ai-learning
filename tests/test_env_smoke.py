import numpy as np

from src.env.tetris_env import TetrisEnv


def test_env_smoke():
    env = TetrisEnv()
    state = env.reset(seed=123)
    assert state[0].shape == (20, 10)
    assert state[1].shape == (7,)
    assert state[2].shape == (7,)

    rng = np.random.default_rng(0)
    for _ in range(10):
        valid_actions = env.get_valid_actions(state)
        action = int(rng.choice(valid_actions))
        state, reward, done, info = env.step(action)
        assert isinstance(reward, float)
        assert done is False
        assert state[0].shape == (20, 10)
        assert state[1].shape == (7,)
        assert state[2].shape == (7,)
        assert "rotation_index" in info
        assert "target_x" in info
