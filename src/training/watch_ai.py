from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tensorflow import keras

from src.env.tetris_env import TetrisEnv
from src.training.features import make_feature_vector


def choose_greedy_action(model: keras.Model, env: TetrisEnv, valid_actions: list[int]) -> tuple[int, float]:
    if not valid_actions:
        return 0, 0.0

    candidate_actions: list[int] = []
    feature_vectors: list[np.ndarray] = []

    for action in valid_actions:
        simulation = env.simulate_action(action)
        if simulation is None:
            continue
        simulated_state, feature_dict = simulation
        feature_vectors.append(make_feature_vector(simulated_state, feature_dict))
        candidate_actions.append(int(action))

    if not candidate_actions:
        return 0, 0.0

    features = np.array(feature_vectors, dtype=np.float32)
    values = model.predict(features, verbose=0).reshape(-1)
    best_idx = int(np.argmax(values))
    return candidate_actions[best_idx], float(values[best_idx])


def load_best_model() -> tuple[keras.Model, Path]:
    model_candidates = [
        Path("models/tetris_value_best_lines.keras"),
        Path("models/tetris_value_best.keras"),
        Path("models/tetris_value_latest.keras"),
    ]
    for model_path in model_candidates:
        if model_path.exists():
            return keras.models.load_model(model_path), model_path
    raise FileNotFoundError(
        "No saved model found at models/tetris_value_best_lines.keras, "
        "models/tetris_value_best.keras, or models/tetris_value_latest.keras"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch trained Tetris AI in a pygame window.")
    parser.add_argument("--seed", type=int, default=123, help="Environment seed.")
    parser.add_argument("--fps", type=float, default=7.0, help="Moves per second for playback speed.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run.")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum steps per episode.")
    return parser.parse_args()


def main() -> None:
    try:
        import pygame
    except ImportError as exc:
        raise SystemExit("Install pygame to use visual playback: pip install pygame") from exc

    args = parse_args()
    fps = max(0.1, float(args.fps))

    model, model_path = load_best_model()
    print(f"Loaded model: {model_path}")

    env = TetrisEnv(seed=int(args.seed))

    cell_size = 28
    board_width_px = env.board_width * cell_size
    board_height_px = env.board_height * cell_size
    side_panel_width = 300

    pygame.init()
    pygame.display.set_caption("Tetris AI Showcase")
    screen = pygame.display.set_mode((board_width_px + side_panel_width, board_height_px))
    font = pygame.font.SysFont("consolas", 24)
    small_font = pygame.font.SysFont("consolas", 20)
    clock = pygame.time.Clock()

    bg_color = (12, 12, 18)
    empty_color = (26, 26, 34)
    grid_color = (45, 45, 60)
    piece_colors = {
        0: (0, 240, 240),
        1: (240, 240, 0),
        2: (160, 0, 240),
        3: (0, 220, 0),
        4: (220, 0, 0),
        5: (0, 0, 220),
        6: (240, 140, 0),
    }
    piece_names = ["I", "O", "T", "S", "Z", "J", "L"]

    total_reward_all = 0.0

    for episode in range(1, int(args.episodes) + 1):
        state = env.reset(seed=int(args.seed) + episode - 1)
        done = False
        steps = 0
        lines_cleared = 0
        total_reward = 0.0
        last_value = 0.0

        while not done and steps < int(args.max_steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return

            valid_actions = env.get_valid_actions()
            action, last_value = choose_greedy_action(model, env, valid_actions)
            state, reward, done, info = env.step(action)

            steps += 1
            total_reward += float(reward)
            lines_cleared += int(info.get("lines_cleared", 0))

            board_raw = np.asarray(env._board, dtype=np.int8)

            screen.fill(bg_color)

            for r in range(env.board_height):
                for c in range(env.board_width):
                    cell_value = int(board_raw[r, c])
                    color = empty_color if cell_value <= 0 else piece_colors.get(cell_value - 1, (200, 200, 200))
                    rect = pygame.Rect(c * cell_size, r * cell_size, cell_size - 1, cell_size - 1)
                    pygame.draw.rect(screen, color, rect)

            for r in range(env.board_height + 1):
                pygame.draw.line(screen, grid_color, (0, r * cell_size), (board_width_px, r * cell_size), 1)
            for c in range(env.board_width + 1):
                pygame.draw.line(screen, grid_color, (c * cell_size, 0), (c * cell_size, board_height_px), 1)

            panel_x = board_width_px + 16
            text_lines = [
                f"Episode: {episode}/{args.episodes}",
                f"Steps: {steps}",
                f"Lines cleared: {lines_cleared}",
                f"Total reward: {total_reward:.2f}",
                f"Pred value: {last_value:.3f}",
                f"Current piece: {piece_names[int(state['current_piece'])]}",
                f"Next piece: {piece_names[int(state['next_piece'])]}",
                f"FPS: {fps:.1f}",
                "ESC or close window to quit",
            ]
            for i, line in enumerate(text_lines):
                text_surface = (font if i < 2 else small_font).render(line, True, (235, 235, 235))
                screen.blit(text_surface, (panel_x, 20 + i * 34))

            pygame.display.flip()
            clock.tick(fps)

        total_reward_all += total_reward
        print(
            f"Episode {episode}/{args.episodes} | steps={steps} | lines={lines_cleared} | "
            f"reward={total_reward:.2f} | done={done}"
        )

    print(f"Completed {args.episodes} episode(s). Total reward: {total_reward_all:.2f}")
    pygame.quit()


if __name__ == "__main__":
    main()
