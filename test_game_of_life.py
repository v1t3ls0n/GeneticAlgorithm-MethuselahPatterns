# test_game_of_life.py

import unittest
import numpy as np
from GameOfLife import GameOfLife
from Config import Config


class TestGameOfLife(unittest.TestCase):
    def setUp(self):
        self.config = Config.get_instance()
        self.config.grid_size = 5
        self.config.max_generations = 10
        self.config.stability_threshold = 3

    def test_stable_pattern(self):
        # Block pattern (still life)
        initial_state = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])
        game = GameOfLife(initial_state=initial_state, config=self.config)
        result = game.detect_pattern()
        self.assertEqual(result[0], "Stable")
        self.assertEqual(result[1], self.config.max_generations)

    def test_oscillator_pattern(self):
        # Blinker pattern (oscillator)
        initial_state = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ])
        game = GameOfLife(initial_state=initial_state, config=self.config)
        result = game.detect_pattern()
        self.assertEqual(result[0], "Oscillator")
        self.assertEqual(result[1], 2)  # Period of Blinker is 2

    def test_extinct_pattern(self):
        # Single live cell (dies out)
        initial_state = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ])
        self.config.grid_size = 3
        self.config.max_generations = 5
        game = GameOfLife(initial_state=initial_state, config=self.config)
        result = game.detect_pattern()
        self.assertEqual(result[0], "Stable")
        self.assertEqual(result[1], self.config.max_generations)
        self.assertEqual(np.sum(game.board), 0)


if __name__ == '__main__':
    unittest.main()
