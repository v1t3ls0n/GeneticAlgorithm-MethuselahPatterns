# Configuration.py

import numpy as np
import logging
from GameOfLife import GameOfLife


class Configuration:
    def __init__(self, name, initial_state, config):
        """
        Initialize a configuration.
        :param name: Name of the configuration.
        :param initial_state: A NumPy array representing the initial state.
        :param config: Config instance for shared parameters.
        """
        self.name = name
        self.initial_state = initial_state
        self.config = config
        self.type = None
        self.lifetime = None
        self.stabilization_time = None
        self.initial_size = np.sum(initial_state)
        self.final_size = None
        self.size_difference = None
        self.live_cells_history = []

    def analyze(self):
        """Analyze the configuration using the GameOfLife class."""
        logging.debug(f"Analyzing configuration: {self.name}")
        game = GameOfLife(initial_state=self.initial_state, config=self.config)
        result = game.detect_pattern()

        self.live_cells_history = [np.sum(gen) for gen in game.generations]

        if result:
            self.type, stabilization_time = result
            if self.type == "Stable":
                # For stable patterns, set stabilization_time to max_generations
                self.stabilization_time = self.config.max_generations
            else:
                # For oscillators and spaceships, set stabilization_time to the period
                self.stabilization_time = stabilization_time
            logging.debug(f"Detected pattern: {self.type} with stabilization time {self.stabilization_time}")
        else:
            self.type = "Stable"
            self.stabilization_time = self.config.max_generations
            logging.debug(f"Pattern is stable with stabilization time {self.stabilization_time}")

        self.final_size = np.sum(game.board)
        self.size_difference = self.final_size - self.initial_size
        logging.debug(f"Final size: {self.final_size}, Size difference: {self.size_difference}")

    def summary(self):
        """Return a summary of the configuration analysis."""
        return {
            "Name": self.name,
            "Type": self.type,
            "Lifetime": self.lifetime,
            "Stabilization Time": self.stabilization_time,
            "Initial Size": self.initial_size,
            "Final Size": self.final_size,
            "Size Difference": self.size_difference,
        }
