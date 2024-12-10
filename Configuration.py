import numpy as np
import logging
from GameOfLife import GameOfLife


class Configuration:
    def __init__(self, name, initial_state, config):
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

        # Track live cells across generations
        self.live_cells_history = [np.sum(gen) for gen in game.generations]

        if result:
            self.type, stabilization_time = result
            self.lifetime = stabilization_time  # Update the lifetime based on the result
            self.stabilization_time = stabilization_time  # Explicitly update stabilization time
            logging.debug(f"Detected pattern: {self.type} with stabilization time or period: {self.lifetime}")
        else:
            self.type = "Stable"
            self.lifetime = self.config.max_generations  # Max generations if stable
            self.stabilization_time = self.lifetime
            logging.debug(f"Pattern is stable with stabilization time: {self.lifetime}")

        # Final size and size difference
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
