import numpy as np
import logging
class GameOfLife:
    def __init__(self, initial_state=None, config=None):
        """
        Initializes the Game of Life board and state.
        :param initial_state: Optional initial state (NumPy array).
        :param config: Config object with simulation parameters.
        """
        self.config = config
        self.rows = config.grid_size
        self.cols = config.grid_size
        self.initial_state = initial_state
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        if initial_state is not None:
            rows, cols = initial_state.shape
            if rows > self.rows or cols > self.cols:
                raise ValueError("Initial state exceeds grid size.")
            self.board[:rows, :cols] = initial_state
        self.generations = [self.board_to_tuple(self.board)]

    @staticmethod
    def board_to_tuple(board):
        """Converts the board (NumPy array) to a tuple for hashing."""
        return tuple(map(tuple, board))

    def next_generation(self):
        """Computes the next generation of the board."""
        new_board = self.compute_next_generation()
        self.board = new_board
        self.generations.append(self.board_to_tuple(new_board))

    def compute_next_generation(self):
        """Helper function to calculate the next generation using NumPy."""
        padded_board = np.pad(self.board, pad_width=1, mode='constant', constant_values=0)
        neighbors = sum(
            np.roll(np.roll(padded_board, dx, axis=0), dy, axis=1)
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1),
                           (0, -1),         (0, 1),
                           (1, -1), (1, 0), (1, 1)]
        )
        neighbors = neighbors[1:-1, 1:-1]
        new_board = (neighbors == 3) | ((self.board == 1) & (neighbors == 2))
        return new_board.astype(int)

    def detect_pattern(self):
        """
        Detects whether the pattern is a Methuselah, Oscillator, Spaceship, or Stable.
        """
        seen_generations = {}  # Maps state (tuple) to generation number

        for gen_num in range(self.config.max_generations):
            current_state = self.board_to_tuple(self.board)

            # Check for repeated state
            if current_state in seen_generations:
                period = gen_num - seen_generations[current_state]
                previous_state = self.generations[seen_generations[current_state]]
                shift = self.detect_shift(previous_state, current_state)

                if shift:
                    return "Spaceship", gen_num  # Use gen_num as stabilization time
                else:
                    return "Oscillator", gen_num  # Use gen_num as stabilization time

            # Check for stabilization
            if self.is_stable_pattern(gen_num):
                return "Stable", gen_num  # Use gen_num as stabilization time

            # Store the current state
            seen_generations[current_state] = gen_num
            self.next_generation()

        # Default to "Stable" with max_generations if no pattern is detected
        return "Stable", self.config.max_generations



    def is_stable_pattern(self, gen_num):
        """Checks if the pattern has stabilized (no change over stability_threshold generations)."""
        stability_threshold = self.config.stability_threshold
        if gen_num < stability_threshold:
            return False
        recent_patterns = self.generations[-stability_threshold:]
        is_identical = all(pattern == recent_patterns[0] for pattern in recent_patterns)

        # Additional extinction check
        if np.sum(self.board) == 0:
            return True

        return is_identical


    def classify_stable_pattern(self, gen_num):
        """
        Classifies a stable pattern as Methuselah or Stable.
        """
        if gen_num >= 1000:
            return "Methuselah", gen_num
        return "Stable", gen_num

    def detect_shift(self, previous_gen_tuple, current_gen_tuple):
        """
        Detects if there's a consistent shift (translation) between two generations.
        If all live cells have shifted by the same amount, it's a Spaceship.
        """
        previous_gen = np.array(previous_gen_tuple)
        current_gen = np.array(current_gen_tuple)

        prev_live = np.argwhere(previous_gen == 1)
        curr_live = np.argwhere(current_gen == 1)

        if len(prev_live) == 0 and len(curr_live) == 0:
            return None

        if len(prev_live) != len(curr_live):
            return None

        shifts = curr_live - prev_live

        if shifts.size == 0:
            return None
        first_shift = shifts[0]
        if np.all(shifts == first_shift):
            return tuple(first_shift)
        return None

    def run(self, generations=None, delay=None):
        """
        Simulates the Game of Life for a specified number of generations.
        """
        generations = generations or self.config.max_generations
        delay = delay or self.config.simulation_delay

        for gen_num in range(generations):
            self.next_generation()

    def simulate_interactively(self):
        """
        Simulate interactively, allowing navigation through states.
        Use keyboard input for next/previous generation.
        """
        index = 0
        while True:
            print(f"Generation {index + 1}:")
            self.print_board()

            command = input("\nPress [n] for next, [p] for previous, [q] to quit: ").lower()
            if command == 'n':  # Next generation
                if index < len(self.generations) - 1:
                    index += 1
                else:
                    self.next_generation()
                    index += 1
            elif command == 'p':  # Previous generation
                if index > 0:
                    index -= 1
            elif command == 'q':  # Quit
                break

    def print_board(self):
        """Prints the current board state."""
        for row in self.board:
            print("".join('⬛' if cell else '⬜' for cell in row))
