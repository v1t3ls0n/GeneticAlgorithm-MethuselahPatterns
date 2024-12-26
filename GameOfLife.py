"""
GameOfLife.py
-------------

Implements Conway's Game of Life logic. Given an initial configuration (a 1D list
representing a 2D grid), this class simulates each generation until the system
reaches a static or periodic state, or until a maximum number of iterations is reached.

Classes:
    GameOfLife
"""

import random
import logging


class GameOfLife:
    """
    The GameOfLife class simulates Conway's Game of Life for a given initial state.
    It tracks the evolution of the grid through multiple generations and determines
    whether the pattern becomes static (unchanged) or periodic (repeats a previously
    seen state).

    Attributes:
        grid_size (int): The dimension N of the NxN grid.
        grid (list[int]): A flat list of length N*N, containing 0s and 1s (dead or alive cells).
        initial_state (tuple[int]): A tuple storing the initial configuration (immutable).
        history (list[tuple[int]]): A record of all the grid states encountered so far.
        game_iteration_limit (int): A hard limit on the total number of generations to simulate.
        stable_count (int): Counts how many consecutive generations remained static or periodic.
        max_stable_generations (int): Once stable_count reaches this limit, the simulation stops.
        lifespan (int): The total number of unique states the grid has passed through before stopping.
        is_static (int): A flag (0 or 1) indicating if the grid has become static.
        is_periodic (int): A flag (0 or 1) indicating if the grid has become periodic.
        max_alive_cells_count (int): The maximum number of living cells observed in any generation.
        alive_growth (float): The ratio between max and min living cells during the simulation.
        alive_history (list[int]): Number of living cells for each generation (for analysis).
        stableness (float): A ratio indicating how many times the grid was detected stable
                            compared to max_stable_generations.
    """

    def __init__(self, grid_size, initial_state=None):
        """
        Initialize the Game of Life simulation.

        Args:
            grid_size (int): The dimension N of the NxN grid.
            initial_state (Iterable[int], optional): A starting configuration.
                If None, a zero-initialized grid is created.
        """
        self.grid_size = grid_size
        self.grid = [
            0]*(grid_size*grid_size) if initial_state is None else list(initial_state)

        # Store the initial state of the grid
        self.initial_state = tuple(self.grid)
        self.history = [self.initial_state]

        self.game_iteration_limit = 15000
        self.stable_count = 0
        self.max_stable_generations = 10
        self.lifespan = 0
        self.is_static = 0
        self.is_periodic = 0

        self.max_alive_cells_count = 0
        self.alive_growth = 0
        self.alive_history = [sum(self.grid)]

    def step(self):
        """
        Perform a single step (one generation) in the Game of Life.
        Applies the classic rules:
            - A living cell (1) with 2 or 3 neighbors stays alive.
            - A dead cell (0) with exactly 3 neighbors becomes alive.
            - Otherwise, the cell dies (or remains dead).
        Checks if the new grid is identical to the current grid (static),
        or matches any previous state (periodic).
        """
        cur_grid = self.grid[:]
        new_grid = [0] * (self.grid_size * self.grid_size)

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                index = x * self.grid_size + y
                alive_neighbors = self.count_alive_neighbors(x, y)
                if cur_grid[index] == 1:
                    # Survives if it has 2 or 3 neighbors
                    if alive_neighbors in [2, 3]:
                        new_grid[index] = 1
                else:
                    # A dead cell becomes alive if it has exactly 3 neighbors
                    if alive_neighbors == 3:
                        new_grid[index] = 1

        newState = tuple(new_grid)
        curState = tuple(cur_grid)

        # Static check: No change from previous generation
        if newState == curState:
            self.is_static = 1
        # Periodic check: If newState appeared before (excluding the immediate last)
        elif newState in self.history[:-1]:
            self.is_periodic = 1
        else:
            self.grid = new_grid

    def run(self):
        """
        Run the simulation until a static or periodic state is reached,
        or until we exceed game_iteration_limit. Also tracks the number
        of living cells each generation, maximum living cells, and alive growth.
        Finally, computes a 'stableness' score based on stable_count and max_stable_generations.
        """
        limiter = self.game_iteration_limit

        while limiter and ((not self.is_static and not self.is_periodic) or self.stable_count < self.max_stable_generations):
            alive_cell_count = self.get_alive_cells_count()
            # If no cells alive, mark as static
            if not alive_cell_count:
                self.is_static = 1

            self.alive_history.append(alive_cell_count)
            self.history.append(tuple(self.grid[:]))

            if self.is_periodic or self.is_static:
                self.stable_count += 1

            self.lifespan += 1
            self.step()
            limiter -= 1

        self.lifespan = len(set(self.history))
        self.max_alive_cells_count = max(self.alive_history)
        self.alive_growth = max(self.alive_history) / max(1,
                                                          min(self.alive_history)) if self.alive_history else 1
        self.stableness = self.stable_count / self.max_stable_generations

    def count_alive_neighbors(self, x, y):
        """
        Count how many neighbors of cell (x, y) are alive.

        Args:
            x (int): Row index.
            y (int): Column index.

        Returns:
            int: Number of living neighbors around (x, y).
        """
        alive = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                nx, ny = x + i, y + j
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    index = nx * self.grid_size + ny
                    alive += self.grid[index]
        return alive

    def get_alive_cells_count(self):
        """
        Returns the total number of living cells in the grid.
        """
        return sum(self.grid)

    def get_lifespan(self):
        """
        Return the total number of unique states the grid went through before stopping.
        """
        return self.lifespan

    def get_alive_history(self):
        """
        Return the list that tracks how many cells were alive at each generation.
        """
        return self.alive_history

    def reset(self):
        """
        Reset the grid to its initial state (useful for repeated experiments).
        """
        logging.debug("Resetting the grid to initial state.")
        self.grid = list(self.initial_state)
        self.history = [self.initial_state]
        self.is_static = False
        self.is_periodic = False
        self.lifespan = 0
        self.stable_count = 0
        self.alive_history = [sum(self.grid)]
