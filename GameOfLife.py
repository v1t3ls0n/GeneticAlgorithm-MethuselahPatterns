"""
GameOfLife.py
-------------

This module implements Conway's Game of Life, a cellular automaton, using a simulation class.
The simulation starts with an initial configuration represented as a 1D list (mapped to a 2D grid)
and evolves based on predefined rules. The simulation stops when the grid reaches a static or
periodic state, or when a maximum iteration limit is exceeded.

Classes:
    GameOfLife: Handles the simulation of the Game of Life logic, including state transitions,
    detecting static/periodic behavior, and tracking relevant statistics.
"""

import random
import logging


class GameOfLife:
    """
    The GameOfLife class simulates Conway's Game of Life on a 2D grid, given an initial state.
    It tracks the evolution of the grid through multiple generations, stopping when the grid
    becomes static (unchanging) or periodic (repeats a previously encountered state), or when
    the simulation reaches a maximum number of iterations.

    Attributes:
        grid_size (int): The dimension N of the NxN grid.
        grid (list[int]): A flat list of length N*N, representing the grid's cells (0=dead, 1=alive).
        initial_state (tuple[int]): The initial configuration of the grid (immutable).
        history (list[tuple[int]]): A record of all states encountered during the simulation.
        game_iteration_limit (int): Maximum number of generations to simulate.
        stable_count (int): Tracks consecutive generations where the state is static or periodic.
        max_stable_generations (int): Threshold for terminating the simulation if stable_count is reached.
        lifespan (int): Number of unique states the grid has passed through before stopping.
        is_static (int): Indicates whether the grid has become static (1=true, 0=false).
        is_periodic (int): Indicates whether the grid has entered a periodic cycle (1=true, 0=false).
        max_alive_cells_count (int): Maximum number of alive cells observed during the simulation.
        alive_growth (float): Ratio between the maximum and minimum alive cell counts across generations.
        alive_history (list[int]): History of alive cell counts for each generation.
        stableness (float): Ratio indicating how often the grid reached stable behavior (static or periodic).
    """

    def __init__(self, grid_size, initial_state=None):
        """
        Initializes the GameOfLife simulation with a given grid size and optional initial state.

        Args:
            grid_size (int): Size of the NxN grid.
            initial_state (Iterable[int], optional): Flat list of the grid's initial configuration.
                If None, the grid is initialized to all zeros (dead cells).

        The initial state is stored as an immutable tuple, and the simulation starts with
        an empty history except for the initial configuration.
        """
        self.grid_size = grid_size
        self.grid = [0] * (grid_size * grid_size) if initial_state is None else list(initial_state)

        # Store the immutable initial state
        self.initial_state = tuple(self.grid)
        self.history = []

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
        Executes one generation step in the Game of Life.
        Each cell's state in the next generation is determined by its neighbors:
            - A living cell (1) with 2 or 3 neighbors survives.
            - A dead cell (0) with exactly 3 neighbors becomes alive.
            - All other cells either die or remain dead.

        After computing the new grid, the method checks if the state is static
        (unchanged from the previous state) or periodic (matches a previous state
        in the simulation history, excluding the immediately prior state).
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

        new_state = tuple(new_grid)
        cur_state = tuple(cur_grid)

        # Static check: No change from the previous generation
        if new_state == cur_state:
            self.is_static = 1
        # Periodic check: If the new state matches any previous state (excluding the immediate last)
        elif new_state in self.history[:-1]:
            self.is_periodic = 1
        else:
            self.grid = new_grid

    def run(self):
        """
        Runs the Game of Life simulation until one of the following conditions is met:
            1. The grid becomes static.
            2. The grid enters a periodic cycle.
            3. The maximum number of iterations is exceeded.
            4. The stable_count exceeds max_stable_generations.

        During the simulation, the method tracks:
            - Alive cells in each generation.
            - Maximum alive cells observed.
            - Alive growth (max/min alive cells ratio).
            - Stableness (ratio of stable states to max_stable_generations).
        """
        limiter = self.game_iteration_limit

        while limiter and ((not self.is_static and not self.is_periodic) or self.stable_count < self.max_stable_generations):
            alive_cell_count = sum(self.grid)
            # If no cells are alive, mark as static
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
        self.alive_growth = max(self.alive_history) / max(1, min(self.alive_history)) if self.alive_history else 1
        self.stableness = self.stable_count / self.max_stable_generations

    def count_alive_neighbors(self, x, y):
        """
        Counts the number of alive neighbors for a cell at position (x, y).

        Args:
            x (int): The row index of the cell.
            y (int): The column index of the cell.

        Returns:
            int: The total number of alive neighbors surrounding the cell.
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

