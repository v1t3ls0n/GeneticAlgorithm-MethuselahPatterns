import random
import logging


class GameOfLife:
    def __init__(self, grid_size, initial_state=None):
        self.grid_size = grid_size
        self.grid = [
            0] * (grid_size * grid_size) if initial_state is None else list(initial_state)

        # Store the initial state of the grid
        self.initial_state = tuple(self.grid)
        self.history = [self.initial_state]

        # Store initial state in history
        self.game_iteration_limit = 15000
        self.stable_count = 0  # Counter for stable generations
        # Set the number of generations before it's considered static
        self.max_stable_generations = 10
        self.lifespan = 0  # Total lifespan (should start at 0)
        # Tracks if the grid has become static (tied to the state)
        self.is_static = 0
        # Tracks if the grid is repeating a cycle (tied to the state)
        self.is_periodic = 0

        self.max_alive_cells_count = 0
        self.alive_growth = 0
        # Starting with the number of alive cells in initial state
        self.alive_history = [sum(self.grid)]

    def step(self):
        """ Perform a single step in the Game of Life and update history. """
        cur_grid = self.grid[:]
        new_grid = [0] * (self.grid_size * self.grid_size)
        # Iterate over the grid to apply the Game of Life rules
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                index = x * self.grid_size + y  # Calculate index for 2D to 1D conversion
                alive_neighbors = self.count_alive_neighbors(x, y)
                if cur_grid[index] == 1:
                    if alive_neighbors in [2, 3]:
                        new_grid[index] = 1
                else:
                    if alive_neighbors == 3:
                        new_grid[index] = 1

        newState = tuple(new_grid)  # New state after the step
        curState = tuple(cur_grid)  # Current state of the grid

        # Check for static state (no change between current and previous grid)
        if newState == curState:
            self.is_static = 1
        # Check for periodicity (if the new state has appeared before)
        elif newState in self.history[:-1]:
            self.is_periodic = 1
        else:
            self.grid = new_grid

    def run(self):
        """ Run the Game of Life until static or periodic state is reached, and calculate fitness. """
        limiter = self.game_iteration_limit

        while limiter and ((not self.is_static and not self.is_periodic) or self.stable_count < self.max_stable_generations):
            alive_cell_count = self.get_alive_cells_count()
            if not alive_cell_count:
                self.is_static = 1
            self.alive_history.append(alive_cell_count)
            self.history.append(tuple(self.grid[:]))
            if self.is_periodic or self.is_static:
                self.stable_count += 1
            self.lifespan += 1  # Increment lifespan on each step
            self.step()  # Run one step of the game
            limiter -= 1

        # Update the total alive cells and alive growth
        self.lifespan = len(set(self.history))
        self.max_alive_cells_count = max(self.alive_history)
        self.alive_growth = max(self.alive_history)/max(1, min(self.alive_history)) if self.alive_history else 1
        self.stableness = self.stable_count/self.max_stable_generations
        # Log the final result
        # logging.info(f"""Inside Game Of Life Instance:
        #                 Total Alive Cells: {self.max_alive_cells_count}, Lifespan: {self.lifespan}, Alive Growth: {self.alive_growth},
        #                 game history length: {len(self.history)}, unique history states: {len(set(self.history))}""")

    def count_alive_neighbors(self, x, y):
        alive = 0
        # Check neighbors for valid indices
        for i in range(-1, 2):  # Iterating over rows
            for j in range(-1, 2):  # Iterating over columns
                if i == 0 and j == 0:
                    continue  # Skip the cell itself
                nx, ny = x + i, y + j
                # Ensure neighbor is within grid bounds
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    index = nx * self.grid_size + ny  # Convert 2D to 1D index
                    alive += self.grid[index]  # Add 1 if neighbor is alive
        return alive

    def get_alive_cells_count(self):
        return sum(self.grid)

    def get_lifespan(self):
        """ Return the lifespan of the current grid (including extra lifespan for static/periodic) """
        return self.lifespan

    def get_alive_history(self):
        """ Return the history of the number of alive cells for each generation """
        return self.alive_history

    def reset(self):
        """ Reset the grid to its initial state """
        logging.debug("""Resetting the grid to initial state.""")
        self.grid = list(self.initial_state)
        self.history = [self.initial_state]
        self.is_static = False
        self.is_periodic = False
        self.lifespan = 0
        self.stable_count = 0
        self.alive_history = [sum(self.grid)]
