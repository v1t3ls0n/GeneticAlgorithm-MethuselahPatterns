"""
InteractiveSimulation.py
------------------------

Provides an interactive visualization of:
1. The top evolved Game of Life configurations found by the GA (allowing the user to iterate through generations).
2. Graphical plots of the statistical metrics (fitness, lifespan, growth, alive cells, mutation rate) over GA generations.
"""

import logging
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

class InteractiveSimulation:
    """
    Displays the best 5 configurations found by the genetic algorithm in an interactive
    viewer, and also plots the evolution of various statistics across generations.

    Attributes:
        configurations (list[tuple[int]]): The top final configurations.
        histories (list[list[tuple[int]]]): The corresponding list of states (one list of states per config).
        grid_size (int): The dimension N of each NxN grid.
        generations_cache (dict): Contains average & std metrics across all generations.
        mutation_rate_history (list): Stores mutation rates across generations.
        current_config_index (int): Which of the best 5 configurations is currently displayed.
        current_generation (int): Which generation in that config's history is shown.
    """

    def __init__(self, configurations, histories, grid_size, generations_cache, mutation_rate_history):
        """
        Initialize figures for grid visualization and for the statistic plots.
        Set up keyboard callbacks for user interaction.

        Args:
            configurations (list): Top final configurations from the GA.
            histories (list): The state history for each of those top configurations.
            grid_size (int): NxN dimension for rendering the grid.
            generations_cache (dict): Aggregated metrics (fitness, lifespan, etc.).
            mutation_rate_history (list): Recorded mutation rates across generations.
        """
        print("Initializing Interactive Simulation and Metrics.")
        
        self.configurations = configurations
        self.histories = histories
        self.grid_size = grid_size
        self.generations_cache = generations_cache
        self.mutation_rate_history = mutation_rate_history
        self.current_config_index = 0
        self.current_generation = 0

        # Figure for the grid
        self.grid_fig, self.grid_ax = plt.subplots(figsize=(5, 5))
        self.grid_ax.set_title(f"Best Initial Configuration No {self.current_config_index + 1},  State No {self.current_generation}")
        self.grid_ax.set_xlabel("Use arrow keys:\n←/→ to move between states (days)\n↑/↓ to move between best initial configurations")

        self.update_grid()

        # Another figure for the metrics
        self.stats_fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 3, figure=self.stats_fig)

        # Subplots for average fitness, lifespan, growth rate, alive cells, mutation rate
        self.standardized_fitness_plot = self.stats_fig.add_subplot(gs[0, 0])
        self.standardized_fitness_plot.set_title('Standardized Fitness')
        self.standardized_fitness_plot.set_xlabel('Generation')
        self.standardized_fitness_plot.set_ylabel('Standardized Fitness')

        self.standardized_lifespan_plot = self.stats_fig.add_subplot(gs[0, 1])
        self.standardized_lifespan_plot.set_title('Standardized Lifespan')
        self.standardized_lifespan_plot.set_xlabel('Generation')
        self.standardized_lifespan_plot.set_ylabel('Standardized Lifespan')

        self.standardized_growth_rate_plot = self.stats_fig.add_subplot(gs[0, 2])
        self.standardized_growth_rate_plot.set_title('Standardized Growth Rate')
        self.standardized_growth_rate_plot.set_xlabel('Generation')
        self.standardized_growth_rate_plot.set_ylabel('Standardized Growth Rate')

        self.standardized_alive_cells_plot = self.stats_fig.add_subplot(gs[1, 0])
        self.standardized_alive_cells_plot.set_title('Standardized Alive Cells')
        self.standardized_alive_cells_plot.set_xlabel('Generation')
        self.standardized_alive_cells_plot.set_ylabel('Standardized Alive Cells')

        self.mutation_rate_plot = self.stats_fig.add_subplot(gs[1, 1:])
        self.mutation_rate_plot.set_title('Mutation Rate')
        self.mutation_rate_plot.set_xlabel('Generation')
        self.mutation_rate_plot.set_ylabel('Mutation Rate')

        # Keyboard and close events
        self.grid_fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.grid_fig.canvas.mpl_connect('close_event', self.on_close)
        self.stats_fig.canvas.mpl_connect('close_event', self.on_close)

        self.grid_fig.tight_layout()
        self.stats_fig.tight_layout()
        self.render_statistics()

    def on_close(self, event):
        """
        Called when the user closes the window. Closes all plots and exits the program.
        """
        logging.info("Window closed. Exiting program.")
        plt.close(self.grid_fig)
        plt.close(self.stats_fig)
        exit()

    def on_key(self, event):
        """
        Handles keyboard events:
            - UP: go to next configuration
            - DOWN: go to previous configuration
            - RIGHT: go to the next generation within the current configuration
            - LEFT: go to the previous generation
        """
        if event.key == 'up':
            self.next_configuration()
        elif event.key == 'down':
            self.previous_configuration()
        elif event.key == 'right':
            self.next_generation()
        elif event.key == 'left':
            self.previous_generation()

    def next_configuration(self):
        """
        Cycle forward among the top 5 best configurations.
        Resets the generation index to 0 upon changing configuration.
        """
        self.current_config_index = (self.current_config_index + 1) % len(self.configurations)
        self.current_generation = 0
        self.update_grid()

    def previous_configuration(self):
        """
        Cycle backward among the top 5 best configurations.
        Resets the generation index to 0 upon changing configuration.
        """
        self.current_config_index = (self.current_config_index - 1) % len(self.configurations)
        self.current_generation = 0
        self.update_grid()

    def next_generation(self):
        """
        Move to the next generation (if any) within the current configuration's evolution history.
        """
        if self.current_generation + 1 < len(self.histories[self.current_config_index]):
            self.current_generation += 1
            self.update_grid()

    def previous_generation(self):
        """
        Move to the previous generation (if any) within the current configuration's evolution history.
        """
        if self.current_generation > 0:
            self.current_generation -= 1
            self.update_grid()

    def update_grid(self):
        """
        Redraw the NxN grid for the current configuration index and generation index.
        """
        grid = [
            self.histories[self.current_config_index][self.current_generation][i*self.grid_size : (i+1)*self.grid_size]
            for i in range(self.grid_size)
        ]
        self.grid_ax.clear()
        self.grid_ax.imshow(grid, cmap="binary")
        self.grid_ax.set_title(f"Configuration {self.current_config_index + 1}, Generation {self.current_generation}")
        self.grid_ax.set_xlabel("Use arrow keys:\n←/→ to move between states (days)\n↑/↓ to move between best initial configurations")
        self.grid_fig.canvas.draw()

    def render_statistics(self):
        """
        Generate plots for average fitness, lifespan, growth rate, alive cells, and mutation rate
        across the generations as stored in generations_cache and mutation_rate_history.
        """
        generations = list(self.generations_cache.keys())

        # Standardized Fitness
        avg_fitness = [self.generations_cache[g]['avg_fitness'] for g in generations]
        std_fitness = [self.generations_cache[g]['std_fitness'] for g in generations]

        self.standardized_fitness_plot.clear()
        self.standardized_fitness_plot.plot(generations, avg_fitness, label='Standardized Fitness', color='blue')
        self.standardized_fitness_plot.fill_between(generations,
                                                    np.subtract(avg_fitness, std_fitness),
                                                    np.add(avg_fitness, std_fitness),
                                                    color='blue', alpha=0.2, label='Standard Deviation')
        self.standardized_fitness_plot.set_title("Standardized Fitness over Generations")
        self.standardized_fitness_plot.set_xlabel("Generation")
        self.standardized_fitness_plot.set_ylabel("Standardized Fitness")
        self.standardized_fitness_plot.legend()

        # Standardized Lifespan
        avg_lifespan = [self.generations_cache[g]['avg_lifespan'] for g in generations]
        std_lifespan = [self.generations_cache[g]['std_lifespan'] for g in generations]

        self.standardized_lifespan_plot.clear()
        self.standardized_lifespan_plot.plot(generations, avg_lifespan, label='Standardized Lifespan', color='green')
        self.standardized_lifespan_plot.fill_between(generations,
                                                     np.subtract(avg_lifespan, std_lifespan),
                                                     np.add(avg_lifespan, std_lifespan),
                                                     color='green', alpha=0.2, label='Standard Deviation')
        self.standardized_lifespan_plot.set_title("Standardized Lifespan over Generations")
        self.standardized_lifespan_plot.set_xlabel("Generation")
        self.standardized_lifespan_plot.set_ylabel("Standardized Lifespan")
        self.standardized_lifespan_plot.legend()

        # Standardized Growth Rate
        avg_alive_growth_rate = [self.generations_cache[g]['avg_alive_growth_rate'] for g in generations]
        std_alive_growth_rate = [self.generations_cache[g]['std_alive_growth_rate'] for g in generations]

        self.standardized_growth_rate_plot.clear()
        self.standardized_growth_rate_plot.plot(generations, avg_alive_growth_rate, label='Standardized Growth Rate', color='red')
        self.standardized_growth_rate_plot.fill_between(generations,
                                                        np.subtract(avg_alive_growth_rate, std_alive_growth_rate),
                                                        np.add(avg_alive_growth_rate, std_alive_growth_rate),
                                                        color='red', alpha=0.2, label='Standard Deviation')
        self.standardized_growth_rate_plot.set_title("Standardized Growth Rate over Generations")
        self.standardized_growth_rate_plot.set_xlabel("Generation")
        self.standardized_growth_rate_plot.set_ylabel("Standardized Growth Rate")
        self.standardized_growth_rate_plot.legend()

        # Standardized Alive Cells
        avg_max_alive_cells_count = [self.generations_cache[g]['avg_max_alive_cells_count'] for g in generations]
        std_max_alive_cells_count = [self.generations_cache[g]['std_max_alive_cells_count'] for g in generations]

        self.standardized_alive_cells_plot.clear()
        self.standardized_alive_cells_plot.plot(generations, avg_max_alive_cells_count, label='Standardized Alive Cells', color='purple')
        self.standardized_alive_cells_plot.fill_between(generations,
                                                        np.subtract(avg_max_alive_cells_count, std_max_alive_cells_count),
                                                        np.add(avg_max_alive_cells_count, std_max_alive_cells_count),
                                                        color='purple', alpha=0.2, label='Standard Deviation')
        self.standardized_alive_cells_plot.set_title("Standardized Alive Cells over Generations")
        self.standardized_alive_cells_plot.set_xlabel("Generation")
        self.standardized_alive_cells_plot.set_ylabel("Standardized Alive Cells")
        self.standardized_alive_cells_plot.legend()

        # Mutation Rate
        self.mutation_rate_plot.clear()
        self.mutation_rate_plot.plot(generations, self.mutation_rate_history, label='Mutation Rate', color='orange')
        self.mutation_rate_plot.set_title("Mutation Rate over Generations")
        self.mutation_rate_plot.set_xlabel("Generation")
        self.mutation_rate_plot.set_ylabel("Mutation Rate")
        self.mutation_rate_plot.legend()

        self.stats_fig.tight_layout()

    def run(self):
        """
        Opens the interactive matplotlib windows. The user can close them to stop the program.
        """
        logging.info("Running interactive simulation.")
        plt.show()
