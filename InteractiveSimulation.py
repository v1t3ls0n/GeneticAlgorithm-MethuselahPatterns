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
from matplotlib.widgets import Button
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

    def __init__(self, configurations, histories, grid_size, generations_cache, mutation_rate_history,run_params):
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
        self.run_params = run_params if run_params else {}  
        self.current_config_index = 0
        self.current_generation = 0

        # Figure for the grid
        self.grid_fig, self.grid_ax = plt.subplots(figsize=(5, 5))
        self.grid_ax.set_title(f"Best Initial Configuration No {self.current_config_index + 1},  State No {self.current_generation}")
        self.grid_ax.set_xlabel("Use arrow keys:\n←/→ to move between states (days)\n↑/↓ to move between best initial configurations")


        # Another figure for the metrics
        self.stats_fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 3, figure=self.stats_fig)

        self.params_plot = self.stats_fig.add_subplot(gs[0, 0])
        self.params_plot.set_title("Run Parameters")
        self.params_plot.axis('off')  # כדי שלא יוצגו צירים מיותרים

        # Subplots for average fitness, lifespan, growth rate, alive cells, mutation rate
        self.standardized_fitness_plot = self.stats_fig.add_subplot(gs[1, 2])
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

        self.mutation_rate_plot = self.stats_fig.add_subplot(gs[1, 1])
        self.mutation_rate_plot.set_title('Mutation Rate')
        self.mutation_rate_plot.set_xlabel('Generation')
        self.mutation_rate_plot.set_ylabel('Mutation Rate')


        # Keyboard and close events
        self.grid_fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.grid_fig.canvas.mpl_connect('close_event', self.on_close)

        # Initial layout adjustment
        self.fig.tight_layout()

        # Render the GA statistics (plots)
        self.render_statistics()
        # Render the initial grid display
        self.update_grid()

        # Create the focus button inside the "params_plot" area
        self.add_focus_button()




    def add_focus_button(self):
        """
        Creates a small button in the params subplot that attempts
        to bring the figure window to the front when clicked.
        """
        # Retrieve bounding box for the 'params_plot' Axes
        left, bottom, width, height = self.params_plot.get_position().bounds

        # Define a rectangle for placing the button
        button_width = 0.2 * width
        button_height = 0.1 * height
        button_left = left + 0.75 * width
        button_bottom = bottom + 0.0 * height

        # Add a new Axes area on the main figure for the button
        self.button_ax = self.fig.add_axes([button_left, button_bottom, button_width, button_height])
        self.focus_button = Button(self.button_ax, "Focus Grid Window")

        # Connect the callback
        self.focus_button.on_clicked(self.bring_window_to_front)

    def bring_window_to_front(self, event):
        """
        Attempts to bring the current figure window to the top of the Z-order,
        typically works on the Qt backend (Qt5Agg).
        """
        try:
            self.fig.canvas.manager.window.activateWindow()
            self.fig.canvas.manager.window.raise_()
        except Exception as e:
            logging.warning(f"Could not bring window to front: {e}")

    def on_close(self, event):
        """
        Called when the user closes the window. Closes all plots and exits the program.
        """
        logging.info("Window closed. Exiting program.")
        plt.close('all')  # <--- This line closes *all* Matplotlib windows
        exit()            #       and then we exit the program.
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
        Moves to the next configuration in self.configurations, wrapping around if needed,
        and resets generation index to 0.
        """
        self.current_config_index = (self.current_config_index + 1) % len(self.configurations)
        self.current_generation = 0
        self.update_grid()

    def previous_configuration(self):
        """
        Moves to the previous configuration in self.configurations, wrapping around if needed,
        and resets generation index to 0.
        """
        self.current_config_index = (self.current_config_index - 1) % len(self.configurations)
        self.current_generation = 0
        self.update_grid()

    def next_generation(self):
        """
        Moves forward one generation in the current config's history, if available.
        """
        if self.current_generation + 1 < len(self.histories[self.current_config_index]):
            self.current_generation += 1
            self.update_grid()

    def previous_generation(self):
        """
        Moves backward one generation in the current config's history, if available.
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
        self.config_ax.set_xlabel("Arrow keys: UP/DOWN = configs, LEFT/RIGHT = gens")
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


        # Params Plot
        self.params_plot.clear()
        self.params_plot.set_title("Run Parameters")
        self.params_plot.axis('off')

        text_lines = []
        text_lines.append("Genetic Algorithm Custom Parameters used in this run:")
        for k, v in self.run_params.items():
            text_lines.append(f"• {k} = {v}")

        display_text = "\n".join(text_lines)
        self.params_plot.text(
            0.0, 1.0,
            display_text,
            transform=self.params_plot.transAxes,
            fontsize=10,
            va='top'
        )



        self.stats_fig.tight_layout()

    def run(self):
        """
        Opens the interactive matplotlib windows. The user can close them to stop the program.
        """
        logging.info("Running interactive simulation.")
        plt.show()
