"""
InteractiveSimulation.py
------------------------

Creates TWO separate windows:
  1) A "Grid Window" that displays the current Game of Life configuration (the NxN grid).
  2) A "Stats Window" that displays:
       - Standardized Fitness, Lifespan, Growth, Alive Cells, Mutation Rate
       - Run parameters text
       - A button to bring the Grid Window to the front
       
Keyboard navigation:
  - UP:    Next configuration
  - DOWN:  Previous configuration
  - RIGHT: Next generation in current config
  - LEFT:  Previous generation in current config
  
Closing either window closes ALL windows and exits the program.
"""

import logging
import matplotlib
# matplotlib.use("Qt5Agg")
matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button
import numpy as np

class InteractiveSimulation:
    """
    Displays the best 5 configurations found by the genetic algorithm in TWO separate windows:
      - A window for the Game of Life grid (to navigate generations, configs).
      - A window for statistics (fitness, lifespan, etc.) plus run parameters and a focus button.
    """

    def __init__(
        self,
        configurations,
        histories,
        grid_size,
        generations_cache,
        mutation_rate_history,
        run_params=None
    ):
        """
        Args:
            configurations (list[tuple]): Top final configurations from the GA.
            histories (list[list[tuple]]): State-history for each of those top configurations.
            grid_size (int): NxN dimension for the grid.
            generations_cache (dict): Aggregated metrics across generations.
            mutation_rate_history (list[float]): Recorded mutation rates across generations.
            run_params (dict, optional): Dictionary of run parameters to display in the stats window.
        """
        print("Initializing InteractiveSimulation with TWO windows.")
        self.configurations = configurations
        self.histories = histories
        self.grid_size = grid_size
        self.generations_cache = generations_cache
        self.mutation_rate_history = mutation_rate_history
        self.run_params = run_params if run_params else {}

        # Navigation state
        self.current_config_index = 0
        self.current_generation = 0

        # 1) Create the 'Grid Window'
        self.grid_fig = plt.figure(figsize=(5, 5))
        self.grid_ax = self.grid_fig.add_subplot(111)
        self.grid_ax.set_title("Grid Window")
        
        # Optional: connect a close event to kill the entire program
        self.grid_fig.canvas.mpl_connect("close_event", self.on_close)

        # 2) Create the 'Stats Window'
        self.stats_fig = plt.figure(figsize=(6, 4))
        self.stats_ax = self.stats_fig.add_subplot(111)
        self.stats_ax.set_title("Stats Window")
        
        # Connect a close event, too
        self.stats_fig.canvas.mpl_connect("close_event", self.on_close)

        self.button_ax = self.stats_fig.add_axes([0.2, 0.02, 0.6, 0.07])
        self.focus_button = Button(self.button_ax, "Focus Grid Window")
        self.focus_button.on_clicked(self.bring_grid_to_front)


        # Top row: Fitness, Lifespan, Growth Rate
        self.standardized_lifespan_plot    = self.stats_fig.add_subplot(gs[0, 1])
        self.standardized_growth_rate_plot = self.stats_fig.add_subplot(gs[0, 2])

        # Bottom row: Alive Cells, Mutation Rate, Run Params
        self.standardized_alive_cells_plot = self.stats_fig.add_subplot(gs[1, 0])
        self.mutation_rate_plot            = self.stats_fig.add_subplot(gs[1, 1])
        self.standardized_fitness_plot     = self.stats_fig.add_subplot(gs[1, 2])

        self.params_plot.set_title("Run Parameters")
        self.params_plot.axis("off")

        # Connect key press & close events for the stats window
        self.stats_fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.stats_fig.canvas.mpl_connect('close_event', self.on_close)

        # Render stats
        self.render_statistics()

        # Add a button in the params_plot to bring the grid window forward
        # self.add_focus_button()

        # Final layout adjustment
        # self.stats_fig.tight_layout()

    def on_close(self, event):
        """
        Called when EITHER window is closed. Closes all plots and exits the program.
        """
        logging.info("A window was closed. Exiting program.")
        plt.close('all')  # Closes all windows
        exit()

    def on_key(self, event):
        """
        Keyboard navigation for either window:
            UP -> next configuration
            DOWN -> previous configuration
            RIGHT -> next generation
            LEFT -> previous generation
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
        self.current_config_index = (self.current_config_index + 1) % len(self.configurations)
        self.current_generation = 0
        self.update_grid()

    def previous_configuration(self):
        self.current_config_index = (self.current_config_index - 1) % len(self.configurations)
        self.current_generation = 0
        self.update_grid()

    def next_generation(self):
        """
        Advance one generation in the current config's history, if available.
        """
        history_len = len(self.histories[self.current_config_index])
        if self.current_generation + 1 < history_len:
            self.current_generation += 1
            self.update_grid()

    def previous_generation(self):
        """
        Go back one generation in the current config's history, if possible.
        """
        if self.current_generation > 0:
            self.current_generation -= 1
            self.update_grid()

    def update_grid(self):
        """
        Redraw the NxN grid in the grid window based on current config/generation.
        """
        grid_2d = [
            self.histories[self.current_config_index][self.current_generation][
                i*self.grid_size : (i+1)*self.grid_size
            ]
            for i in range(self.grid_size)
        ]
        self.grid_ax.clear()
        self.grid_ax.imshow(grid_2d, cmap="binary")
        self.grid_ax.set_title(
            f"Config #{self.current_config_index+1}, Generation {self.current_generation}"
        )
        self.grid_ax.set_xlabel("ARROWS: UP/DOWN=configs, LEFT/RIGHT=gens")
        self.grid_fig.canvas.draw_idle()

    def add_focus_button(self):
        """
        Creates a small button in the 'params_plot' to bring the GRID WINDOW to the front.
        """
        left, bottom, width, height = self.params_plot.get_position().bounds
        button_width  = 0.3 * width
        button_height = 0.1 * height
        button_left   = left + 0.65 * width
        button_bottom = bottom + 0.0 * height

        self.button_ax = self.stats_fig.add_axes(
            [button_left, button_bottom, button_width, button_height]
        )
        self.focus_button = Button(self.button_ax, "Focus Grid Window")
        self.focus_button.on_clicked(self.bring_grid_to_front)

    def bring_grid_to_front(self, event):
        """
        Attempt to bring the 'Grid Window' to the front.
        This only works reliably on Qt-based backends (Qt5Agg).
        On some OS or window managers, focus stealing may be blocked.
        """
        try:
            self.grid_fig.canvas.manager.window.showNormal()
            self.grid_fig.canvas.manager.window.activateWindow()
            self.grid_fig.canvas.manager.window.raise_()
        except Exception as e:
            logging.warning(f"Could not bring the grid window to the front: {e}")



    def render_statistics(self):
        """
        Plots the GA metrics (fitness, lifespan, growth, alive cells, mutation rate)
        in the Stats Window, and displays run_params in the params_plot.
        """
        gens = list(self.generations_cache.keys())

        # ========== Fitness ==========
        avg_fitness = [self.generations_cache[g]['avg_fitness'] for g in gens]
        std_fitness = [self.generations_cache[g]['std_fitness'] for g in gens]
        self.standardized_fitness_plot.clear()
        self.standardized_fitness_plot.plot(gens, avg_fitness, label='Standardized Fitness', color='blue')
        self.standardized_fitness_plot.fill_between(
            gens,
            np.subtract(avg_fitness, std_fitness),
            np.add(avg_fitness, std_fitness),
            color='blue', alpha=0.2, label='Std Dev'
        )
        self.standardized_fitness_plot.set_title("Standardized Fitness")
        self.standardized_fitness_plot.legend()

        # ========== Lifespan ==========
        avg_lifespan = [self.generations_cache[g]['avg_lifespan'] for g in gens]
        std_lifespan = [self.generations_cache[g]['std_lifespan'] for g in gens]
        self.standardized_lifespan_plot.clear()
        self.standardized_lifespan_plot.plot(gens, avg_lifespan, label='Standardized Lifespan', color='green')
        self.standardized_lifespan_plot.fill_between(
            gens,
            np.subtract(avg_lifespan, std_lifespan),
            np.add(avg_lifespan, std_lifespan),
            color='green', alpha=0.2, label='Std Dev'
        )
        self.standardized_lifespan_plot.set_title("Standardized Lifespan")
        self.standardized_lifespan_plot.legend()

        # ========== Growth Rate ==========
        avg_growth = [self.generations_cache[g]['avg_alive_growth_rate'] for g in gens]
        std_growth = [self.generations_cache[g]['std_alive_growth_rate'] for g in gens]
        self.standardized_growth_rate_plot.clear()
        self.standardized_growth_rate_plot.plot(gens, avg_growth, label='Std Growth', color='red')
        self.standardized_growth_rate_plot.fill_between(
            gens,
            np.subtract(avg_growth, std_growth),
            np.add(avg_growth, std_growth),
            color='red', alpha=0.2, label='Std Dev'
        )
        self.standardized_growth_rate_plot.set_title("Standardized Growth Rate")
        self.standardized_growth_rate_plot.legend()

        # ========== Alive Cells ==========
        avg_alive_cells = [self.generations_cache[g]['avg_max_alive_cells_count'] for g in gens]
        std_alive_cells = [self.generations_cache[g]['std_max_alive_cells_count'] for g in gens]
        self.standardized_alive_cells_plot.clear()
        self.standardized_alive_cells_plot.plot(gens, avg_alive_cells, label='Std Alive', color='purple')
        self.standardized_alive_cells_plot.fill_between(
            gens,
            np.subtract(avg_alive_cells, std_alive_cells),
            np.add(avg_alive_cells, std_alive_cells),
            color='purple', alpha=0.2, label='Std Dev'
        )
        self.standardized_alive_cells_plot.set_title("Standardized Alive Cells")
        self.standardized_alive_cells_plot.legend()

        # ========== Mutation Rate ==========
        self.mutation_rate_plot.clear()
        self.mutation_rate_plot.plot(gens, self.mutation_rate_history, label='Mutation Rate', color='orange')
        self.mutation_rate_plot.set_title("Mutation Rate")
        self.mutation_rate_plot.legend()

        # ========== Run Parameters Text ==========
        self.params_plot.clear()
        self.params_plot.set_title("Run Parameters")
        self.params_plot.axis("off")
        lines = ["Genetic Algorithm Custom Parameters used in this run:"]
        for k, v in self.run_params.items():
            lines.append(f"• {k} = {v}")
        text_str = "\n".join(lines)
        self.params_plot.text(0.0, 1.0, text_str, transform=self.params_plot.transAxes,
                              fontsize=10, va='top')

        # self.stats_fig.tight_layout()

    def run(self):
        """
        Displays the two windows:
          - The Grid Window (grid_fig)
          - The Stats Window (stats_fig)
        Each has its own close button and arrow-key callbacks.
        """
        logging.info("Running interactive simulation with separate Grid and Stats windows.")
        # We call plt.show() once; it will display ALL open figures
        plt.show()
