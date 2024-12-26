import logging
import matplotlib
matplotlib.use("Qt5Agg")  # Ensure Qt5Agg is available
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button
from PyQt5.QtWidgets import QPushButton

import numpy as np

class InteractiveSimulation:
    """
    Two-window layout:
      - Window 1 (Grid Window): shows the NxN grid.
      - Window 2 (Stats Window): shows subplots for fitness, lifespan, etc.,
        plus run parameters, plus a "Focus Grid Window" button at the bottom.
    """

    def __init__(
        self,
        configurations,
        histories,
        grid_size,
        generations_cache,
        mutation_rate_history,
        best_params,
        run_params=None
    ):
        print("Initializing InteractiveSimulation with TWO windows.")
        self.configurations = configurations
        self.histories = histories
        self.grid_size = grid_size
        self.generations_cache = generations_cache
        self.mutation_rate_history = mutation_rate_history
        self.run_params = run_params or {}
        self.best_params = best_params or []

        # Navigation state
        self.current_config_index = 0
        self.current_generation = 0

        # 1) Create the separate "Grid Window"
        self.grid_fig = plt.figure(figsize=(6, 8))
        self.grid_ax = self.grid_fig.add_subplot(111)
        self.grid_ax.set_title("Grid Window")

        # If user closes the grid window => close everything
        self.grid_fig.canvas.mpl_connect("close_event", self.on_close)

        # 2) Create the "Stats Window"
        self.stats_fig = plt.figure(figsize=(18, 6))
        # self.stats_fig.suptitle("Stats Window")

        # We'll use a 3-row × 3-column GridSpec:
        #   Row 0:    params_plot,  lifespan_plot,   growth_plot
        #   Row 1:    alive_plot,   mutation_plot,   fitness_plot
        #   Row 2:    entire row for the "Focus Grid Window" button
        gs = GridSpec(2, 3, figure=self.stats_fig)

        # Connect close event => kill entire app if user closes stats
        self.stats_fig.canvas.mpl_connect("close_event", self.on_close)

        # Create subplots in row 0
        self.params_plot = self.stats_fig.add_subplot(gs[0, 0])
        self.standardized_lifespan_plot = self.stats_fig.add_subplot(gs[0, 1])
        self.standardized_growth_rate_plot = self.stats_fig.add_subplot(gs[0, 2])

        # Create subplots in row 1
        self.standardized_alive_cells_plot = self.stats_fig.add_subplot(gs[1, 0])
        self.mutation_rate_plot = self.stats_fig.add_subplot(gs[1, 1])
        self.standardized_fitness_plot = self.stats_fig.add_subplot(gs[1, 2])

        # Row 2 for the button: single Axes spanning all 3 columns
        # self.button_ax = self.stats_fig.add_subplot(gs[2, :])
        # Turn off the spines, ticks, etc. so we see only the button
        # self.button_ax.axis("off")

        # self.focus_button = Button(self.button_ax, "Focus Grid Window")
        # self.focus_button.on_clicked(self.bring_grid_to_front)

        # Also connect arrow-key events in the stats figure
        self.grid_fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.stats_fig.canvas.mpl_connect('key_press_event', self.on_key)


        self.add_focus_button_to_toolbar(
            figure=self.stats_fig,
            button_text="Focus Grid Window",
            on_click=self.bring_grid_to_front
        )

        self.add_focus_button_to_toolbar(
            figure=self.grid_fig,
            button_text="Focus Metrics Window",
            on_click=self.bring_metrics_to_front
        )
        self.update_grid()
        self.render_statistics()




    def on_close(self, event):
        """
        Called when EITHER window is closed. Closes all plots and exits.
        """
        logging.info("A window was closed. Exiting program.")
        plt.close('all')
        exit()
    
    def add_focus_button_to_toolbar(self, figure, button_text, on_click):
        """
        Insert a custom button in the given figure's Qt toolbar.
        """
        try:
            toolbar = figure.canvas.manager.toolbar
            button = QPushButton(button_text)
            button.setStyleSheet(
                """
                QPushButton {
                    margin: 8px;        /* space around the button in the toolbar */
                    padding: 6px 10px;  /* inside spacing around the text */
                    font-size: 12px;    /* bigger font */
                    font-weight: bold;  /* make it stand out */
                }
                """
            )
            toolbar.addWidget(button)
            button.clicked.connect(on_click)
        except Exception as e:
            logging.warning(f"Could not add custom button to {button_text} in toolbar: {e}")

    def bring_grid_to_front(self, e=None):
        """
        Attempt to bring the 'Grid Window' to the front (Qt-based).
        Some OS/WM can block focus-stealing, so this may not always succeed.
        """
        try:
            # self.stats_fig.canvas.manager.window.lower()
            self.grid_fig.canvas.manager.window.activateWindow()
            self.grid_fig.canvas.manager.window.raise_()
            # self.grid_fig.canvas.manager.window.showNormal()
            self.grid_fig.canvas.manager.window.showMaximized()

        except Exception as e:
            logging.warning(f"Could not bring the Grid window to the front: {e}")


    def bring_metrics_to_front(self, e=None):
        """
        Attempt to bring the 'Metrics Window' to the front (Qt-based).
        Some OS/WM can block focus-stealing, so this may not always succeed.
        """
        try:
            # self.grid_fig.canvas.manager.window.lower()
            self.stats_fig.canvas.manager.window.showNormal()
            self.stats_fig.canvas.manager.window.activateWindow()
            self.stats_fig.canvas.manager.window.raise_()
            # self.stats_fig.canvas.manager.window.showMaximized()

            # self.grid_fig.canvas.manager.window.showMinimized()

        except Exception as e:
            logging.warning(f"Could not bring the Stats window to the front: {e}")



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
        hist_len = len(self.histories[self.current_config_index])
        if self.current_generation + 1 < hist_len:
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
        Redraw the NxN grid in the "Grid Window" for the current config/generation.
        """
        grid_2d = [
            self.histories[self.current_config_index][self.current_generation][
                i * self.grid_size:(i+1) * self.grid_size
            ]
            for i in range(self.grid_size)
        ]
        self.grid_ax.clear()
        self.grid_ax.imshow(grid_2d, cmap="binary")
        self.grid_ax.set_title(
            f"Best Initial Config #{self.current_config_index + 1}, Day {self.current_generation}"
        )
        self.grid_ax.set_ylabel("ARROWS: UP/DOWN=configs, LEFT/RIGHT=gens")

        param_dict = self.best_params[self.current_config_index]
        lifespan = param_dict.get('lifespan', 0)
        max_alive = param_dict.get('max_alive_cells_count', 0)
        growth = param_dict.get('alive_growth', 1.0)
        stableness = param_dict.get('stableness', 0.0)
        text_str = (f"lifespan={lifespan} | "
                        f"max_alive={max_alive} | "
                        f"growth={growth:.2f} | "
                        f"stableness={stableness:.2f}")
        self.grid_ax.set_xlabel(text_str)
        


        self.grid_fig.canvas.draw_idle()
 
    def render_statistics(self):
        """
        Fill in each subplot with the relevant data, including the run_params in self.params_plot.
        """
        gens = list(self.generations_cache.keys())

        # ---------------- Fitness ----------------
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

        # ---------------- Lifespan ----------------
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

        # ---------------- Growth Rate ----------------
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

        # ---------------- Alive Cells ----------------
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

        # ---------------- Mutation Rate ----------------
        self.mutation_rate_plot.clear()
        self.mutation_rate_plot.plot(gens, self.mutation_rate_history, label='Mutation Rate', color='orange')
        self.mutation_rate_plot.set_title("Mutation Rate")
        self.mutation_rate_plot.legend()

        # ---------------- Params (text) ----------------
        self.params_plot.clear()
        self.params_plot.set_title("Run Parameters")
        self.params_plot.axis("off")

        lines = ["Genetic Algorithm Custom Parameters used in this run:"]
        for k, v in self.run_params.items():
            lines.append(f"• {k} = {v}")
        text_str = "\n".join(lines)

        self.params_plot.text(
            0.0, 1.0, 
            text_str, 
            transform=self.params_plot.transAxes,
            fontsize=10, 
            va='top'
        )

        # Optionally adjust spacing:
        self.stats_fig.tight_layout()

    def run(self):
        """
        Show both windows at once.  plt.show() blocks until user closes them.
        """


        # self.grid_fig.tight_layout(pad=1.0, h_pad=3.0, w_pad=1.0)
        # self.stats_fig.tight_layout(pad=1.0, h_pad=3.0, w_pad=1.0)

        logging.info("Running interactive simulation with separate Grid and Stats windows.")
        plt.show()
        
        self.grid_fig.canvas.manager.window.activateWindow()
        self.grid_fig.canvas.manager.window.raise_()



