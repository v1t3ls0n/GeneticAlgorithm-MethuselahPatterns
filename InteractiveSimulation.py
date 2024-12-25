import logging
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class InteractiveSimulation:
    def __init__(self, configurations, histories, grid_size, generations_cache):
        logging.info("""Initializing InteractiveSimulation.""")
        self.configurations = configurations
        self.histories = histories
        self.grid_size = grid_size
        self.generations_cache = generations_cache
        self.current_config_index = 0
        self.current_generation = 0

        # Set up the root window with Tkinter
        self.root = tk.Tk()
        self.root.title("Interactive Simulation")

        # Create a frame for the grid and graphs
        self.grid_frame = tk.Frame(self.root)
        self.grid_frame.pack(side=tk.LEFT, padx=10)

        self.graph_frame = tk.Frame(self.root)
        self.graph_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)

        # Create the grid and initialize it
        self.grid_fig, self.grid_ax = plt.subplots(figsize=(5, 5))
        self.update_grid()

        # Set up the canvas for the grid figure
        self.grid_canvas = FigureCanvasTkAgg(self.grid_fig, self.grid_frame)
        self.grid_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Create a scrollable canvas for the graphs
        self.graph_canvas = tk.Canvas(self.graph_frame)
        self.graph_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.graph_scrollbar = tk.Scrollbar(self.graph_frame, orient="vertical", command=self.graph_canvas.yview)
        self.graph_scrollbar.pack(side=tk.RIGHT, fill="y")

        self.graph_canvas.config(yscrollcommand=self.graph_scrollbar.set)

        self.graph_window = tk.Frame(self.graph_canvas)
        self.graph_canvas.create_window((0, 0), window=self.graph_window, anchor="nw")

        # Create the figure for the stats and attach it to the graph window
        self.stats_fig, self.stats_ax = plt.subplots(3, 2, figsize=(15, 15))  # 3x2 grid for the subplots
        self.stats_ax = self.stats_ax.flatten()  # Flatten to iterate easily

        self.standardized_fitness_plot, = self.stats_ax[0].plot([], [], label='Standardized Fitness', color='blue')
        self.standardized_lifespan_plot, = self.stats_ax[1].plot([], [], label='Standardized Lifespan', color='green')
        self.standardized_growth_rate_plot, = self.stats_ax[2].plot([], [], label='Standardized Growth Rate', color='red')
        self.standardized_alive_cells_plot, = self.stats_ax[3].plot([], [], label='Standardized Alive Cells', color='purple')

        # Render the statistics
        self.render_statistics()

        # Attach key press events to control grid navigation
        self.root.bind('<KeyPress>', self.on_key)

        # Adjust layout and add padding for scrollable area
        self.graph_window.update_idletasks()
        self.graph_canvas.config(scrollregion=self.graph_canvas.bbox("all"))

    def on_key(self, event):
        if event.keycode == 39:  # Right Arrow Key
            self.next_configuration()
        elif event.keycode == 37:  # Left Arrow Key
            self.previous_configuration()
        elif event.keycode == 38:  # Up Arrow Key
            self.next_generation()
        elif event.keycode == 40:  # Down Arrow Key
            self.previous_generation()

    def next_configuration(self):
        logging.info(f"""Switching to next configuration. Current index: {self.current_config_index}""")
        self.current_config_index = (self.current_config_index + 1) % len(self.configurations)
        self.current_generation = 0
        self.update_grid()

    def previous_configuration(self):
        logging.info(f"""Switching to previous configuration. Current index: {self.current_config_index}""")
        self.current_config_index = (self.current_config_index - 1) % len(self.configurations)
        self.current_generation = 0
        self.update_grid()

    def next_generation(self):
        if self.current_generation + 1 < len(self.histories[self.current_config_index]):
            logging.info(f"""Switching to next generation. Current generation: {self.current_generation}""")
            self.current_generation += 1
            self.update_grid()

    def previous_generation(self):
        if self.current_generation > 0:
            logging.info(f"""Switching to previous generation. Current generation: {self.current_generation}""")
            self.current_generation -= 1
            self.update_grid()

    def update_grid(self):
        # Update grid with current generation from history
        grid = [
            self.histories[self.current_config_index][self.current_generation][i * self.grid_size:(i + 1) * self.grid_size]
            for i in range(self.grid_size)]
        self.grid_ax.clear()  # Clear the current axis to update the grid
        self.grid_ax.imshow(grid, cmap="binary")
        self.grid_ax.set_title(f"""Configuration {self.current_config_index + 1}, Generation {self.current_generation}""")
        self.grid_canvas.draw()

    def render_statistics(self):
        """
        Render relevant statistics as graphs from generations_cache with shaded areas for standard deviation.
        """
        generations = list(self.generations_cache.keys())

        # Average fitness over generations
        avg_fitness = [self.generations_cache[generation]['avg_fitness'] for generation in generations]
        std_fitness = [self.generations_cache[generation]['std_fitness'] for generation in generations]
        
        self.stats_ax[0].clear()  # Clear the plot to avoid overlap
        self.stats_ax[0].plot(generations, avg_fitness, label='Standardized Fitness', color='blue')
        self.stats_ax[0].fill_between(generations, np.subtract(avg_fitness, std_fitness), np.add(avg_fitness, std_fitness), color='blue', alpha=0.2, label='Standard Deviation')
        self.stats_ax[0].set_xlabel("Generation")
        self.stats_ax[0].set_ylabel("Standardized Fitness")
        self.stats_ax[0].set_title("Standardized Fitness over Generations")
        self.stats_ax[0].legend()
        self.stats_ax[0].grid(True)

        # Standardized Lifespan
        avg_lifespan = [self.generations_cache[generation]['avg_lifespan'] for generation in generations]
        std_lifespan = [self.generations_cache[generation]['std_lifespan'] for generation in generations]
        
        self.stats_ax[1].clear()  # Clear the plot to avoid overlap
        self.stats_ax[1].plot(generations, avg_lifespan, label='Standardized Lifespan', color='green')
        self.stats_ax[1].fill_between(generations, np.subtract(avg_lifespan, std_lifespan), np.add(avg_lifespan, std_lifespan), color='green', alpha=0.2, label='Standard Deviation')
        self.stats_ax[1].set_xlabel("Generation")
        self.stats_ax[1].set_ylabel("Standardized Lifespan")
        self.stats_ax[1].set_title("Standardized Lifespan over Generations")
        self.stats_ax[1].legend()
        self.stats_ax[1].grid(True)

        # Standardized Growth Rate
        avg_alive_growth_rate = [self.generations_cache[generation]['avg_alive_growth_rate'] for generation in generations]
        std_alive_growth_rate = [self.generations_cache[generation]['std_alive_growth_rate'] for generation in generations]
        
        self.stats_ax[2].clear()  # Clear the plot to avoid overlap
        self.stats_ax[2].plot(generations, avg_alive_growth_rate, label='Standardized Growth Rate', color='red')
        self.stats_ax[2].fill_between(generations, np.subtract(avg_alive_growth_rate, std_alive_growth_rate), np.add(avg_alive_growth_rate, std_alive_growth_rate), color='red', alpha=0.2, label='Standard Deviation')
        self.stats_ax[2].set_xlabel("Generation")
        self.stats_ax[2].set_ylabel("Standardized Growth Rate")
        self.stats_ax[2].set_title("Standardized Growth Rate over Generations")
        self.stats_ax[2].legend()
        self.stats_ax[2].grid(True)

        # Standardized Alive Cells 
        avg_total_alive_cells = [self.generations_cache[generation]['avg_total_alive_cells'] for generation in generations]
        std_total_alive_cells = [self.generations_cache[generation]['std_total_alive_cells'] for generation in generations]
        
        self.stats_ax[3].clear()  # Clear the plot to avoid overlap
        self.stats_ax[3].plot(generations, avg_total_alive_cells, label='Standardized Alive Cells', color='purple')
        self.stats_ax[3].fill_between(generations, np.subtract(avg_total_alive_cells, std_total_alive_cells), np.add(avg_total_alive_cells, std_total_alive_cells), color='purple', alpha=0.2, label='Standard Deviation')
        self.stats_ax[3].set_xlabel("Generation")
        self.stats_ax[3].set_ylabel("Standardized Alive Cells")
        self.stats_ax[3].set_title("Standardized Alive Cells over Generations")
        self.stats_ax[3].legend()
        self.stats_ax[3].grid(True)

        # Adjust layout and display the graphs
        self.stats_fig.tight_layout()

    def run(self):
        logging.info("""Running interactive simulation.""")
        self.root.mainloop()  # Start the Tkinter main loop
