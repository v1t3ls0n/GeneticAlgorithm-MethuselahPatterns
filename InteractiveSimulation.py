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

        # Create Tkinter root window (this is only initialized)
        self.root = tk.Tk()

        # Create a frame to hold the grid and the graph
        self.frame = tk.Frame(self.root)
        self.frame.pack()

        # Initialize the grid and statistics (separate windows)
        self.setup_grid()
        self.update_grid()
        self.setup_statistics()
        self.render_statistics()

    def setup_grid(self):
        """Initialize the grid plot in a separate window."""
        self.grid_canvas_frame = tk.Frame(self.frame)
        self.grid_canvas_frame.pack(side=tk.LEFT)

        # Create FigureCanvasTkAgg for the grid plot
        self.grid_fig, self.grid_ax = plt.subplots(figsize=(5, 5))  # Single plot for the grid
        self.grid_canvas = FigureCanvasTkAgg(self.grid_fig, master=self.grid_canvas_frame)
        self.grid_canvas.get_tk_widget().pack()

        # Attach key press events to control grid navigation
        self.grid_fig.canvas.mpl_connect('key_press_event', self.on_key)

        # Adjust layout
        self.grid_fig.tight_layout()

    def setup_statistics(self):
        """Initialize the statistics plot in a separate window."""
        self.graph_canvas_frame = tk.Frame(self.frame)
        self.graph_canvas_frame.pack(side=tk.LEFT)

        # Create FigureCanvasTkAgg for the statistics graph
        self.stats_fig, self.stats_ax = plt.subplots(3, 2, figsize=(15, 15))  # 3x2 grid for the subplots
        self.stats_ax = self.stats_ax.flatten()  # Flatten to iterate easily
        self.stats_canvas = FigureCanvasTkAgg(self.stats_fig, master=self.graph_canvas_frame)
        self.stats_canvas.get_tk_widget().pack()

        # Set up standardized plot references for metrics
        self.standardized_fitness_plot, = self.stats_ax[0].plot([], [], label='Standardized Fitness', color='blue')
        self.standardized_lifespan_plot, = self.stats_ax[1].plot([], [], label='Standardized Lifespan', color='green')
        self.standardized_growth_rate_plot, = self.stats_ax[2].plot([], [], label='Standardized Growth Rate', color='red')
        self.standardized_alive_cells_plot, = self.stats_ax[3].plot([], [], label='Standardized Alive Cells', color='purple')

        # Adjust layout
        self.stats_fig.tight_layout()

    def on_key(self, event):
        """Handles key presses for grid navigation."""
        if event.key == 'right':
            self.next_configuration()
        elif event.key == 'left':
            self.previous_configuration()
        elif event.key == 'up':
            self.next_generation()
        elif event.key == 'down':
            self.previous_generation()

    def next_configuration(self):
        """Go to the next configuration."""
        logging.info(f"""Switching to next configuration. Current index: {self.current_config_index}""")
        self.current_config_index = (self.current_config_index + 1) % len(self.configurations)
        self.current_generation = 0
        self.update_grid()

    def previous_configuration(self):
        """Go to the previous configuration."""
        logging.info(f"""Switching to previous configuration. Current index: {self.current_config_index}""")
        self.current_config_index = (self.current_config_index - 1) % len(self.configurations)
        self.current_generation = 0
        self.update_grid()

    def next_generation(self):
        """Go to the next generation."""
        if self.current_generation + 1 < len(self.histories[self.current_config_index]):
            logging.info(f"""Switching to next generation. Current generation: {self.current_generation}""")
            self.current_generation += 1
            self.update_grid()

    def previous_generation(self):
        """Go to the previous generation."""
        if self.current_generation > 0:
            logging.info(f"""Switching to previous generation. Current generation: {self.current_generation}""")
            self.current_generation -= 1
            self.update_grid()

    def update_grid(self):
        """Update the grid with current generation data."""
        grid = [
            self.histories[self.current_config_index][self.current_generation][i * self.grid_size:(i + 1) * self.grid_size]
            for i in range(self.grid_size)]
        self.grid_ax.clear()  # Clear the current axis to update the grid
        self.grid_ax.imshow(grid, cmap="binary")
        self.grid_ax.set_title(f"""Configuration {self.current_config_index + 1}, Generation {self.current_generation}""")
        self.grid_canvas.draw_idle()

    def render_statistics(self):
        """Render the statistics as graphs using precomputed values from generations_cache."""
        generations = list(self.generations_cache.keys())

        # Fetch data from generations_cache
        avg_fitness = [self.generations_cache[generation]['avg_fitness'] for generation in generations]
        std_fitness = [self.generations_cache[generation]['std_fitness'] for generation in generations]
        
        # Standardized values
        std_fitness_values = [self.generations_cache[generation]['std_fitness_values'] for generation in generations]
        
        # Clear the plot to avoid overlap before updating it
        self.stats_ax[0].clear()
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
        std_lifespan_values = [self.generations_cache[generation]['std_lifespan_values'] for generation in generations]
        
        self.stats_ax[1].clear()
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
        std_alive_growth_rate_values = [self.generations_cache[generation]['std_alive_growth_rate_values'] for generation in generations]
        
        self.stats_ax[2].clear()
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
        std_total_alive_cells_values = [self.generations_cache[generation]['std_total_alive_cells_values'] for generation in generations]
        
        self.stats_ax[3].clear()
        self.stats_ax[3].plot(generations, avg_total_alive_cells, label='Standardized Alive Cells', color='purple')
        self.stats_ax[3].fill_between(generations, np.subtract(avg_total_alive_cells, std_total_alive_cells), np.add(avg_total_alive_cells, std_total_alive_cells), color='purple', alpha=0.2, label='Standard Deviation')
        self.stats_ax[3].set_xlabel("Generation")
        self.stats_ax[3].set_ylabel("Standardized Alive Cells")
        self.stats_ax[3].set_title("Standardized Alive Cells over Generations")
        self.stats_ax[3].legend()
        self.stats_ax[3].grid(True)

        # Adjust layout and display the graphs
        self.stats_fig.tight_layout(pad=4.0)  # Increased padding to avoid overlap
        self.stats_fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # Adjust margins for clarity
        self.stats_canvas.draw_idle()  # Update the statistics canvas

    def on_close(self):
        logging.info("""Closing the simulation.""")
        self.root.quit()

    def run(self):
        logging.info("""Running interactive simulation.""")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)  # Handle closing event
        self.root.mainloop()  # Start the Tkinter event loop
