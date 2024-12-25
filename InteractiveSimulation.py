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

        # Create Tkinter root window
        self.root = tk.Tk()

        # Create a frame to hold the grid and the graph
        self.frame = tk.Frame(self.root)
        self.frame.pack()

        # Create the grid canvas
        self.grid_canvas_frame = tk.Frame(self.frame)
        self.grid_canvas_frame.pack(side=tk.LEFT)

        # Create the graph canvas (for the statistics)
        self.graph_canvas_frame = tk.Frame(self.frame)
        self.graph_canvas_frame.pack(side=tk.RIGHT)

        # Create the matplotlib figure for grid and for stats
        self.fig, self.ax = plt.subplots(3, 2, figsize=(15, 15))  # 3x2 grid for the subplots
        self.ax = self.ax.flatten()  # Flatten to iterate easily

        # Create FigureCanvasTkAgg for the grid plot
        self.grid_canvas = FigureCanvasTkAgg(self.fig, master=self.grid_canvas_frame)
        self.grid_canvas.get_tk_widget().pack()

        # Create FigureCanvasTkAgg for the statistics graph
        self.graph_canvas = FigureCanvasTkAgg(self.fig, master=self.graph_canvas_frame)
        self.graph_canvas.get_tk_widget().pack()

        # Setup the first grid display
        self.update_grid()  # Initial grid display

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.update_statistics()  # Initial statistics graph rendering

        # Add scrollbar for the statistics window
        self.canvas_scrollbar = tk.Scrollbar(self.graph_canvas_frame)
        self.canvas_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas_scrollbar.config(command=self.graph_canvas.get_tk_widget().yview)

        self.root.mainloop()

    def update_grid(self):
        # Update grid with current generation from history
        grid = [
            self.histories[self.current_config_index][self.current_generation][i * self.grid_size:(i + 1) * self.grid_size]
            for i in range(self.grid_size)]
        self.ax[0].clear()  # Clear the current axis to update the grid
        self.ax[0].imshow(grid, cmap="binary")
        self.ax[0].set_title(f"""Configuration {self.current_config_index + 1}, Generation {self.current_generation}""")
        self.grid_canvas.draw_idle()

    def update_statistics(self):
        generations = list(self.generations_cache.keys())

        avg_fitness = [self.generations_cache[generation]['avg_fitness'] for generation in generations]
        std_fitness = [self.generations_cache[generation]['std_fitness'] for generation in generations]
        
        self.ax[1].clear()  # Clear the plot to avoid overlap
        self.ax[1].plot(generations, avg_fitness, label='Standardized Fitness', color='blue')
        self.ax[1].fill_between(generations, np.subtract(avg_fitness, std_fitness), np.add(avg_fitness, std_fitness), color='blue', alpha=0.2, label='Standard Deviation')
        self.ax[1].set_xlabel("Generation")
        self.ax[1].set_ylabel("Standardized Fitness")
        self.ax[1].set_title("Standardized Fitness over Generations")
        self.ax[1].legend()
        self.ax[1].grid(True)

        # Standardized Lifespan
        avg_lifespan = [self.generations_cache[generation]['avg_lifespan'] for generation in generations]
        std_lifespan = [self.generations_cache[generation]['std_lifespan'] for generation in generations]
        
        self.ax[2].clear()  # Clear the plot to avoid overlap
        self.ax[2].plot(generations, avg_lifespan, label='Standardized Lifespan', color='green')
        self.ax[2].fill_between(generations, np.subtract(avg_lifespan, std_lifespan), np.add(avg_lifespan, std_lifespan), color='green', alpha=0.2, label='Standard Deviation')
        self.ax[2].set_xlabel("Generation")
        self.ax[2].set_ylabel("Standardized Lifespan")
        self.ax[2].set_title("Standardized Lifespan over Generations")
        self.ax[2].legend()
        self.ax[2].grid(True)

        # Standardized Growth Rate
        avg_alive_growth_rate = [self.generations_cache[generation]['avg_alive_growth_rate'] for generation in generations]
        std_alive_growth_rate = [self.generations_cache[generation]['std_alive_growth_rate'] for generation in generations]
        
        self.ax[3].clear()  # Clear the plot to avoid overlap
        self.ax[3].plot(generations, avg_alive_growth_rate, label='Standardized Growth Rate', color='red')
        self.ax[3].fill_between(generations, np.subtract(avg_alive_growth_rate, std_alive_growth_rate), np.add(avg_alive_growth_rate, std_alive_growth_rate), color='red', alpha=0.2, label='Standard Deviation')
        self.ax[3].set_xlabel("Generation")
        self.ax[3].set_ylabel("Standardized Growth Rate")
        self.ax[3].set_title("Standardized Growth Rate over Generations")
        self.ax[3].legend()
        self.ax[3].grid(True)

        # Standardized Alive Cells 
        avg_total_alive_cells = [self.generations_cache[generation]['avg_total_alive_cells'] for generation in generations]
        std_total_alive_cells = [self.generations_cache[generation]['std_total_alive_cells'] for generation in generations]
        
        self.ax[4].clear()  # Clear the plot to avoid overlap
        self.ax[4].plot(generations, avg_total_alive_cells, label='Standardized Alive Cells', color='purple')
        self.ax[4].fill_between(generations, np.subtract(avg_total_alive_cells, std_total_alive_cells), np.add(avg_total_alive_cells, std_total_alive_cells), color='purple', alpha=0.2, label='Standard Deviation')
        self.ax[4].set_xlabel("Generation")
        self.ax[4].set_ylabel("Standardized Alive Cells")
        self.ax[4].set_title("Standardized Alive Cells over Generations")
        self.ax[4].legend()
        self.ax[4].grid(True)

        self.graph_canvas.draw_idle()

    def on_key(self, event):
        if event.key == 'right':
            self.next_configuration()
        elif event.key == 'left':
            self.previous_configuration()
        elif event.key == 'up':
            self.next_generation()
        elif event.key == 'down':
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

    def run(self):
        logging.info("""Running interactive simulation.""")
        self.root.mainloop()
