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

        # Create the grid and the graphs as separate canvases
        self.grid_canvas_frame = tk.Frame(self.frame)
        self.grid_canvas_frame.pack(side=tk.LEFT)

        self.graph_canvas_frame = tk.Frame(self.frame)
        self.graph_canvas_frame.pack(side=tk.RIGHT)

        # Create the matplotlib figure and axes
        self.fig, self.ax = plt.subplots(3, 2, figsize=(15, 15))  # 3x2 grid for the subplots
        self.ax = self.ax.flatten()  # Flatten to iterate easily
        
        # Create a FigureCanvasTkAgg for the grid plot
        self.grid_canvas = FigureCanvasTkAgg(self.fig, master=self.grid_canvas_frame)
        self.grid_canvas.get_tk_widget().pack()
        
        # Create the grid plot (first configuration)
        grid = [
            self.histories[self.current_config_index][self.current_generation][i * grid_size:(i + 1) * grid_size]
            for i in range(grid_size)]
        self.grid_ax = self.fig.add_subplot(111)
        self.grid_ax.imshow(grid, cmap="binary")
        self.grid_ax.set_title(f"Configuration {self.current_config_index + 1}, Generation {self.current_generation}")
        
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # Create a separate canvas for the graph (statistics)
        self.graph_canvas = FigureCanvasTkAgg(self.fig, master=self.graph_canvas_frame)
        self.graph_canvas.get_tk_widget().pack()

        self.update_grid()  # Initially update the grid

    def update_grid(self):
        # Update grid with current generation from history
        grid = [
            self.histories[self.current_config_index][self.current_generation][i * self.grid_size:(i + 1) * self.grid_size]
            for i in range(self.grid_size)]
        self.grid_ax.clear()  # Clear the current axis to update the grid
        self.grid_ax.imshow(grid, cmap="binary")
        self.grid_ax.set_title(f"Configuration {self.current_config_index + 1}, Generation {self.current_generation}")

        # Redraw the grid canvas
        self.grid_canvas.draw_idle()

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
