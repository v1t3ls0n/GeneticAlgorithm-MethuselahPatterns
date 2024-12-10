import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import logging
from GameOfLife import GameOfLife
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'  # Replace 'Arial' with your desired font


class GeneticAlgorithmGUI:
    def __init__(self, genetic_algorithm):
        """
        Initialize the GUI for displaying genetic algorithm results.
        :param genetic_algorithm: An instance of GeneticAlgorithm.
        """
        if genetic_algorithm is None:
            logging.error("GeneticAlgorithm instance is required for GUI.")
            raise ValueError("GeneticAlgorithm instance is required.")

        self.ga = genetic_algorithm
        self.config = genetic_algorithm.config  # Shared Config instance

        if not self.config.top_5_configs:
            logging.error("No top configurations available to display in GUI.")
            raise ValueError("No top configurations available.")

        self.root = tk.Tk()
        self.root.title("Genetic Algorithm Results")

        # Current configuration and generation being displayed
        self.current_config_index = 0
        self.current_generation_index = 0
        self.game = None

        # Create frames for simulation and metrics
        self.simulation_frame = ttk.Frame(self.root, padding="10")
        self.simulation_frame.grid(row=0, column=0, sticky="nsew")

        self.metrics_frame = ttk.Frame(self.root, padding="10")
        self.metrics_frame.grid(row=0, column=1, sticky="nsew")

        # Populate the frames
        self.create_simulation_frame()
        self.create_metrics_frame()

        # Resize grid dynamically
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Bind keyboard events to the main window
        self.root.bind("<Right>", lambda event: self.show_next_generation())
        self.root.bind("<Left>", lambda event: self.show_previous_generation())
        self.root.bind("<Up>", lambda event: self.show_next_configuration())
        self.root.bind("<Down>", lambda event: self.show_previous_configuration())

        # Initialize the simulation for the first configuration
        self.update_simulation_canvas()
        self.update_metrics_plot()

    def create_simulation_frame(self):
        """Create widgets for the simulation frame."""
        self.simulation_label = ttk.Label(self.simulation_frame, text="Simulation", font=("Arial", 16))
        self.simulation_label.pack()

        # Dynamic label for the iteration number
        self.iteration_label = ttk.Label(self.simulation_frame, text="Generation: 0", font=("Arial", 12))
        self.iteration_label.pack()

        self.simulation_canvas = tk.Canvas(self.simulation_frame, width=400, height=400, bg="white")
        self.simulation_canvas.pack()

        self.simulation_controls = ttk.Frame(self.simulation_frame)
        self.simulation_controls.pack()

        self.previous_button = ttk.Button(self.simulation_controls, text="Previous Generation", command=self.show_previous_generation)
        self.previous_button.pack(side=tk.LEFT, padx=5)

        self.next_button = ttk.Button(self.simulation_controls, text="Next Generation", command=self.show_next_generation)
        self.next_button.pack(side=tk.LEFT, padx=5)

        self.previous_config_button = ttk.Button(self.simulation_controls, text="Previous Config", command=self.show_previous_configuration)
        self.previous_config_button.pack(side=tk.LEFT, padx=5)

        self.next_config_button = ttk.Button(self.simulation_controls, text="Next Config", command=self.show_next_configuration)
        self.next_config_button.pack(side=tk.LEFT, padx=5)

    def create_metrics_frame(self):
        """Create widgets for the metrics frame."""
        self.metrics_label = ttk.Label(self.metrics_frame, text="Metrics", font=("Arial", 16))
        self.metrics_label.pack()

        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.plot_ax = self.figure.add_subplot(111)
        self.plot_ax.set_title("Live Cells Over Time")
        self.plot_ax.set_xlabel("Generation")
        self.plot_ax.set_ylabel("Live Cells")

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.metrics_frame)
        self.canvas.get_tk_widget().pack()

    def update_metrics_plot(self):
        """Update the metrics plot for the current configuration."""
        if not self.config.top_5_configs:
            logging.debug("No top configurations to display in metrics plot.")
            return  # No configurations to display

        config = self.config.top_5_configs[self.current_config_index]
        live_cells_history = config.live_cells_history

        logging.debug(f"Updating metrics plot for {config.name} with live_cells_history: {live_cells_history}")

        self.plot_ax.clear()
        self.plot_ax.plot(range(1, len(live_cells_history) + 1), live_cells_history, label=config.name)
        self.plot_ax.set_title(f"Live Cells Over Time - {config.name}")
        self.plot_ax.set_xlabel("Generation")
        self.plot_ax.set_ylabel("Live Cells")
        self.plot_ax.legend()
        self.canvas.draw_idle()

    def update_simulation_canvas(self):
        """Update the simulation canvas with the current generation."""
        self.simulation_canvas.delete("all")
        if not self.config.top_5_configs:
            logging.debug("No top configurations to display in simulation canvas.")
            return  # No configurations to display

        config = self.config.top_5_configs[self.current_config_index]

        # Initialize GameOfLife instance if needed
        if self.game is None or not np.array_equal(self.game.initial_state, config.initial_state):
            self.game = GameOfLife(initial_state=config.initial_state, config=self.config)
            self.current_generation_index = 0
            logging.debug(f"Initialized GameOfLife for {config.name}")

        # Compute necessary generations
        while len(self.game.generations) <= self.current_generation_index:
            self.game.next_generation()
            logging.debug(f"Computed generation {len(self.game.generations)} for {config.name}")

        # Render current generation
        generation = self.game.generations[self.current_generation_index]
        generation_array = np.array(generation)
        cell_size = max(1, 400 // self.config.grid_size)

        live_cells = 0
        for x, row in enumerate(generation_array):
            for y, cell in enumerate(row):
                if cell:
                    live_cells += 1
                    self.simulation_canvas.create_rectangle(
                        y * cell_size, x * cell_size,
                        (y + 1) * cell_size, (x + 1) * cell_size,
                        fill="black", outline=""  # Improved performance
                    )

        logging.debug(f"Generation {self.current_generation_index + 1} for {config.name}: {live_cells} live cells")

        # Update iteration number
        self.iteration_label.config(text=f"Generation: {self.current_generation_index + 1}")

    def show_next_generation(self):
        """Show the next generation."""
        if not self.game:
            logging.warning("GameOfLife instance is not initialized.")
            return

        if self.current_generation_index < len(self.game.generations) - 1:
            self.current_generation_index += 1
        else:
            self.game.next_generation()
            self.current_generation_index += 1
        self.update_simulation_canvas()

    def show_previous_generation(self):
        """Show the previous generation."""
        if self.current_generation_index > 0:
            self.current_generation_index -= 1
            self.update_simulation_canvas()
        else:
            logging.info("Already at the first generation.")

    def show_next_configuration(self):
        """Show the next configuration."""
        if self.current_config_index < len(self.config.top_5_configs) - 1:
            self.current_config_index += 1
            self.current_generation_index = 0
            self.game = None  # Reset the game instance
            self.update_simulation_canvas()
            self.update_metrics_plot()
        else:
            logging.info("Already at the last configuration.")

    def show_previous_configuration(self):
        """Show the previous configuration."""
        if self.current_config_index > 0:
            self.current_config_index -= 1
            self.current_generation_index = 0
            self.game = None  # Reset the game instance
            self.update_simulation_canvas()
            self.update_metrics_plot()
        else:
            logging.info("Already at the first configuration.")

    def run(self):
        """Run the Tkinter main loop."""
        self.root.mainloop()
