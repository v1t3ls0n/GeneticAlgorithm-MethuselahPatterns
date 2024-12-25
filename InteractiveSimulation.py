import logging
import matplotlib.pyplot as plt
import numpy as np

class InteractiveSimulation:
    def __init__(self, configurations, histories, grid_size, generations_cache):
        logging.info("""Initializing InteractiveSimulation.""")
        self.configurations = configurations
        self.histories = histories
        self.grid_size = grid_size
        self.generations_cache = generations_cache  # Store generations cache for plotting
        self.current_config_index = 0
        self.current_generation = 0

        # No need to rerun the game now
        self.fig, self.ax = plt.subplots()

        # Display the first generation of the first configuration from history
        grid = [
            self.histories[self.current_config_index][self.current_generation][i * grid_size:(i + 1) * grid_size]
            for i in range(grid_size)]
        self.img = self.ax.imshow(grid, cmap="binary")
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.update_plot()

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
        self.update_plot()

    def previous_configuration(self):
        logging.info(f"""Switching to previous configuration. Current index: {self.current_config_index}""")
        self.current_config_index = (self.current_config_index - 1) % len(self.configurations)
        self.current_generation = 0
        self.update_plot()

    def next_generation(self):
        if self.current_generation + 1 < len(self.histories[self.current_config_index]):
            logging.info(f"""Switching to next generation. Current generation: {self.current_generation}""")
            self.current_generation += 1
            self.update_plot()

    def previous_generation(self):
        if self.current_generation > 0:
            logging.info(f"""Switching to previous generation. Current generation: {self.current_generation}""")
            self.current_generation -= 1
            self.update_plot()

    def update_plot(self):
        # Update the grid with the current generation from the history
        grid = [
            self.histories[self.current_config_index][self.current_generation][i * self.grid_size:(i + 1) * self.grid_size]
            for i in range(self.grid_size)]
        self.img.set_data(grid)
        self.ax.set_title(f"""Configuration {self.current_config_index + 1}, Generation {self.current_generation}""")
        self.fig.canvas.draw()

    def render_statistics(self):
        """
        Render relevant statistics as graphs from generations_cache with shaded areas for standard deviation.
        """
        generations = list(self.generations_cache.keys())

        # Create a single root plot for all the subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))  # 2x2 grid for the subplots

        # Average fitness over generations
        avg_fitness = [self.generations_cache[generation]['avg_fitness'] for generation in generations]
        std_fitness = [self.generations_cache[generation]['std_fitness'] for generation in generations]
        
        axes[0, 0].plot(generations, avg_fitness, label='Average Fitness', color='blue')
        axes[0, 0].fill_between(generations, np.subtract(avg_fitness, std_fitness), np.add(avg_fitness, std_fitness), color='blue', alpha=0.2, label='Standard Deviation')
        axes[0, 0].set_xlabel("Generation")
        axes[0, 0].set_ylabel("Average Fitness")
        axes[0, 0].set_title("Average Fitness over Generations")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Average lifespan over generations
        avg_lifespan = [self.generations_cache[generation]['avg_lifespan'] for generation in generations]
        std_lifespan = [self.generations_cache[generation]['std_lifespan'] for generation in generations]
        
        axes[0, 1].plot(generations, avg_lifespan, label='Average Lifespan', color='green')
        axes[0, 1].fill_between(generations, np.subtract(avg_lifespan, std_lifespan), np.add(avg_lifespan, std_lifespan), color='green', alpha=0.2, label='Standard Deviation')
        axes[0, 1].set_xlabel("Generation")
        axes[0, 1].set_ylabel("Average Lifespan")
        axes[0, 1].set_title("Average Lifespan over Generations")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Average alive growth rate over generations
        avg_alive_growth_rate = [self.generations_cache[generation]['avg_alive_growth_rate'] for generation in generations]
        std_alive_growth_rate = [self.generations_cache[generation]['std_alive_growth_rate'] for generation in generations]
        
        axes[1, 0].plot(generations, avg_alive_growth_rate, label='Average Alive Growth Rate', color='red')
        axes[1, 0].fill_between(generations, np.subtract(avg_alive_growth_rate, std_alive_growth_rate), np.add(avg_alive_growth_rate, std_alive_growth_rate), color='red', alpha=0.2, label='Standard Deviation')
        axes[1, 0].set_xlabel("Generation")
        axes[1, 0].set_ylabel("Average Alive Growth Rate")
        axes[1, 0].set_title("Average Alive Growth Rate over Generations")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Average total alive cells over generations
        avg_total_alive_cells = [self.generations_cache[generation]['avg_total_alive_cells'] for generation in generations]
        std_total_alive_cells = [self.generations_cache[generation]['std_total_alive_cells'] for generation in generations]
        
        axes[1, 1].plot(generations, avg_total_alive_cells, label='Average Total Alive Cells', color='purple')
        axes[1, 1].fill_between(generations, np.subtract(avg_total_alive_cells, std_total_alive_cells), np.add(avg_total_alive_cells, std_total_alive_cells), color='purple', alpha=0.2, label='Standard Deviation')
        axes[1, 1].set_xlabel("Generation")
        axes[1, 1].set_ylabel("Average Total Alive Cells")
        axes[1, 1].set_title("Average Total Alive Cells over Generations")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        # Adjust layout and display the graphs
        plt.tight_layout()
        plt.show()

    def run(self):
        logging.info("""Running interactive simulation.""")
        self.render_statistics()  # Render the statistics as graphs
        plt.show()
