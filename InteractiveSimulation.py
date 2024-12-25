import logging
import matplotlib.pyplot as plt

class InteractiveSimulation:
    def __init__(self, configurations, histories, grid_size):
        logging.info("""Initializing InteractiveSimulation.""")
        self.configurations = configurations
        self.histories = histories
        self.grid_size = grid_size
        self.current_config_index = 0
        self.current_generation = 0

        # לא נדרש כעת להפעיל את המשחק מחדש
        self.fig, self.ax = plt.subplots()

        # הצגת הדור הראשון של הקונפיגורציה הראשונה מתוך ההיסטוריה
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
        # עדכון הגריד עם הדור הנוכחי מתוך ההיסטוריה
        grid = [
            self.histories[self.current_config_index][self.current_generation][i * self.grid_size:(i + 1) * self.grid_size]
            for i in range(self.grid_size)]
        self.img.set_data(grid)
        self.ax.set_title(f"""Configuration {self.current_config_index + 1}, Generation {self.current_generation}""")
        self.fig.canvas.draw()

    def run(self):
        logging.info("""Running interactive simulation.""")
        plt.show()
