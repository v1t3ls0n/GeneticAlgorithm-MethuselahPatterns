import random
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import logging

# הגדרת לוגינג
logging.basicConfig(filename="simulation.log",
                    filemode='w',  # קובץ הלוג יידרס בכל הפעלה
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class GameOfLife:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
        self.history = []  # לשמור את כל המצבים

    def set_initial_state(self, initial_state):
        self.grid = [row[:] for row in initial_state]
        self.history = [tuple(map(tuple, self.grid))]  # שמירת המצב ההתחלתי

    def step(self):
        new_grid = [[0 for _ in range(self.grid_size)]
                    for _ in range(self.grid_size)]
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                alive_neighbors = self.count_alive_neighbors(x, y)
                if self.grid[x][y] == 1:
                    if alive_neighbors in [2, 3]:
                        new_grid[x][y] = 1
                else:
                    if alive_neighbors == 3:
                        new_grid[x][y] = 1
        self.grid = new_grid
        self.history.append(tuple(map(tuple, self.grid)))  # שמירת המצב הנוכחי

    def count_alive_neighbors(self, x, y):
        alive = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                nx, ny = (x + i) % self.grid_size, (y + j) % self.grid_size
                alive += self.grid[nx][ny]
        return alive

    def get_alive_cells_count(self):
        return sum(sum(row) for row in self.grid)

    def is_static_or_repeating(self):
        """בדיקה אם המשחק סטטי או חזר על עצמו בעבר"""
        if self.get_alive_cells_count() == 0:
            return True
        # המרת המצב הנוכחי ל-tuple
        current_state = tuple(map(tuple, self.grid))
        # בדיקה אם המצב הנוכחי הופיע בעבר
        if current_state in self.history[:-1]:
            return True
        return False

    def get_lifespan(self, max_generations):
        """בדיקת אורך החיים של קונפיגורציה עד להתייצבות או למוות"""
        for generation in range(max_generations):
            if self.is_static_or_repeating():
                return generation
            self.step()
        return max_generations


class GeneticAlgorithm:
    def __init__(self, grid_size, population_size, generations, mutation_rate, initial_alive_cells, alive_cells_weight, max_lifespan):
        self.grid_size = grid_size
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.initial_alive_cells = initial_alive_cells
        self.alive_cells_weight = alive_cells_weight
        self.max_lifespan = max_lifespan
        self.population = [self.random_configuration()
                           for _ in range(population_size)]

    def random_configuration(self):
        configuration = [[0 for _ in range(self.grid_size)]
                         for _ in range(self.grid_size)]
        alive_cells = 0
        while alive_cells < self.initial_alive_cells:
            x, y = random.randint(0, self.grid_size -
                                  1), random.randint(0, self.grid_size - 1)
            if configuration[x][y] == 0:
                configuration[x][y] = 1
                alive_cells += 1
        return configuration

    def fitness(self, configuration):
        game = GameOfLife(self.grid_size)
        game.set_initial_state(configuration)
        lifespan = game.get_lifespan(self.max_lifespan)
        total_alive_cells = sum(sum(sum(row) for row in state)
                                for state in game.history)
        # שימוש בפרמטר מותאם אישית
        return lifespan * 2 + total_alive_cells / self.alive_cells_weight

    def select_parents(self):
        fitness_scores = [self.fitness(config) for config in self.population]
        total_fitness = sum(fitness_scores)
        probabilities = [score / total_fitness for score in fitness_scores]
        parents = random.choices(self.population, weights=probabilities, k=2)
        return parents

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, self.grid_size - 2)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        if sum(sum(row) for row in child) == 0:  # אם הילד ריק
            return parent1
        return child

    def mutate(self, configuration):
        num_mutations = max(
            1, int(self.mutation_rate * self.grid_size * self.grid_size))
        for _ in range(num_mutations):
            x, y = random.randint(0, self.grid_size -
                                  1), random.randint(0, self.grid_size - 1)
            configuration[x][y] = 1 - configuration[x][y]
        return configuration

    def run(self):
        for generation in range(self.generations):
            logging.info(f"Generation {generation + 1}/{self.generations}")
            new_population = []
            for i in range(self.population_size):
                parent1, parent2 = self.select_parents()
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                logging.info(
                    f"Child {i + 1}: Created via crossover and mutation")
                new_population.append(child)

            # עדכון האוכלוסייה
            self.population = new_population

            # לוג נוסף כדי להציג את המידע של כל פרמטר באוכלוסייה
            for idx, config in enumerate(self.population):
                game = GameOfLife(self.grid_size)
                game.set_initial_state(config)
                lifespan = game.get_lifespan(self.max_lifespan)
                total_alive_cells = sum(sum(row) for row in game.grid)
                logging.info(
                    f"Config {idx + 1}: Lifespan = {lifespan}, Alive Cells = {total_alive_cells}")

        fitness_scores = [(config, self.fitness(config))
                          for config in self.population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        best_configs = [config for config, _ in fitness_scores[:5]]

        # שמירת ההיסטוריות של התצורות הטובות ביותר
        self.best_histories = []
        for config in best_configs:
            game = GameOfLife(self.grid_size)
            game.set_initial_state(config)
            for _ in range(self.max_lifespan):
                if game.is_static_or_repeating():
                    break
                game.step()
            self.best_histories.append(game.history)

        # לוגים: כתיבת ההיסטוריות לקובץ
        for idx, history in enumerate(self.best_histories):
            logging.info(f"""History for best configuration {idx + 1}:
""")
            for gen, state in enumerate(history):
                logging.info(f"""Generation {gen}:
{state}
Alive cells: {sum(sum(row) for row in state)}""")

        return best_configs


class InteractiveSimulation:
    def __init__(self, configurations, histories, grid_size):
        self.configurations = configurations
        self.histories = histories
        self.grid_size = grid_size
        self.current_config_index = 0
        self.current_generation = 0
        self.game = GameOfLife(grid_size)
        self.game.set_initial_state(
            self.configurations[self.current_config_index])
        self.fig, self.ax = plt.subplots()
        self.img = self.ax.imshow(self.game.grid, cmap="binary")
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
        self.current_config_index = (
            self.current_config_index + 1) % len(self.configurations)
        self.current_generation = 0
        self.game.set_initial_state(
            self.configurations[self.current_config_index])
        self.update_plot()

    def previous_configuration(self):
        self.current_config_index = (
            self.current_config_index - 1) % len(self.configurations)
        self.current_generation = 0
        self.game.set_initial_state(
            self.configurations[self.current_config_index])
        self.update_plot()

    def next_generation(self):
        if self.current_generation + 1 < len(self.histories[self.current_config_index]):
            self.current_generation += 1
            self.game.grid = [
                row[:] for row in self.histories[self.current_config_index][self.current_generation]]
            self.update_plot()

    def previous_generation(self):
        if self.current_generation > 0:
            self.current_generation -= 1
            self.game.grid = [
                row[:] for row in self.histories[self.current_config_index][self.current_generation]]
            self.update_plot()

    def update_plot(self):
        self.img.set_data(self.game.grid)
        self.ax.set_title(f"Configuration {
                          self.current_config_index + 1}, Generation {self.current_generation}")
        self.fig.canvas.draw()

    def run(self):
        plt.show()


# דוגמה לשימוש
def main(grid_size, population_size, generations, mutation_rate, initial_alive_cells, alive_cells_weight, max_lifespan):
    algorithm = GeneticAlgorithm(grid_size, population_size, generations,
                                 mutation_rate, initial_alive_cells, alive_cells_weight, max_lifespan)
    best_configs = algorithm.run()

    logging.info("""חמש התצורות הטובות ביותר שנמצאו:""")
    for idx, config in enumerate(best_configs):
        logging.info(f"""Configuration {idx + 1}:
{config}""")

    simulation = InteractiveSimulation(
        best_configs, algorithm.best_histories, grid_size)
    simulation.run()


# קריאה לפונקציה הראשית עם פרמטרים מותאמים אישית
main(grid_size=30, population_size=20, generations=100, mutation_rate=0.02,
     initial_alive_cells=10, alive_cells_weight=50, max_lifespan=5000)
