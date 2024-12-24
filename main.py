import random
import logging
import matplotlib.pyplot as plt

# הגדרת לוגינג
logging.basicConfig(filename="simulation.log",
                    filemode='w',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class GameOfLife:
    def __init__(self, grid_size, initial_state=None):
        self.grid_size = grid_size
        grid = [0] * (grid_size * grid_size) if initial_state is None else list(initial_state)
        self.grid = grid
        self.history = [tuple(grid)]  # לשמור את כל המצבים

    def step(self):
        new_grid = [0] * (self.grid_size * self.grid_size)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                index = x * self.grid_size + y  # חישוב האינדקס הייחודי של התא
                alive_neighbors = self.count_alive_neighbors(x, y)
                if self.grid[index] == 1:
                    if alive_neighbors in [2, 3]:
                        new_grid[index] = 1
                else:
                    if alive_neighbors == 3:
                        new_grid[index] = 1
        self.grid = new_grid
        self.history.append(tuple(new_grid))  # שמירת המצב הנוכחי כ-tuple

    def count_alive_neighbors(self, x, y):
        alive = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                nx, ny = x + i, y + j
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    index = nx * self.grid_size + ny
                    alive += self.grid[index]
        return alive

    def get_alive_cells_count(self):
        alive_count = sum(self.grid)
        return alive_count

    def is_static_or_repeating(self):
        if self.get_alive_cells_count() == 0:
            return True
        current_state = tuple(self.grid)
        if current_state in self.history[:-1]:
            return True
        return False

    def get_lifespan(self, max_generations):
        for generation in range(max_generations):
            if self.is_static_or_repeating():
                return generation
            self.step()
        return max_generations


class GeneticAlgorithm:
    def __init__(self, grid_size, population_size, generations, mutation_rate, initial_alive_cells,
                 alive_cells_weight, max_lifespan, lifespan_weight, alive_growth_weight,
                 partition_ratio=4, predefined_configurations=None, min_fitness_score=1):
        self.grid_size = grid_size
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.initial_alive_cells = initial_alive_cells
        self.alive_cells_weight = alive_cells_weight
        self.max_lifespan = max_lifespan
        self.lifespan_weight = lifespan_weight
        self.alive_growth_weight = alive_growth_weight
        self.fitness_cache = {}
        self.predefined_configurations = predefined_configurations
        self.partition_ratio = partition_ratio  # הגדרת ה-partition_ratio
        self.population = self.initialize_population()
        self.min_fitness_score = min_fitness_score  # מינימום ציון כושר חיובי
        self.best_histories = []  # הוספת האטריבוט לשמירת ההיסטוריה של הקונפיגורציות הטובות ביותר

    def fitness(self, configuration):
        configuration_tuple = tuple(configuration)

        if configuration_tuple in self.fitness_cache:
            return self.fitness_cache[configuration_tuple]

        if configuration_tuple not in self.fitness_cache:
            alive_cells_count = sum(configuration_tuple)
            if alive_cells_count != self.initial_alive_cells:
                self.fitness_cache[configuration_tuple] = self.min_fitness_score - 1
                return self.min_fitness_score - 1

        game = GameOfLife(self.grid_size, configuration_tuple)

        lifespan = game.get_lifespan(self.max_lifespan)

        alive_counts = [game.get_alive_cells_count()]
        for _ in range(lifespan):
            game.step()
            alive_counts.append(game.get_alive_cells_count())

        total_alive_cells = sum(alive_counts)
        alive_growth = max(alive_counts) - self.initial_alive_cells

        fitness_score = (lifespan * self.lifespan_weight +
                         total_alive_cells / self.alive_cells_weight +
                         alive_growth * self.alive_growth_weight)

        fitness_score = max(fitness_score, self.min_fitness_score)
        self.fitness_cache[configuration_tuple] = fitness_score
        return fitness_score

    def initialize_population(self):
        if self.predefined_configurations is not None:
            population = [self.center_configuration(config) for config in self.predefined_configurations]
            while len(population) < self.population_size:
                population.append(self.random_configuration())
            return population
        return [self.random_configuration() for _ in range(self.population_size)]

    def center_configuration(self, configuration):
        rows, cols = len(configuration), len(configuration[0])
        top_offset = (self.grid_size - rows) // 2
        left_offset = (self.grid_size - cols) // 2

        centered = tuple([0] * self.grid_size for _ in range(self.grid_size))

        for i in range(rows):
            for j in range(cols):
                centered[top_offset + i][left_offset + j] = configuration[i][j]

        return tuple(cell for row in centered for cell in row)

    def random_configuration(self):
        while True:
            configuration = [0] * (self.grid_size * self.grid_size)
            all_cells = list(range(self.grid_size * self.grid_size))

            # חישוב מספר האזורים לפי הפרמטר partition_ratio
            num_subquarters = self.grid_size  // self.partition_ratio

            k = random.randint(1, num_subquarters)  # בחר את תת-הרבע מתוך האפשרויות

            quarter_size = len(all_cells) // num_subquarters
            start_index = (k - 1) * quarter_size
            end_index = start_index + quarter_size

            # לבדוק אם יש מספיק תאים ברבע שנבחר
            if self.initial_alive_cells <= quarter_size:
                chosen_cells = random.sample(all_cells[start_index:end_index], self.initial_alive_cells)
            else:
                # אם אין מספיק תאים, נבחר את כל התאים
                chosen_cells = all_cells[start_index:end_index]

            for cell in chosen_cells:
                configuration[cell] = 1

            if sum(configuration) == self.initial_alive_cells:
                break

        return tuple(configuration)

    def mutate(self, configuration):
        configuration = list(configuration)  # המרה מ-tuple ל-list
        N = self.grid_size * self.grid_size  # מספר התאים במערך

        alive_cells = [i for i, cell in enumerate(configuration) if cell == 1]

        for i in alive_cells:
            move = random.choice([1, -1, N, -N])  # הזזה אקראית
            new_index = (i + move) % N  # מציאת המיקום החדש

            while new_index != i and configuration[new_index] == 1:
                move = random.choice([1, -1, N, -N])  # בחר מחדש אם המיקום תפוס
                new_index = (i + move) % N  # עדכון המיקום

            configuration[i] = 0
            configuration[new_index] = 1

        return tuple(configuration)

    def select_parents(self):
        fitness_scores = [self.fitness(config) for config in self.population]
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            return random.choices(self.population, k=2)
        probabilities = [score / total_fitness for score in fitness_scores]
        parents = random.choices(self.population, weights=probabilities, k=2)
        return parents

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, self.grid_size - 1)
        child = list(parent1)
        child[crossover_point:] = parent2[crossover_point:]
        if sum(child) == 0:
            return parent1
        return tuple(child)

    def run(self):
        all_fitness_scores = []
        for generation in range(self.generations):
            new_population = []
            for i in range(self.population_size):
                parent1, parent2 = self.select_parents()
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)

            self.population = new_population
            self.fitness_cache.clear()

            fitness_scores = [self.fitness(config) for config in self.population]
            all_fitness_scores.append(fitness_scores)

            # לוגינג לדור הנוכחי
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            std_fitness = (sum([(score - avg_fitness) ** 2 for score in fitness_scores]) / len(fitness_scores)) ** 0.5
            logging.info(f"Generation {generation + 1}: Avg Fitness: {avg_fitness}, Std Dev: {std_fitness}")
            # לוגינג של פרטי הדור הנוכחי לכל קונפיגורציה
            for idx, config in enumerate(self.population):
                game = GameOfLife(self.grid_size, config)
                lifespan = game.get_lifespan(self.max_lifespan)
                total_alive_cells = sum([game.get_alive_cells_count() for _ in range(lifespan)])
                alive_growth = max([game.get_alive_cells_count() for _ in range(lifespan)]) - self.initial_alive_cells

                logging.info(f"  Configuration {idx + 1}:")
                logging.info(f"    Lifespan: {lifespan}")
                logging.info(f"    Total Alive Cells: {total_alive_cells}")
                logging.info(f"    Alive Growth: {alive_growth}")


        fitness_scores = [(config, self.fitness(config)) for config in self.population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        best_configs = [config for config, _ in fitness_scores[:5]]

        self.best_histories = []  # לאתחל את ההיסטוריות עבור הקונפיגורציות הטובות ביותר
        for config in best_configs:
            game = GameOfLife(self.grid_size, config)
            lifespan = game.get_lifespan(self.max_lifespan)
            history = game.history  # שמירה של ההיסטוריה של כל הדורות
            self.best_histories.append(history)

            # לוגינג של 5 הקונפיגורציות הטובות ביותר
            total_alive_cells = sum([game.get_alive_cells_count() for _ in range(lifespan)])
            alive_growth = max([game.get_alive_cells_count() for _ in range(lifespan)]) - self.initial_alive_cells
            logging.info(f"Top {config} Configuration:")
            logging.info(f"  Fitness Score: {self.fitness(config)}")
            logging.info(f"  Lifespan: {lifespan}")
            logging.info(f"  Total Alive Cells: {total_alive_cells}")
            logging.info(f"  Alive Growth: {alive_growth}")

        return best_configs


class InteractiveSimulation:
    def __init__(self, configurations, histories, grid_size):
        self.configurations = configurations
        self.histories = histories
        self.grid_size = grid_size
        self.current_config_index = 0
        self.current_generation = 0
        self.game = GameOfLife(grid_size, self.configurations[self.current_config_index])
        self.fig, self.ax = plt.subplots()

        grid = [self.game.grid[i * grid_size:(i + 1) * grid_size] for i in range(grid_size)]
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
        self.current_config_index = (self.current_config_index + 1) % len(self.configurations)
        self.current_generation = 0
        self.game = GameOfLife(self.grid_size, self.configurations[self.current_config_index])
        self.update_plot()

    def previous_configuration(self):
        self.current_config_index = (self.current_config_index - 1) % len(self.configurations)
        self.current_generation = 0
        self.game = GameOfLife(self.grid_size, self.configurations[self.current_config_index])
        self.update_plot()

    def next_generation(self):
        if self.current_generation + 1 < len(self.histories[self.current_config_index]):
            self.current_generation += 1
            self.game.grid = self.histories[self.current_config_index][self.current_generation]
            self.update_plot()

    def previous_generation(self):
        if self.current_generation > 0:
            self.current_generation -= 1
            self.game.grid = self.histories[self.current_config_index][self.current_generation]
            self.update_plot()

    def update_plot(self):
        grid = [self.game.grid[i * self.grid_size:(i + 1) * self.grid_size] for i in range(self.grid_size)]
        self.img.set_data(grid)
        self.ax.set_title(f"Configuration {self.current_config_index + 1}, Generation {self.current_generation}")
        self.fig.canvas.draw()

    def run(self):
        plt.show()


def main(grid_size, population_size, generations, mutation_rate, initial_alive_cells,
         alive_cells_weight, max_lifespan, lifespan_weight, alive_growth_weight, partition_ratio=4, predefined_configurations=None):
    # יצירת מופע של אלגוריתם גנטי עם כל הפרמטרים, כולל partition_ratio
    algorithm = GeneticAlgorithm(grid_size, population_size, generations, mutation_rate, initial_alive_cells,
                                 alive_cells_weight, max_lifespan, lifespan_weight, alive_growth_weight, partition_ratio=partition_ratio, predefined_configurations=predefined_configurations)
    
    best_configs = algorithm.run()

    simulation = InteractiveSimulation(
        best_configs, algorithm.best_histories, grid_size)
    simulation.run()

# קריאה לפונקציה הראשית עם הפרמטר החדש partition_ratio
main(grid_size=20, population_size=20, generations=100, mutation_rate=0.02,
     initial_alive_cells=5, alive_cells_weight=50, max_lifespan=5000,
     lifespan_weight=10, alive_growth_weight=5, partition_ratio=4)

# main(grid_size=30, population_size=20, generations=2, mutation_rate=0.02,
#      initial_alive_cells=5, alive_cells_weight=50, max_lifespan=5000,
#      lifespan_weight=10, alive_growth_weight=5,predefined_configurations=PREDEFINED_CONFIGS)
