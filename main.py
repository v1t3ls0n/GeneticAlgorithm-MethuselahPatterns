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
        grid = [
            0] * (grid_size * grid_size) if initial_state is None else list(initial_state)
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
        logging.info(f"Alive cells count: {alive_count}")
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
                 predefined_configurations=None, min_fitness_score=1):
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
        self.population = self.initialize_population()
        self.min_fitness_score = min_fitness_score  # מינימום ציון כושר חיובי


    def fitness(self, configuration):
        configuration_tuple = tuple(configuration)

        # לא בודקים שוב את מספר התאים החיים, אלא אם נדרש
        if configuration_tuple in self.fitness_cache:
            return self.fitness_cache[configuration_tuple]

        # כאן אנו בודקים ומגבילים את מספר התאים החיים רק בעת יצירת הקונפיגורציה הראשונה
        # כל מצב בהתחלה חייב להכיל בדיוק `self.initial_alive_cells` תאים חיים
        if configuration_tuple not in self.fitness_cache:  # אם זה המצב הראשון לקונפיגורציה הזו
            alive_cells_count = sum(configuration_tuple)
            if alive_cells_count != self.initial_alive_cells:
                logging.warning(f"Configuration skipped: Found {alive_cells_count} alive cells, expected {self.initial_alive_cells}.")
                self.fitness_cache[configuration_tuple] = self.min_fitness_score -1 # ציון כושר חיובי נמוך אם הקונפיגורציה לא עומדת בדרישות
                return self.min_fitness_score -1 # ציון כושר מינימלי חיובי אם מספר התאים החיים לא תואם למגבלה

        # יצירת המשחק עם הקונפיגורציה שהוגבלה מראש
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

        # מנגנון לוודא שהציון לא יהיה נמוך מהמינימום
        fitness_score = max(fitness_score, self.min_fitness_score)

        logging.info(f"Fitness score: {fitness_score}, Lifespan: {lifespan}, Total alive cells: {total_alive_cells}, Alive growth: {alive_growth}")
        self.fitness_cache[configuration_tuple] = fitness_score
        return fitness_score
    

    def initialize_population(self):
        if self.predefined_configurations is not None:
            population = [self.center_configuration(
                config) for config in self.predefined_configurations]
            while len(population) < self.population_size:
                population.append(self.random_configuration())
            return population
        return [self.random_configuration() for _ in range(self.population_size)]

    def center_configuration(self, configuration):
        rows, cols = len(configuration), len(configuration[0])
        top_offset = (self.grid_size - rows) // 2
        left_offset = (self.grid_size - cols) // 2

        # יצירת גריד ריק עם tuple (ולא list)
        centered = tuple([0] * self.grid_size for _ in range(self.grid_size))

        # העתקת הערכים מהקונפיגורציה המקורית למיקום החדש במרכז
        for i in range(rows):
            for j in range(cols):
                centered[top_offset + i][left_offset + j] = configuration[i][j]

        # החזרת הקונפיגורציה הממוקמת כ- tuple
        return tuple(cell for row in centered for cell in row)

    def random_configuration(self):
        while True:
            # יצירת גריד ריק בעזרת list (שניתן לשנות)
            configuration = [0] * (self.grid_size * self.grid_size)

            # יצירת רשימת כל התאים במטריצה
            all_cells = list(range(self.grid_size * self.grid_size))

            # בחירת תאים חיים אקראית בדיוק לפי מספר התאים שצריך
            chosen_cells = random.sample(all_cells, self.initial_alive_cells)
            logging.info(f"Chosen cells for initial configuration: {chosen_cells}")

            # הצבת התאים החיים במיקומים שנבחרו
            for cell in chosen_cells:
                configuration[cell] = 1

            # מוודאים שאין יותר תאים חיים מהדרישה
            if sum(configuration) == self.initial_alive_cells:
                logging.info(f"Generated valid configuration: {configuration}")
                break  # יצאנו מהלולאה כאשר יש בדיוק את מספר התאים החיים שנדרש
            else:
                logging.warning(f"Invalid configuration generated: {configuration}, retrying...")

        return tuple(configuration)

    def mutate(self, configuration):
        # חישוב המוטציה כ- shift של כל התאים החיים
        configuration = list(configuration)  # המרה מ-tuple ל-list
        N = self.grid_size * self.grid_size

        # בחרת כל התאים החיים
        alive_cells = [i for i, cell in enumerate(configuration) if cell == 1]
        logging.info(f"""alive cells : {alive_cells}""")

        # בחר כיוון (למשל שמאלה או ימינה או למעלה/למטה)

        for i in alive_cells:
            k = random.randint(1, N)
            new_index = (i + k) % N
            while new_index != i and configuration[new_index] == 1:
                k = random.randint(1, N)  # offset אקראי (הזזה)
                new_index = (i + k) % N

            # המרת התא המת שנמצא באינדקס החדש לתא חי
            configuration[i] = 0
            configuration[new_index] = 1

        # להחזיר ל-tuple אחרי ביצוע ההזזה
        return tuple(configuration)

    def select_parents(self):
        fitness_scores = [self.fitness(config) for config in self.population]
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            logging.warning(
                "Total fitness is zero, selecting parents randomly.")
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
            logging.info(f"Generation {generation + 1}/{self.generations}")
            new_population = []
            for i in range(self.population_size):
                parent1, parent2 = self.select_parents()
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                logging.info(
                    f"Child {i + 1}: Created via crossover and mutation")
                new_population.append(child)

            self.population = new_population
            self.fitness_cache.clear()

            fitness_scores = [self.fitness(config)
                              for config in self.population]
            all_fitness_scores.append(fitness_scores)
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            std_fitness = (sum(
                [(score - avg_fitness) ** 2 for score in fitness_scores]) / len(fitness_scores)) ** 0.5
            logging.info(f"Avg Fitness for Generation {generation + 1}: {avg_fitness}")
            logging.info(f"Std Dev of Fitness for Generation {generation + 1}: {std_fitness}")

        fitness_scores = [(config, self.fitness(config))
                          for config in self.population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        best_configs = [config for config, _ in fitness_scores[:5]]

        self.best_histories = []
        for config in best_configs:
            game = GameOfLife(self.grid_size, config)
            for _ in range(self.max_lifespan):
                if game.is_static_or_repeating():
                    break
                game.step()
            self.best_histories.append(game.history)

        return best_configs


class InteractiveSimulation:
    def __init__(self, configurations, histories, grid_size):
        self.configurations = configurations
        self.histories = histories
        self.grid_size = grid_size
        self.current_config_index = 0
        self.current_generation = 0
        self.game = GameOfLife(
            grid_size, self.configurations[self.current_config_index])
        self.fig, self.ax = plt.subplots()

        grid = [
            self.game.grid[i * grid_size:(i + 1) * grid_size] for i in range(grid_size)]
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
        self.current_config_index = (
            self.current_config_index + 1) % len(self.configurations)
        self.current_generation = 0
        self.game = GameOfLife(
            self.grid_size, self.configurations[self.current_config_index])
        self.update_plot()

    def previous_configuration(self):
        self.current_config_index = (
            self.current_config_index - 1) % len(self.configurations)
        self.current_generation = 0
        self.game = GameOfLife(
            self.grid_size, self.configurations[self.current_config_index])
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
        grid = [self.game.grid[i *
                               self.grid_size:(i + 1) * self.grid_size] for i in range(self.grid_size)]
        self.img.set_data(grid)
        self.ax.set_title(f"Configuration {self.current_config_index + 1}, Generation {self.current_generation}")
        self.fig.canvas.draw()

    def run(self):
        plt.show()


def main(grid_size, population_size, generations, mutation_rate, initial_alive_cells,
         alive_cells_weight, max_lifespan, lifespan_weight, alive_growth_weight, predefined_configurations=None):

    algorithm = GeneticAlgorithm(grid_size, population_size, generations, mutation_rate, initial_alive_cells,
                                 alive_cells_weight, max_lifespan, lifespan_weight, alive_growth_weight, predefined_configurations=predefined_configurations)
    best_configs = algorithm.run()

    logging.info("Top 5 configs")
    for idx, config in enumerate(best_configs):
        logging.info(f"Configuration {idx + 1}: {config}")

    simulation = InteractiveSimulation(
        best_configs, algorithm.best_histories, grid_size)
    simulation.run()


# קריאה לפונקציה הראשית עם פרמטרים מותאמים אישית
main(grid_size=5, population_size=20, generations=2, mutation_rate=0.02,
     initial_alive_cells=2, alive_cells_weight=50, max_lifespan=5000,
     lifespan_weight=10, alive_growth_weight=5)


# main(grid_size=30, population_size=20, generations=2, mutation_rate=0.02,
#      initial_alive_cells=5, alive_cells_weight=50, max_lifespan=5000,
#      lifespan_weight=10, alive_growth_weight=5,predefined_configurations=PREDEFINED_CONFIGS)
