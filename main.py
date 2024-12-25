from math import sqrt
import random
import logging
import matplotlib.pyplot as plt

# Set up logging to append to the same file for each run of the program
logging.basicConfig(filename="simulation.log",
                    filemode='a',  # Use 'a' to append to the file
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class InteractiveSimulation:
    def __init__(self, configurations, histories, grid_size):
        logging.info("Initializing InteractiveSimulation.")
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
        logging.debug(f"""Switching to next configuration. Current index: {
                      self.current_config_index}""")
        self.current_config_index = (
            self.current_config_index + 1) % len(self.configurations)
        self.current_generation = 0
        self.game = GameOfLife(
            self.grid_size, self.configurations[self.current_config_index])
        self.update_plot()

    def previous_configuration(self):
        logging.debug(f"""Switching to previous configuration. Current index: {
                      self.current_config_index}""")
        self.current_config_index = (
            self.current_config_index - 1) % len(self.configurations)
        self.current_generation = 0
        self.game = GameOfLife(
            self.grid_size, self.configurations[self.current_config_index])
        self.update_plot()

    def next_generation(self):
        if self.current_generation + 1 < len(self.histories[self.current_config_index]):
            logging.debug(f"""Switching to next generation. Current generation: {
                          self.current_generation}""")
            self.current_generation += 1
            self.game.grid = self.histories[self.current_config_index][self.current_generation]
            self.update_plot()

    def previous_generation(self):
        if self.current_generation > 0:
            logging.debug(f"""Switching to previous generation. Current generation: {
                          self.current_generation}""")
            self.current_generation -= 1
            self.game.grid = self.histories[self.current_config_index][self.current_generation]
            self.update_plot()

    def update_plot(self):
        grid = [self.game.grid[i *
                               self.grid_size:(i + 1) * self.grid_size] for i in range(self.grid_size)]
        self.img.set_data(grid)
        self.ax.set_title(f"""Configuration {
                          self.current_config_index + 1}, Generation {self.current_generation}""")
        self.fig.canvas.draw()

    def run(self):
        logging.info("Running interactive simulation.")
        plt.show()


class GameOfLife:
    def __init__(self, grid_size, initial_state=None):
        self.grid_size = grid_size
        self.grid = [
            0] * (grid_size * grid_size) if initial_state is None else list(initial_state)
        self.history = [tuple(self.grid)]  # לשמור את כל המצבים
        self.stable_count = 0  # מונה את מספר הדורות שבהם המצב נשאר יציב
        self.max_stable_generations = 10  # הגדרת גבול לדורות יציבים
        self.lifespan = 0  # ספירה של אורך החיים של הגריד
        self.extra_lifespan = 0  # תוספת לאורך החיים עבור גרידים יציבים או מחזוריים
        self.static_state = False  # מצב סטטי (לא משתנה)
        self._is_periodic = False  # מצב מחזורי (חוזר על עצמו)
        self.alive_history = [self.get_alive_cells_count()]

    def step(self):
        """ ביצוע שלב אחד במשחק החיים """
        new_grid = [0] * (self.grid_size * self.grid_size)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                index = x * self.grid_size + y
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

    def run(self):
        """ ריצה של משחק החיים עד למצב סטטי או מחזורי """
        while not self.static_state and not self._is_periodic:
            self.step()  # ריצה של שלב במשחק החיים

        # אם הגריד הגיע למצב יציב או מחזורי, נוסיף לו את הזמן הנוסף
        self.lifespan += self.extra_lifespan

        # הוספת לוגים עבור lifespan
        logging.debug(f"""Final Lifespan: {self.lifespan}, Extra Lifespan: {
                      self.extra_lifespan}""")

        # חישוב "צמיחה של תאים חיים"
        alive_history = self.alive_history
        alive_growth = 0
        if len(alive_history) > 1:
            # הצמיחה של התאים החיים מהתחלה ועד הסוף
            alive_growth = alive_history[-1] - alive_history[0]

        total_alive_cells = sum(alive_history)
        lifespan = self.lifespan  # אורך החיים הסופי לאחר כל השלבים

        # הוספת לוג של התוצאה הסופית
        logging.info(f"""Total Alive Cells: {total_alive_cells}, Lifespan: {
                     lifespan}, Alive Growth: {alive_growth}""")

        return total_alive_cells, alive_growth, alive_history, lifespan

    def reset(self):
        """ לאתחל את הגריד למצב ההתחלתי """
        logging.debug("Resetting the grid to initial state.")
        self.grid = list(self.history[0])
        self.history = [tuple(self.grid)]
        self.static_state = False
        self._is_periodic = False
        self.lifespan = 0
        self.extra_lifespan = 0
        self.stable_count = 0
        self.alive_history = [self.get_alive_cells_count()]


class GeneticAlgorithm:
    def __init__(self, grid_size, population_size, generations, mutation_rate,
                 alive_cells_weight, lifespan_weight, alive_growth_weight,
                 cells_per_part, parts_with_cells, predefined_configurations=None, min_fitness_score=1):
        logging.info("Initializing GeneticAlgorithm.")
        self.grid_size = grid_size
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.alive_cells_weight = alive_cells_weight
        self.lifespan_weight = lifespan_weight
        self.alive_growth_weight = alive_growth_weight
        self.fitness_cache = {}
        self.predefined_configurations = predefined_configurations
        self.cells_per_part = cells_per_part
        self.parts_with_cells = parts_with_cells
        self.min_fitness_score = min_fitness_score
        self.best_histories = []
        self.population = self.initialize_population()

    def fitness(self, configuration):
        configuration_tuple = tuple(configuration)
        expected_size = self.grid_size * self.grid_size

        # Ensure the configuration is of the correct size
        if len(configuration_tuple) != expected_size:
            raise ValueError(f"Configuration size must be {expected_size}, but got {len(configuration_tuple)}")

        if configuration_tuple in self.fitness_cache:
            return self.fitness_cache[configuration_tuple]

        # Create a GameOfLife instance with the current configuration
        game = GameOfLife(self.grid_size, configuration_tuple)
        total_alive_cells, alive_growth, alive_history, lifespan = game.run()  # Run the Game of Life to calculate fitness

        # Calculate the fitness score based on the game data
        fitness_score = (lifespan * self.lifespan_weight +
                         total_alive_cells / self.alive_cells_weight +
                         alive_growth * self.alive_growth_weight)

        # Cache the fitness score for the current configuration
        self.fitness_cache[configuration_tuple] = fitness_score
        return fitness_score

    def mutate(self, configuration):
        N = self.grid_size
        expected_size = N * N

        # Ensure the configuration is of the correct size
        if len(configuration) != expected_size:
            logging.error(f"Configuration size must be {expected_size}, but got {len(configuration)}")
            raise ValueError(f"Configuration size must be {expected_size}, but got {len(configuration)}")

        # Split the vector into an NxN matrix
        matrix = [configuration[i * N:(i + 1) * N] for i in range(N)]

        # Shuffle columns to create mutations
        blocks = []
        for i in range(N):
            block = [matrix[j][i] for j in range(N)]  # Create a block from N cells in a column
            blocks.append(block)

        # Shuffle the blocks
        random.shuffle(blocks)

        # Create the new configuration from the shuffled blocks
        new_configuration = [cell for block in blocks for cell in block]

        # Log the new configuration
        logging.debug(f"new_configuration : {new_configuration}")

        return tuple(new_configuration)

    def select_parents(self):
        # Calculate fitness scores for the entire population
        fitness_scores = []
        for config in self.population:
            score = self.fitness(config)
            if score is not None:
                fitness_scores.append(score)
            else:
                fitness_scores.append(0)  # Assign a minimum score if fitness could not be computed

        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            logging.warning("Total fitness is 0, selecting random parents.")
            return random.choices(self.population, k=2)

        probabilities = [score / total_fitness for score in fitness_scores]
        parents = random.choices(self.population, weights=probabilities, k=2)
        return parents

    def crossover(self, parent1, parent2):
        N = self.grid_size
        mid = N // 2  # Half the size of the matrix

        # Ensure parents are of the correct size
        if len(parent1) != N * N or len(parent2) != N * N:
            logging.error(f"Parent configurations must be {N * N}, but got sizes: {len(parent1)} and {len(parent2)}")
            raise ValueError(f"Parent configurations must be {N * N}, but got sizes: {len(parent1)} and {len(parent2)}")

        # Calculate the sections separated from the first matrix
        father_top_left = [parent1[i * N + j] for i in range(mid) for j in range(mid)]
        father_bottom_right = [parent1[(i + mid) * N + (j + mid)] for i in range(mid) for j in range(mid)]

        # Calculate the sections separated from the second matrix
        mother_top_right = [parent2[i * N + (j + mid)] for i in range(mid) for j in range(mid)]
        mother_bottom_left = [parent2[(i + mid) * N + j] for i in range(mid) for j in range(mid)]

        # Create the child from the different parts of the parents
        child_top_left = father_top_left
        child_top_right = mother_top_right
        child_bottom_left = mother_bottom_left
        child_bottom_right = father_bottom_right

        # Combine all parts to create the full child
        child = []
        child.extend(child_top_left)
        child.extend(child_top_right)
        child.extend(child_bottom_left)
        child.extend(child_bottom_right)

        # Ensure child is of the correct size
        if len(child) != N * N:
            logging.debug(f"Child size mismatch, expected {N * N}, got {len(child)}")
            child = child + [0] * (N * N - len(child))  # Pad the child if size mismatch occurs

        return tuple(child)

    def initialize_population(self):
        return [self.random_configuration() for _ in range(self.population_size)]

    def random_configuration(self):
        N = self.grid_size * self.grid_size
        configuration = [0] * N  # Initialize matrix with zeros

        num_parts = self.grid_size  # Number of parts is equal to the matrix dimension
        part_size = self.grid_size  # Size of each part

        # Choose parts to assign live cells
        parts_with_cells_indices = random.sample(range(self.grid_size), self.parts_with_cells)

        total_alive_cells = 0  # Track the number of alive cells assigned

        # Choose cells for each selected part
        for part_index in parts_with_cells_indices:
            start_idx = part_index * part_size
            end_idx = start_idx + part_size

            # Choose the cells allocated for the selected part
            chosen_cells = random.sample(range(start_idx, end_idx), self.cells_per_part)

            logging.debug(f"Block {part_index} chosen cells: {chosen_cells}")

            for cell in chosen_cells:
                configuration[cell] = 1
                total_alive_cells += 1  # Update total alive cells

        logging.debug(f"Generated configuration: {configuration} with {total_alive_cells} alive cells")

        return tuple(configuration)

    def run(self):
        all_fitness_scores = []
        for generation in range(self.generations):
            logging.info(f"Generation {generation + 1} started.")
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

            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            std_fitness = (sum([(score - avg_fitness) ** 2 for score in fitness_scores]) / len(fitness_scores)) ** 0.5
            logging.info(f"Generation {generation + 1}: Avg Fitness: {avg_fitness}, Std Dev: {std_fitness}")

            best_fitness_score = max(fitness_scores)
            best_config_index = fitness_scores.index(best_fitness_score)
            best_config = self.population[best_config_index]

            game = GameOfLife(self.grid_size, best_config)
            total_alive_cells, alive_growth, alive_history, lifespan = game.run()

            logging.info(f"Best Configuration in Generation {generation + 1}:")
            logging.info(f"    Fitness Score: {best_fitness_score}")
            logging.info(f"    Lifespan: {lifespan}")
            logging.info(f"    Total Alive Cells: {total_alive_cells}")
            logging.info(f"    Alive Growth: {alive_growth}")

        fitness_scores = [(config, self.fitness(config)) for config in self.population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        best_configs = [config for config, _ in fitness_scores[:5]]
        logging.info(f"Best configurations: {best_configs}")

        self.best_histories = []
        for config in best_configs:
            game = GameOfLife(self.grid_size, config)
            total_alive_cells, alive_growth, alive_history, lifespan = game.run()
            self.best_histories.append(game.history)

            logging.info(f"Top {config} Configuration:")
            logging.info(f"  Fitness Score: {self.fitness(config)}")
            logging.info(f"  Lifespan: {lifespan}")
            logging.info(f"  Total Alive Cells: {total_alive_cells}")
            logging.info(f"  Alive Growth: {alive_growth}")

        return best_configs


def main(grid_size, population_size, generations, mutation_rate, alive_cells_weight,
         lifespan_weight, alive_growth_weight, cells_per_part, parts_with_cells, predefined_configurations=None):
    logging.info(f"""Starting run with parameters: grid_size={grid_size}, population_size={population_size}, generations={generations}, mutation_rate={mutation_rate}, alive_cells_weight={
                 alive_cells_weight}, lifespan_weight={lifespan_weight}, alive_growth_weight={alive_growth_weight}, cells_per_part={cells_per_part}, parts_with_cells={parts_with_cells}""")

    algorithm = GeneticAlgorithm(
        grid_size, population_size, generations, mutation_rate,
        alive_cells_weight, lifespan_weight, alive_growth_weight,
        cells_per_part=cells_per_part, parts_with_cells=parts_with_cells,
        predefined_configurations=predefined_configurations
    )

    best_configs = algorithm.run()

    simulation = InteractiveSimulation(
        best_configs, algorithm.best_histories, grid_size)
    simulation.run()


# Example call to main function
main(grid_size=20, population_size=20, generations=20, mutation_rate=0.02,
     alive_cells_weight=50, lifespan_weight=10, alive_growth_weight=5,
     cells_per_part=5, parts_with_cells=2)
