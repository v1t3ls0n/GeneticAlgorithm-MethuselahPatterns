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
        self.ax.set_title(f"""Configuration {
                          self.current_config_index + 1}, Generation {self.current_generation}""")
        self.fig.canvas.draw()

    def run(self):
        plt.show()


class GameOfLife:
    def __init__(self, grid_size, initial_state=None):
        self.grid_size = grid_size
        self.grid = [0] * (grid_size * grid_size) if initial_state is None else list(initial_state)
        self.history = set([tuple(self.grid)])  # Store initial state in history
        self.stable_count = 0  # Counter for stable generations
        self.max_stable_generations = 10  # Set the number of generations before it's considered static
        self.dynamic_lifespan = 0  # Track dynamic lifespan of the grid
        self.lifespan = 0  # Total lifespan
        self.extra_lifespan = 0  # Lifespan for static or periodic grids
        self.static_state = False  # Tracks if the grid has become static (tied to the state)
        self._is_periodic = False  # Tracks if the grid is repeating a cycle (tied to the state)
        self.alive_history = []  # Track the number of alive cells per generation

    def step(self):
        """ Perform a single step in the Game of Life and update history. """
        logging.info(f"life span = {self.lifespan}")
        if self.is_periodic or self.static_state:
            # If static or periodic, do not update history, but track extra lifespan
            self.extra_lifespan += 1
            return
        curState = tuple(self.grid)  # Current state of the grid
        new_grid = [0] * (self.grid_size ** 2)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                index = x * self.grid_size + y
                alive_neighbors = self.count_alive_neighbors(x, y)
                if curState[index] == 1:
                    if alive_neighbors in [2, 3]:
                        new_grid[index] = 1
                else:
                    if alive_neighbors == 3:
                        new_grid[index] = 1

        self.grid = new_grid
        newState = tuple(new_grid)  # New state after the step

        # Check for static state (no change between current and previous grid)
        if newState == curState:
            self.static_state = True
        # Check for periodicity (if the new state has appeared before)
        elif newState in self.history:
            self._is_periodic = True
        else:
            self.lifespan += 1  # Increment lifespan for non-static, non-periodic grids
            self.history.add(newState)  # Add the new state to history
            self.alive_history.append(self.get_alive_cells_count())

    def check_periodicity(self):
        """ Check for periodicity and return if the grid is periodic. """
        return self._is_periodic

    def run(self):
        """ Run the Game of Life until static or periodic state is reached, and calculate fitness. """
        while not self.static_state and not self.check_periodicity():
            self.step()

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
        return sum(self.grid)

    def get_lifespan(self):
        """ Return the lifespan of the current grid (including extra lifespan for static/periodic) """
        return self.lifespan + self.extra_lifespan

    def get_alive_history(self):
        """ Return the history of the number of alive cells for each generation """
        return self.alive_history

    def is_static(self):
        """ Return if the grid is static """
        return self.static_state

    def is_periodic(self):
        """ Return if the grid is periodic (repeating) """
        return self._is_periodic


class GeneticAlgorithm:
    def __init__(self, grid_size, population_size, generations, mutation_rate, initial_alive_cells,
                 alive_cells_weight, lifespan_weight, alive_growth_weight,
                 num_parts=4, cells_per_part=1, parts_with_cells=2, predefined_configurations=None, min_fitness_score=1):
        self.grid_size = grid_size
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.initial_alive_cells = initial_alive_cells
        self.alive_cells_weight = alive_cells_weight
        self.lifespan_weight = lifespan_weight
        self.alive_growth_weight = alive_growth_weight
        self.fitness_cache = {}
        self.predefined_configurations = predefined_configurations
        self.num_parts = num_parts
        self.cells_per_part = cells_per_part
        self.parts_with_cells = parts_with_cells
        self.min_fitness_score = min_fitness_score
        self.best_histories = []
        self.population = self.initialize_population()

    def fitness(self, configuration):
        configuration_tuple = tuple(configuration)

        if configuration_tuple not in self.fitness_cache:
            alive_cells_count = sum(configuration_tuple)
            if alive_cells_count != self.initial_alive_cells:
                self.fitness_cache[configuration_tuple] = self.min_fitness_score - 1
                return self.min_fitness_score - 1

        game = GameOfLife(self.grid_size, configuration_tuple)

        # הרצה של המשחק עד שמגיעים למצב סטטי או מחזורי
        game.run()  # הפעלת המתודה run של GameOfLife שתחשב את ציון ה- fitness

        # חישוב ציון ההתאמה לאחר שהמשחק הגיע למצב סטטי או מחזורי
        if game.is_static():
            alive_history = game.get_alive_history()
            total_alive_cells = sum(alive_history)
            alive_growth = max(alive_history) - self.initial_alive_cells
            lifespan = game.get_lifespan()
            fitness_score = (lifespan * self.lifespan_weight +
                             total_alive_cells / self.alive_cells_weight +
                             alive_growth * self.alive_growth_weight)

        elif game.is_periodic():
            fitness_score = -self.lifespan_weight  # ציון שלילי עבור קונפיגורציות מחזוריות

        else:
            # חישוב ציון ההתאמה עבור קונפיגורציות שלא הגיעו למצב סטטי או מחזורי
            alive_history = game.get_alive_history()
            total_alive_cells = sum(alive_history)
            alive_growth = max(alive_history) - self.initial_alive_cells
            lifespan = game.get_lifespan()
            fitness_score = (lifespan * self.lifespan_weight +
                             total_alive_cells / self.alive_cells_weight +
                             alive_growth * self.alive_growth_weight)

        self.fitness_cache[configuration_tuple] = fitness_score
        return fitness_score

    def mutate(self, configuration):
        N = self.grid_size
        mid = N // 2

        matrix = [configuration[i * N:(i + 1) * N] for i in range(N)]

        top_left = [row[:mid] for row in matrix[:mid]]
        top_right = [row[mid:] for row in matrix[:mid]]
        bottom_left = [row[:mid] for row in matrix[mid:]]
        bottom_right = [row[mid:] for row in matrix[mid:]]

        if N % 2 != 0:
            extra_row = matrix[N-1]
            top_left.append(extra_row[:mid])
            top_right.append(extra_row[mid:])
            bottom_left.append(extra_row[:mid])
            bottom_right.append(extra_row[mid:])

        quarters = [top_left, top_right, bottom_left, bottom_right]

        random.shuffle(quarters)

        new_matrix = []
        for i in range(mid):
            new_matrix.append(quarters[0][i] + quarters[1][i])
        for i in range(mid):
            new_matrix.append(quarters[2][i] + quarters[3][i])

        if N % 2 != 0:
            new_matrix[-1] += extra_row

        new_configuration = [cell for row in new_matrix for cell in row]

        return tuple(new_configuration)

    def select_parents(self):
        fitness_scores = [self.fitness(config) for config in self.population]
        total_fitness = sum(fitness_scores)
        if total_fitness == self.min_fitness_score:
            return random.choices(self.population, k=2)
        probabilities = [score / total_fitness for score in fitness_scores]
        parents = random.choices(self.population, weights=probabilities, k=2)
        return parents

    def crossover(self, parent1, parent2):
        N = self.grid_size
        mid = N // 2

        father_top_left = [parent1[i * N + j]
                           for i in range(mid) for j in range(mid)]
        father_bottom_right = [parent1[(i + mid) * N + (j + mid)]
                               for i in range(mid) for j in range(mid)]

        mother_top_right = [parent2[i * N + (j + mid)]
                            for i in range(mid) for j in range(mid)]
        mother_bottom_left = [parent2[(i + mid) * N + j] for i in range(mid)
                              for j in range(mid)]

        child_top_left = father_top_left
        child_top_right = mother_top_right
        child_bottom_left = mother_bottom_left
        child_bottom_right = father_bottom_right

        child = []
        child.extend(child_top_left)
        child.extend(child_top_right)
        child.extend(child_bottom_left)
        child.extend(child_bottom_right)

        return tuple(child)

    def initialize_population(self):
        if self.predefined_configurations is not None:
            population = [self.expand_configuration(
                config) for config in self.predefined_configurations]
            while len(population) < self.population_size:
                population.append(self.random_configuration())
            return population
        return [self.random_configuration() for _ in range(self.population_size)]

    def expand_configuration(self, configuration):
        rows, cols = len(configuration), len(configuration[0])
        expanded = [0] * (self.grid_size * self.grid_size)
        for i in range(rows):
            for j in range(cols):
                expanded[i * self.grid_size + j] = configuration[i][j]
        return tuple(expanded)

    def random_configuration(self):
        while True:
            configuration = [0] * (self.grid_size * self.grid_size)
            all_cells = list(range(self.grid_size * self.grid_size))

            part_size = self.grid_size * self.grid_size // self.num_parts
            parts_with_cells_indices = random.sample(
                range(self.num_parts), self.parts_with_cells)

            for part_index in parts_with_cells_indices:
                start_idx = part_index * part_size
                end_idx = start_idx + part_size
                chosen_cells = random.sample(
                    all_cells[start_idx:end_idx], self.cells_per_part)
                for cell in chosen_cells:
                    configuration[cell] = 1

            if sum(configuration) == self.initial_alive_cells:
                break

        return tuple(configuration)

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

            fitness_scores = [self.fitness(config)
                            for config in self.population]
            all_fitness_scores.append(fitness_scores)

            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            std_fitness = (sum(
                [(score - avg_fitness) ** 2 for score in fitness_scores]) / len(fitness_scores)) ** 0.5
            logging.info(f"""Generation {generation + 1}: Avg Fitness: {avg_fitness}, Std Dev: {std_fitness}""")

            best_fitness_score = max(fitness_scores)
            best_config_index = fitness_scores.index(best_fitness_score)
            best_config = self.population[best_config_index]

            game = GameOfLife(self.grid_size, best_config)
            lifespan = game.get_lifespan()

            alive_growth = 0
            total_alive_cells = 0

            if lifespan > 0:
                alive_history = game.get_alive_history()

                # אם יש היסטוריה, נחשב את הגידול
                if alive_history:
                    alive_growth = max(alive_history) - self.initial_alive_cells
                total_alive_cells = sum(alive_history)

            logging.info(f"""  Best Configuration in Generation {generation + 1}:""")
            logging.info(f"""    Fitness Score: {best_fitness_score}""")
            logging.info(f"""    Lifespan: {lifespan}""")
            logging.info(f"""    Total Alive Cells: {total_alive_cells}""")
            logging.info(f"""    Alive Growth: {alive_growth}""")

        fitness_scores = [(config, self.fitness(config))
                        for config in self.population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        best_configs = [config for config, _ in fitness_scores[:5]]

        self.best_histories = []
        for config in best_configs:
            game = GameOfLife(self.grid_size, config)
            lifespan = game.get_lifespan()
            history = game.history
            alive_history = game.get_alive_history()
            self.best_histories.append(history)

            total_alive_cells = sum(alive_history)
            alive_growth = max(alive_history) - self.initial_alive_cells

            logging.info(f"""Top {config} Configuration:""")
            logging.info(f"""  Fitness Score: {self.fitness(config)}""")
            logging.info(f"""  Lifespan: {lifespan}""")
            logging.info(f"""  Total Alive Cells: {total_alive_cells}""")
            logging.info(f"""  Alive Growth: {alive_growth}""")

        return best_configs


def main(grid_size, population_size, generations, mutation_rate, initial_alive_cells,
         alive_cells_weight, lifespan_weight, alive_growth_weight, predefined_configurations=None):
    logging.info(f"""Starting run with parameters:
                    grid_size={grid_size}, population_size={population_size}, generations={generations},
                    mutation_rate={mutation_rate}, initial_alive_cells={initial_alive_cells},
                    alive_cells_weight={alive_cells_weight}, lifespan_weight={lifespan_weight},
                    alive_growth_weight={alive_growth_weight}""")

    num_parts = (grid_size ** 2) // initial_alive_cells
    cells_per_part = initial_alive_cells
    parts_with_cells = min(num_parts, (initial_alive_cells // cells_per_part))

    # Initialize the GeneticAlgorithm without max_lifespan
    algorithm = GeneticAlgorithm(
        grid_size, population_size, generations, mutation_rate, initial_alive_cells,
        alive_cells_weight, lifespan_weight, alive_growth_weight,
        num_parts=num_parts, cells_per_part=cells_per_part, parts_with_cells=parts_with_cells,
        predefined_configurations=predefined_configurations
    )

    best_configs = algorithm.run()

    simulation = InteractiveSimulation(
        best_configs, algorithm.best_histories, grid_size)
    simulation.run()


# Example call to main function
main(grid_size=20, population_size=5, generations=20, mutation_rate=0.05,
     initial_alive_cells=5, alive_cells_weight=50, lifespan_weight=10,
     alive_growth_weight=5)
