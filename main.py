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
        self.grid = [
            0] * (grid_size * grid_size) if initial_state is None else list(initial_state)
        # Store initial state in history
        self.history = set([tuple(self.grid)])
        self.stable_count = 0  # Counter for stable generations
        # Set the number of generations before it's considered static
        self.max_stable_generations = 10
        self.dynamic_lifespan = 0  # Track dynamic lifespan of the grid
        self.lifespan = 0  # Total lifespan
        self.extra_lifespan = 0  # Lifespan for static or periodic grids
        # Tracks if the grid has become static (tied to the state)
        self.static_state = False
        # Tracks if the grid is repeating a cycle (tied to the state)
        self._is_periodic = False
        # Track the number of alive cells per generation
        self.alive_history = [self.get_alive_cells_count()]

    def step(self):
        """ Perform a single step in the Game of Life and update history. """
        if self._is_periodic or self.static_state:
            self.extra_lifespan += 1
            self.stable_count += 1
            return False

        cur_grid = self.grid[:]
        curState = tuple(cur_grid)  # Current state of the grid
        # Size of the grid (N x N)
        new_grid = [0] * (self.grid_size * self.grid_size)

        # logging.info(f"life span = {self.lifespan}")
        # logging.info(f"grid = {tuple(cur_grid)}")
        # logging.info(f"history = {list(self.history)}")

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                index = x * self.grid_size + y  # Calculate index for 2D to 1D conversion
                alive_neighbors = self.count_alive_neighbors(x, y)
                if cur_grid[index] == 1:
                    if alive_neighbors in [2, 3]:
                        new_grid[index] = 1
                else:
                    if alive_neighbors == 3:
                        new_grid[index] = 1

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

        self.grid = new_grid
        return True

    def run(self):
        """ Run the Game of Life until static or periodic state is reached, and calculate fitness. """
        while not self.static_state and not self._is_periodic:
            self.step()

    def count_alive_neighbors(self, x, y):
        alive = 0
        # Check neighbors for valid indices
        for i in range(-1, 2):  # Iterating over rows
            for j in range(-1, 2):  # Iterating over columns
                if i == 0 and j == 0:
                    continue  # Skip the cell itself
                nx, ny = x + i, y + j
                # Ensure neighbor is within grid bounds
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    index = nx * self.grid_size + ny  # Convert 2D to 1D index
                    # logging.info(f"index:{index} self.grid:{self.grid}")
                    alive += self.grid[index]  # Add 1 if neighbor is alive
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

    def check_periodicity(self):
        """ Return if the grid is periodic (repeating) """
        return self._is_periodic


class GeneticAlgorithm:
    def __init__(self, grid_size, population_size, generations, mutation_rate,
                 alive_cells_weight, lifespan_weight, alive_growth_weight,
                 cells_per_part, parts_with_cells, predefined_configurations=None, min_fitness_score=1):
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
        self.population = []

    def fitness(self, configuration):
        configuration_tuple = tuple(configuration)
        expected_size = self.grid_size * self.grid_size

        # ודא שהקונפיגורציה בגודל הנכון
        if len(configuration_tuple) != expected_size:
            raise ValueError(f"""Configuration size must be {
                             expected_size}, but got {len(configuration_tuple)}""")

        if configuration_tuple in self.fitness_cache:
            return self.fitness_cache[configuration_tuple]

        if configuration_tuple not in self.fitness_cache:
            # יצירת מופע של GameOfLife עם הקונפיגורציה הנוכחית
            game = GameOfLife(self.grid_size, configuration_tuple)
            game.run()  # ריצה של משחק החיים כדי לחשב את ההיסטוריה של התאים החיים

            alive_history = game.get_alive_history()  # היסטוריית התאים החיים

            # אם אין היסטוריה, הגדר את צמיחת התאים כ-0
            alive_growth = 0
            if len(alive_history) > 1:
                alive_growth = alive_history[-1] - alive_history[0]  # צמיחה של תאים בין הדורות


            total_alive_cells = sum(alive_history)  # סך התאים החיים

            # חישוב ה-Fitness score על בסיס נתוני המשחק
            lifespan = game.get_lifespan()
            fitness_score = (lifespan * self.lifespan_weight +
                             total_alive_cells / self.alive_cells_weight +
                             alive_growth * self.alive_growth_weight)

            self.fitness_cache[configuration_tuple] = fitness_score
            return fitness_score

    def mutate(self, configuration):
        N = self.grid_size
        expected_size = N * N

        # ודא שהקונפיגורציה בגודל הנכון
        if len(configuration) != expected_size:
            raise ValueError(f"""Configuration size must be {
                             expected_size}, but got {len(configuration)}""")

        # חילוק הוקטור למטריצה בגודל N x N (אך עדיין וקטור חד-ממדי)
        matrix = [configuration[i * N:(i + 1) * N]
                  for i in range(N)]  # חילוק הוקטור למטריצה

        # הפוך את המטריצה לבלוקים בעמודות
        blocks = []
        for i in range(N):
            block = [matrix[j][i]
                     for j in range(N)]  # בנה בלוק מ- N תאים בעמודה
            blocks.append(block)

        # ערבב את הבלוקים
        random.shuffle(blocks)

        # צור את הוקטור החדש מהבלוקים המעורבבים
        # המרת הבלוקים חזרה לוקטור
        new_configuration = [cell for block in blocks for cell in block]

        # הדפסת הקונפיגורציה החדשה
        logging.debug(f"new_configuration : {new_configuration}")

        return tuple(new_configuration)

    def select_parents(self):
        # חישוב ציוני ה-Fitness לכל חבר באוכלוסייה
        fitness_scores = []
        # logging.info(f"""{self.population}""")
        for config in self.population:
            score = self.fitness(config)
            if score is not None:
                fitness_scores.append(score)
            else:
                # אם לא הצלחנו לחשב את ה-Fitness, ניתן לו ציון מינימלי (למשל, אפס)
                fitness_scores.append(0)

        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            # במקרה בו כל הציונים הם אפס, בחר הורים אקראיים
            return random.choices(self.population, k=2)

        probabilities = [score / total_fitness for score in fitness_scores]
        parents = random.choices(self.population, weights=probabilities, k=2)
        return parents

    def crossover(self, parent1, parent2):
        N = self.grid_size
        mid = N // 2  # חצי מהגודל של המטריצה

        # ודא שההורים הם בגודל נכון
        if len(parent1) != N * N or len(parent2) != N * N:
            raise ValueError(f"""Parent configurations must be {
                             N * N}, but got sizes: {len(parent1)} and {len(parent2)}""")

        # חישוב החלקים המופרדים מהמטריצה הראשונה
        father_top_left = [parent1[i * N + j]
                           for i in range(mid) for j in range(mid)]
        father_bottom_right = [
            parent1[(i + mid) * N + (j + mid)] for i in range(mid) for j in range(mid)]

        # חישוב החלקים המופרדים מהמטריצה השנייה
        mother_top_right = [parent2[i * N + (j + mid)]
                            for i in range(mid) for j in range(mid)]
        mother_bottom_left = [parent2[(i + mid) * N + j]
                              for i in range(mid) for j in range(mid)]

        # יצירת הילד מהחלקים השונים של ההורים
        child_top_left = father_top_left
        child_top_right = mother_top_right
        child_bottom_left = mother_bottom_left
        child_bottom_right = father_bottom_right

        # חיבור כל החלקים יחד ליצירת הילד המלא
        child = []
        child.extend(child_top_left)
        child.extend(child_top_right)
        child.extend(child_bottom_left)
        child.extend(child_bottom_right)

        # ווידוא שה-child בגודל הנכון
        if len(child) != N * N:
            logging.debug(f"""child size mismatch, expected {
                          N * N}, got {len(child)}""")
            # רפד את ה-child במקרה של שגיאה בגודל
            child = child + [0] * (N * N - len(child))

        return tuple(child)

    def initialize_population(self):
        for _ in range(self.population_size):
            conf = self.random_configuration()
            if len(conf) < self.grid_size**2:
                logging.info("len(conf) < self.grid_size**2")
            else:
                self.population.append(conf)

    def expand_configuration(self, configuration):
        # התאמת הגודל לגודל מטריצה נכונה
        expanded = [0] * (self.grid_size * self.grid_size)
        for i in range(len(configuration)):
            expanded[i] = configuration[i]
        return tuple(expanded)

    def random_configuration(self):
        # הגודל של המטריצה (grid_size * grid_size)
        N = self.grid_size * self.grid_size
        configuration = [0] * N  # אתחול המטריצה עם אפסים

        # נחשב את num_parts מתוך grid_size, כך שיהיה תואם למימד המטריצה
        num_parts = self.grid_size  # מספר החלקים יהיה שווה למימד המטריצה

        # חלק את המטריצה לחלקים בגודל שווה
        part_size = self.grid_size  # גודל כל חלק

        # בחר את מספר הבלוקים שיקבלו תאים חיים
        parts_with_cells_indices = random.sample(
            range(self.grid_size), self.parts_with_cells)

        total_alive_cells = 0  # נתון זה יעקוב אחרי מספר התאים החיים שהוקצו

        # בחר תאים בכל בלוק שנבחר
        for part_index in parts_with_cells_indices:
            start_idx = part_index * part_size
            end_idx = start_idx + part_size

            # בחר את התאים המוקצים לבלוק הנבחר
            chosen_cells = random.sample(
                range(start_idx, end_idx), self.cells_per_part)

            # הדפס את המידע על הבלוק שנבחר
            logging.debug(f"Block {part_index} chosen cells: {chosen_cells}")

            # סמן את התאים שנבחרו כחיים
            for cell in chosen_cells:
                configuration[cell] = 1
                total_alive_cells += 1  # עדכון סך התאים החיים

        # הדפס את הקונפיגורציה הסופית
        logging.debug(f"""Generated configuration: {configuration} with {
                      total_alive_cells} alive cells""")

        return tuple(configuration)

    def run(self):
        all_fitness_scores = []
        for generation in range(self.generations):
            logging.info(f"Generation {generation + 1} started.")
            new_population = []
            for i in range(self.population_size):
                parent1, parent2 = self.select_parents()
                child = self.crossover(parent1, parent2)
                # logging.info(f"""child = {child}""")
                child = self.mutate(child)
                new_population.append(child)

            self.population = new_population
            self.fitness_cache.clear()

            logging.info(f"""self.population : {self.population}""")
            fitness_scores = [self.fitness(config) for config in self.population]
            all_fitness_scores.append(fitness_scores)

            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            std_fitness = (sum([(score - avg_fitness) ** 2 for score in fitness_scores]) / len(fitness_scores)) ** 0.5
            logging.info(f"""Generation {generation + 1}: Avg Fitness: {avg_fitness}, Std Dev: {std_fitness}""")

            best_fitness_score = max(fitness_scores)
            best_config_index = fitness_scores.index(best_fitness_score)
            best_config = self.population[best_config_index]
            logging.info(f""" self.population[best_config_index] : {self.population[best_config_index]}""")

            game = GameOfLife(self.grid_size, best_config)
            lifespan = game.get_lifespan()

            alive_growth = 0
            total_alive_cells = 0

            if lifespan > 0:
                alive_history = game.get_alive_history()
                if alive_history:
                    alive_growth = max(alive_history) - sum(alive_history)
                total_alive_cells = sum(alive_history)

            logging.info(f"""Best Configuration in Generation {generation + 1}:""")
            logging.info(f"""    Fitness Score: {best_fitness_score}""")
            logging.info(f"""    Lifespan: {lifespan}""")
            logging.info(f"""    Total Alive Cells: {total_alive_cells}""")
            logging.info(f"""    Alive Growth: {alive_growth}""")

        fitness_scores = [(config, self.fitness(config)) for config in self.population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        best_configs = [config for config, _ in fitness_scores[:5]]
        logging.info(f"""best configs : {best_configs}""")

        self.best_histories = []
        for config in best_configs:
            game = GameOfLife(self.grid_size, config)
            lifespan = game.get_lifespan()
            history = game.history
            alive_history = game.get_alive_history()
            self.best_histories.append(history)

            total_alive_cells = sum(alive_history)
            alive_growth = max(alive_history) - sum(alive_history)

            logging.info(f"""Top {config} Configuration:""")
            logging.info(f"""  Fitness Score: {self.fitness(config)}""")
            logging.info(f"""  Lifespan: {lifespan}""")
            logging.info(f"""  Total Alive Cells: {total_alive_cells}""")
            logging.info(f"""  Alive Growth: {alive_growth}""")

        return best_configs

def main(grid_size, population_size, generations, mutation_rate, alive_cells_weight,
         lifespan_weight, alive_growth_weight, cells_per_part, parts_with_cells, predefined_configurations=None):

    logging.info(f"""Starting run with parameters:
                    grid_size={grid_size}, population_size={population_size}, generations={generations},
                    mutation_rate={mutation_rate}, alive_cells_weight={alive_cells_weight},
                    lifespan_weight={lifespan_weight}, alive_growth_weight={alive_growth_weight},
                    cells_per_part={cells_per_part}, parts_with_cells={parts_with_cells}""")

    # Initialize the GeneticAlgorithm without max_lifespan
    algorithm = GeneticAlgorithm(
        grid_size, population_size, generations, mutation_rate,
        alive_cells_weight, lifespan_weight, alive_growth_weight,
        cells_per_part=cells_per_part, parts_with_cells=parts_with_cells,
        predefined_configurations=predefined_configurations
    )

    # Initialize the population for the algorithm
    algorithm.initialize_population()

    # Run the genetic algorithm to find the best configurations
    best_configs = algorithm.run()

    # Start the interactive simulation with the best configurations
    simulation = InteractiveSimulation(
        best_configs, algorithm.best_histories, grid_size)
    simulation.run()


# Example call to main function
main(grid_size=5, population_size=5, generations=20, mutation_rate=0.05,
     alive_cells_weight=50, lifespan_weight=10, alive_growth_weight=5,
     cells_per_part=5, parts_with_cells=2)
