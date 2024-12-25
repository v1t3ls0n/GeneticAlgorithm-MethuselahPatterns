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
        logging.debug(f"Switching to next configuration. Current index: {self.current_config_index}")
        self.current_config_index = (
            self.current_config_index + 1) % len(self.configurations)
        self.current_generation = 0
        self.game = GameOfLife(
            self.grid_size, self.configurations[self.current_config_index])
        self.update_plot()

    def previous_configuration(self):
        logging.debug(f"Switching to previous configuration. Current index: {self.current_config_index}")
        self.current_config_index = (
            self.current_config_index - 1) % len(self.configurations)
        self.current_generation = 0
        self.game = GameOfLife(
            self.grid_size, self.configurations[self.current_config_index])
        self.update_plot()

    def next_generation(self):
        if self.current_generation + 1 < len(self.histories[self.current_config_index]):
            logging.debug(f"Switching to next generation. Current generation: {self.current_generation}")
            self.current_generation += 1
            self.game.grid = self.histories[self.current_config_index][self.current_generation]
            self.update_plot()

    def previous_generation(self):
        if self.current_generation > 0:
            logging.debug(f"Switching to previous generation. Current generation: {self.current_generation}")
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
        self.grid = [0] * (grid_size * grid_size) if initial_state is None else list(initial_state)
        
        # Store the initial state of the grid
        self.initial_state = tuple(self.grid)
        
        # Store initial state in history
        self.history = [self.initial_state]
        self.stable_count = 0  # Counter for stable generations
        self.max_stable_generations = 10  # Set the number of generations before it's considered static
        self.dynamic_lifespan = 0  # Track dynamic lifespan of the grid
        self.lifespan = 0  # Total lifespan (should start at 0)
        self.extra_lifespan = 0  # Lifespan for static or periodic grids
        self.static_state = False  # Tracks if the grid has become static (tied to the state)
        self._is_periodic = False  # Tracks if the grid is repeating a cycle (tied to the state)
        self.total_alive_cells = 0
        self.alive_growth = 0
        self.alive_history = [sum(self.grid)]

    def step(self):
        """ Perform a single step in the Game of Life and update history. """
        cur_grid = self.grid[:]
        new_grid = [0] * (self.grid_size * self.grid_size)
        # Iterate over the grid to apply the Game of Life rules
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
        curState = tuple(cur_grid)  # Current state of the grid

        # Check for static state (no change between current and previous grid)
        if newState == curState:
            self.static_state = True
        # Check for periodicity (if the new state has appeared before)
        elif newState in self.history[:-1]:
            self._is_periodic = True
        else:
            self.grid = new_grid







    def run(self):
        """ Run the Game of Life until static or periodic state is reached, and calculate fitness. """

        while (not self.static_state and not self._is_periodic) or self.stable_count < self.max_stable_generations:
            self.alive_history.append(self.get_alive_cells_count())
            self.history.append(tuple(self.grid[:]))
            if self._is_periodic or self.static_state:
                self.stable_count += 1
            self.lifespan += 1  # Increment lifespan on each step
            self.step()  # Run one step of the game

        self.total_alive_cells = sum(self.alive_history)

        # הוספת לוג של התוצאה הסופית
        logging.info(f"Total Alive Cells: {self.total_alive_cells}, Lifespan: {self.lifespan}, Alive Growth: {self.alive_growth}")
        

        
        
    
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

    def reset(self):
        """ Reset the grid to its initial state """
        logging.debug("Resetting the grid to initial state.")
        self.grid = list(self.initial_state)
        self.history = [self.initial_state]
        self.static_state = False
        self._is_periodic = False
        self.lifespan = 0
        self.extra_lifespan = 0
        self.stable_count = 0
        self.alive_history = [sum(self.grid)]


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

        # ודא שהקונפיגורציה בגודל הנכון
        if len(configuration_tuple) != expected_size:
            raise ValueError(f"Configuration size must be {expected_size}, but got {len(configuration_tuple)}")

        if configuration_tuple in self.fitness_cache:
            return self.fitness_cache[configuration_tuple]

        # יצירת מופע של GameOfLife עם הקונפיגורציה הנוכחית
        game = GameOfLife(self.grid_size, configuration_tuple)
        game.run()
         # ריצה של משחק החיים כדי לחשב את ההיסטוריה של התאים החיים
        if not game.alive_history: 
            game.alive_growth = 0
        else:
            game.alive_growth = max(game.alive_history) - game.alive_history[0]

        logging.info(f"Configuration: {configuration_tuple}, Lifespan: {game.lifespan}, Alive Growth: {game.alive_growth}, Alive Cells: {game.total_alive_cells}")

        # הוספת חישוב ה-Fitness score כך שה-lifespan לא יתאפס
        fitness_score = (game.lifespan * self.lifespan_weight +
                        game.total_alive_cells / self.alive_cells_weight +
                        game.alive_growth * self.alive_growth_weight)

        if game.lifespan == 1:  # אם ה-lifespan לא מתעדכן כראוי, נוודא שהחישוב לא יתבצע
            logging.warning(f"Warning: Lifespan is 1, which may indicate an issue with the simulation!")

        self.fitness_cache[configuration_tuple] = fitness_score
        return fitness_score


    def mutate(self, configuration):
        N = self.grid_size
        expected_size = N * N

        # Ensure configuration is of the correct size
        if len(configuration) != expected_size:
            logging.error(f"Configuration size must be {expected_size}, but got {len(configuration)}")
            raise ValueError(f"""Configuration size must be {expected_size}, but got {len(configuration)}""")

        # Split the vector into an NxN matrix (still one-dimensional)
        matrix = [configuration[i * N:(i + 1) * N] for i in range(N)]  # Split the vector into a matrix

        # Shuffle columns to create mutations
        blocks = []
        for i in range(N):
            block = [matrix[j][i] for j in range(N)]  # Create block from N cells in column
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
                # If we failed to compute fitness, assign it a minimum score (e.g., 0)
                fitness_scores.append(0)

        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            # If all scores are 0, choose random parents
            logging.warning("Total fitness is 0, selecting random parents.")
            return random.choices(self.population, k=2)

        probabilities = [score / total_fitness for score in fitness_scores]
        parents = random.choices(self.population, weights=probabilities, k=2)
        return parents

    def crossover(self, parent1, parent2):
        N = self.grid_size
        mid = N // 2  # Half the size of the matrix

        # Ensure parents are the correct size
        if len(parent1) != N * N or len(parent2) != N * N:
            logging.error(f"Parent configurations must be {N * N}, but got sizes: {len(parent1)} and {len(parent2)}")
            raise ValueError(f"""Parent configurations must be {N * N}, but got sizes: {len(parent1)} and {len(parent2)}""")

        # Calculate the sections separated from the first matrix
        father_top_left = [parent1[i * N + j] for i in range(mid) for j in range(mid)]
        father_bottom_right = [
            parent1[(i + mid) * N + (j + mid)] for i in range(mid) for j in range(mid)]

        # Calculate the sections separated from the second matrix
        mother_top_right = [parent2[i * N + (j + mid)] for i in range(mid) for j in range(mid)]
        mother_bottom_left = [parent2[(i + mid) * N + j]
                              for i in range(mid) for j in range(mid)]

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
            logging.debug(f"""child size mismatch, expected {N * N}, got {len(child)}""")
            # Pad the child if size mismatch occurs
            child = child + [0] * (N * N - len(child))

        return tuple(child)

    def initialize_population(self):
        return [self.random_configuration() for _ in range(self.population_size)]

    def random_configuration(self):
        # Size of the matrix (grid_size * grid_size)
        N = self.grid_size * self.grid_size
        configuration = [0] * N  # Initialize matrix with zeros

        # Calculate num_parts based on grid_size to match matrix dimensions
        num_parts = self.grid_size  # Number of parts is equal to the matrix dimension

        # Divide the matrix into equal parts
        part_size = self.grid_size  # Size of each part

        # Choose parts to assign live cells
        parts_with_cells_indices = random.sample(
            range(self.grid_size), self.parts_with_cells)

        total_alive_cells = 0  # This will track the number of alive cells assigned

        # Choose cells for each selected part
        for part_index in parts_with_cells_indices:
            start_idx = part_index * part_size
            end_idx = start_idx + part_size

            # Choose the cells allocated for the selected part
            chosen_cells = random.sample(
                range(start_idx, end_idx), self.cells_per_part)

            # Log the selected block information
            logging.debug(f"Block {part_index} chosen cells: {chosen_cells}")

            # Mark the chosen cells as alive
            for cell in chosen_cells:
                configuration[cell] = 1
                total_alive_cells += 1  # Update total alive cells

        # Log the final configuration
        logging.debug(f"""Generated configuration: {configuration} with {total_alive_cells} alive cells""")

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
                if alive_history:
                    alive_growth = max(alive_history) - sum(alive_history)
                total_alive_cells = sum(alive_history)

            logging.info(f"""Best Configuration in Generation {generation + 1}:""")
            logging.info(f"""    Fitness Score: {best_fitness_score}""")
            logging.info(f"""    Lifespan: {lifespan}""")
            logging.info(f"""    Total Alive Cells: {total_alive_cells}""")
            logging.info(f"""    Alive Growth: {alive_growth}""")

        fitness_scores = [(config, self.fitness(config))
                          for config in self.population]
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
    logging.info(f"Starting run with parameters: grid_size={grid_size}, population_size={population_size}, generations={generations}, mutation_rate={mutation_rate}, alive_cells_weight={alive_cells_weight}, lifespan_weight={lifespan_weight}, alive_growth_weight={alive_growth_weight}, cells_per_part={cells_per_part}, parts_with_cells={parts_with_cells}")

    algorithm = GeneticAlgorithm(
        grid_size, population_size, generations, mutation_rate,
        alive_cells_weight, lifespan_weight, alive_growth_weight,
        cells_per_part=cells_per_part, parts_with_cells=parts_with_cells,
        predefined_configurations=predefined_configurations
    )

    best_configs = algorithm.run()

    simulation = InteractiveSimulation(best_configs, algorithm.best_histories, grid_size)
    simulation.run()


# Example call to main function
main(grid_size=20, population_size=20, generations=20, mutation_rate=0.02,
     alive_cells_weight=50, lifespan_weight=10, alive_growth_weight=5,
     cells_per_part=5, parts_with_cells=2)