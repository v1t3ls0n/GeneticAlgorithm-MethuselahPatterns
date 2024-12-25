
import logging
from GameOfLife import GameOfLife
import random


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
            raise ValueError(f"""Configuration size must be {
                             expected_size}, but got {len(configuration_tuple)}""")

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

        logging.info(f"""Configuration: {configuration_tuple}, Lifespan: {
                     game.lifespan}, Alive Growth: {game.alive_growth}, Alive Cells: {game.total_alive_cells}""")

        # הוספת חישוב ה-Fitness score כך שה-lifespan לא יתאפס
        fitness_score = (game.lifespan * self.lifespan_weight +
                         game.total_alive_cells / self.alive_cells_weight +
                         game.alive_growth * self.alive_growth_weight)

        if game.lifespan == 1:  # אם ה-lifespan לא מתעדכן כראוי, נוודא שהחישוב לא יתבצע
            logging.warning(
                f"""Warning: Lifespan is 1, which may indicate an issue with the simulation!""")

        self.fitness_cache[configuration_tuple] = fitness_score
        return fitness_score

    def mutate(self, configuration):
        N = self.grid_size
        expected_size = N * N

        # Ensure configuration is of the correct size
        if len(configuration) != expected_size:
            logging.error(f"""Configuration size must be {
                          expected_size}, but got {len(configuration)}""")
            raise ValueError(f"""Configuration size must be {
                             expected_size}, but got {len(configuration)}""")

        # Split the vector into an NxN matrix (still one-dimensional)
        # Split the vector into a matrix
        matrix = [configuration[i * N:(i + 1) * N] for i in range(N)]

        # Shuffle columns to create mutations
        blocks = []
        for i in range(N):
            # Create block from N cells in column
            block = [matrix[j][i] for j in range(N)]
            blocks.append(block)

        # Shuffle the blocks
        random.shuffle(blocks)

        # Create the new configuration from the shuffled blocks
        new_configuration = [cell for block in blocks for cell in block]

        # Log the new configuration
        logging.debug(f"""old_configuration : {configuration}""")
        logging.debug(f"""new_configuration : {new_configuration}""")
        logging.debug(f"""old_configuration == new_configuration : {new_configuration==configuration}""")

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
        logging.info(f"""in crossover parent1 : {parent1} parent2 : {parent2}""")
        N = self.grid_size
        mid = N // 2  # Half the size of the matrix

        # Ensure parents are the correct size
        if len(parent1) != N * N or len(parent2) != N * N:
            logging.error(f"""Parent configurations must be {
                          N * N}, but got sizes: {len(parent1)} and {len(parent2)}""")
            raise ValueError(f"""Parent configurations must be {
                             N * N}, but got sizes: {len(parent1)} and {len(parent2)}""")

        # Calculate the sections separated from the first matrix
        father_top_left = [parent1[i * N + j]
                           for i in range(mid) for j in range(mid)]
        father_bottom_right = [
            parent1[(i + mid) * N + (j + mid)] for i in range(mid) for j in range(mid)]

        # Calculate the sections separated from the second matrix
        mother_top_right = [parent2[i * N + (j + mid)]
                            for i in range(mid) for j in range(mid)]
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
            logging.debug(f"""child size mismatch, expected {
                          N * N}, got {len(child)}""")
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
            logging.info(f"""Generation {
                         generation + 1}: Avg Fitness: {avg_fitness}, Std Dev: {std_fitness}""")
            logging.info(f"fitness scores:{fitness_scores}")

            best_fitness_score = max(fitness_scores)
            best_config_index = fitness_scores.index(best_fitness_score)
            best_config = self.population[best_config_index]

            game = GameOfLife(self.grid_size, best_config)
            logging.info(f"""game history:{game.history}\n history length {len(game.history)} uniq history states : {len(set(game.history))}""")
            lifespan = game.get_lifespan()

            alive_growth = 0
            total_alive_cells = 0

            if lifespan > 0:
                alive_history = game.get_alive_history()
                if alive_history:
                    alive_growth = max(alive_history) - sum(alive_history)
                total_alive_cells = sum(alive_history)

            logging.info(f"""Best Configuration in Generation {
                         generation + 1}:""")
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

