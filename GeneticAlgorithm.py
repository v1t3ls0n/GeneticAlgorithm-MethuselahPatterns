
import logging
from GameOfLife import GameOfLife
import random
import math
import numpy as np
import collections


class GeneticAlgorithm:
    def __init__(self, grid_size, population_size, generations, mutation_rate,
                 alive_cells_weight, lifespan_weight, alive_growth_weight,stableness_weight,
                 alive_cells_per_block, alive_blocks, predefined_configurations=None, min_fitness_score=1):
        logging.info("Initializing GeneticAlgorithm.")
        self.grid_size = grid_size
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.alive_cells_weight = alive_cells_weight
        self.lifespan_weight = lifespan_weight
        self.lifespan_threshold = (grid_size * grid_size) * 3
        self.alive_growth_weight = alive_growth_weight
        self.stableness_weight = stableness_weight
        self.configuration_cache = collections.defaultdict(
            collections.defaultdict)
        self.generations_cache = collections.defaultdict(
            collections.defaultdict)
        self.predefined_configurations = predefined_configurations
        self.alive_cells_per_block = alive_cells_per_block
        self.alive_blocks = alive_blocks
        self.min_fitness_score = min_fitness_score
        self.best_histories = []
        self.population = set()

    def calc_fitness(self, lifespan, max_alive_cells_count, alive_growth, stableness):
        return (lifespan * self.lifespan_weight + max_alive_cells_count * self.alive_cells_weight + alive_growth * self.alive_growth_weight + stableness * self.stableness_weight) 

    def evaluate(self, configuration):
        configuration_tuple = tuple(configuration)
        expected_size = self.grid_size * self.grid_size

        # Ensure the configuration is of the correct size
        if len(configuration_tuple) != expected_size:
            raise ValueError(f"""Configuration size must be {
                             expected_size}, but got {len(configuration_tuple)}""")

        if configuration_tuple in self.configuration_cache:
            return self.configuration_cache[configuration_tuple]

        # Create an instance of GameOfLife with the current configuration
        game = GameOfLife(self.grid_size, configuration_tuple)
        game.run()  # Run the simulation
        fitness_score = self.calc_fitness(
            lifespan=game.lifespan, max_alive_cells_count=game.max_alive_cells_count, alive_growth=game.alive_growth, stableness=game.stableness)
        # logging.info(f"""configuration is static?{game.is_static}\n""")
        # logging.info(f"""configuration is peridioc?{game.is_periodic}\n""")

        self.configuration_cache[configuration_tuple] = {
            'fitness_score': fitness_score, 'history': tuple(game.history),
            'lifespan': game.lifespan, 'alive_growth': game.alive_growth,
            'max_alive_cells_count': game.max_alive_cells_count,
            'is_static': game.is_static,
            'is periodic': game.is_periodic,
            'stableness': game.stableness
        }

        return self.configuration_cache[configuration_tuple]

    def populate(self):
        new_population = set()  # לא נרצה כפילויות, נשתמש ב-set
        for i in range(self.population_size):
            parent1, parent2 = self.select_parents()
            child = self.crossover(parent1, parent2)
            if random.uniform(0, 1) < self.mutation_rate:
                child = self.mutate(child)
            new_population.add(child)

        # עכשיו נשלב את האוכלוסיות: הישנה והחדשה
        combined_population = list(self.population) + list(new_population)

        # חישוב הכושר של כל הקונפיגורציות
        fitness_scores = [(config, self.evaluate(config)['fitness_score'])
                          for config in combined_population]
        # מיון לפי ציון כושר, מהטוב ביותר לפחות טוב
        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        # שמירה רק על הקונפיגורציות הטובות ביותר
        # אנחנו שומרים את ה-set, אבל כדי לשמור על הסדר (נחוץ כדי לבחור את הטובות ביותר)
        self.population = [config for config,
                           _ in fitness_scores[:self.population_size]]

        # הפיכת ה-population ל-set כדי לשמור על הייחודיות (למנוע כפילויות)
        self.population = set(self.population)

    def mutate(self, configuration):
        N = self.grid_size
        total_cells = N * N

        # Split the vector into a matrix (NxN grid)
        matrix = [configuration[i * N:(i + 1) * N] for i in range(N)]

        # Create blocks from the matrix
        blocks = []
        for i in range(N):
            # Create block from N cells in column
            block = [matrix[j][i] for j in range(N)]
            blocks.append(block)

        # Calculate the number of living cells in each block
        block_alive_counts = [sum(block) for block in blocks]

        # Create a list of tuples with each block and its corresponding count of living cells
        blocks_with_alive_counts = list(zip(blocks, block_alive_counts))

        # Sort the blocks by the number of living cells (ascending order)
        blocks_with_alive_counts.sort(key=lambda x: x[1])

        # Create a list of blocks sorted by their living cell count
        sorted_blocks = [block for block, _ in blocks_with_alive_counts]

        # Shuffle the sorted blocks to create a new configuration
        random.shuffle(sorted_blocks)

        # Now, create the new configuration based on shuffled blocks
        new_configuration = []

        # Modify the blocks based on the shuffled order
        for block in sorted_blocks:
            for j in range(N):
                # For each cell in the block, decide if it should remain the same or mutate
                if block[j] == 1:
                    # If the cell is alive, it remains alive
                    new_configuration.append(1)
                else:
                    # If the cell is dead, mutate based on the block's life probability
                    if random.uniform(0, 1) < sum(block) / N:
                        new_configuration.append(1)
                    else:
                        new_configuration.append(0)

        return tuple(new_configuration)

    def select_parents(self):
        # Calculate fitness scores for the entire population
        fitness_scores = []
        population = list(self.population)
        for config in population:
            score = self.evaluate(config)['fitness_score']
            if score is not None:
                fitness_scores.append(score)
            else:
                # If we failed to compute fitness, assign it a minimum score (e.g., 0)
                fitness_scores.append(0)

        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            # If all scores are 0, choose random parents
            logging.info("Total fitness is 0, selecting random parents.")
            return random.choices(population, k=2)

        probabilities = [score / total_fitness for score in fitness_scores]
        parents = random.choices(population, weights=probabilities, k=2)
        return parents

    def crossover_old_basic(self, parent1, parent2):
        N = self.grid_size
        total_cells = N * N

        # ווידוא שההורים הם בגודל הנכון
        if len(parent1) != total_cells or len(parent2) != total_cells:
            logging.info(f"""Parent configurations must be {total_cells}, but got sizes: {
                len(parent1)} and {len(parent2)}""")
            raise ValueError(f"""Parent configurations must be {
                             total_cells}, but got sizes: {len(parent1)} and {len(parent2)}""")

        # נחלק את הווקטור ל-N בלוקים
        block_size = total_cells // N
        blocks_parent1 = [
            parent1[i * block_size:(i + 1) * block_size] for i in range(N)]
        blocks_parent2 = [
            parent2[i * block_size:(i + 1) * block_size] for i in range(N)]

        # נבחר את הבלוקים לפי האינדקסים האי זוגיים של האב והזוגיים של האם
        child_blocks = []
        for i in range(N):
            if i % 2 == 0:  # אינדקסים זוגיים (מאמא)
                child_blocks.extend(blocks_parent2[i])
            else:  # אינדקסים אי זוגיים (מאבא)
                child_blocks.extend(blocks_parent1[i])

        # ווידוא שהילד בגודל הנכון
        if len(child_blocks) != total_cells:
            logging.info(f"""Child size mismatch, expected {
                         total_cells}, got {len(child_blocks)}""")
            # במקרה של שגיאת גודל, נוסיף תאים חיים נוספים
            child_blocks = child_blocks + [0] * \
                (total_cells - len(child_blocks))

        return tuple(child_blocks)

    def crossover(self, parent1, parent2):
        """
        Performs crossover between two parent configurations to create a child configuration.

        The crossover is done by dividing each parent into blocks, selecting blocks based on 
        the number of living cells in each block, and combining the selected blocks from the parent
        with more living cells, while giving it a higher probability for block selection.

        Args:
            parent1 (tuple): The first parent configuration.
            parent2 (tuple): The second parent configuration.

        Returns:
            tuple: The child configuration generated by combining selected blocks from both parents.

        Raises:
            ValueError: If the sizes of parent1 or parent2 do not match the expected grid size.
        """
        N = self.grid_size
        total_cells = N * N
        reminder = N % 2  # Handle remainder for odd N

        # Ensure that the parent configurations are of the correct size
        if len(parent1) != total_cells or len(parent2) != total_cells:
            logging.info(f"""Parent configurations must be {total_cells}, but got sizes: {
                         len(parent1)} and {len(parent2)}""")
            raise ValueError(f"""Parent configurations must be {
                             total_cells}, but got sizes: {len(parent1)} and {len(parent2)}""")

        # Divide the configurations into N blocks
        block_size = total_cells // N
        blocks_parent1 = [
            parent1[i * block_size:(i + 1) * block_size] for i in range(N)]
        blocks_parent2 = [
            parent2[i * block_size:(i + 1) * block_size] for i in range(N)]

        # Count the number of alive cells in each block
        block_alive_counts_parent1 = [sum(block) for block in blocks_parent1]
        block_alive_counts_parent2 = [sum(block) for block in blocks_parent2]

        # Calculate the total number of alive cells in both parents
        total_alive_cells_parent1 = sum(block_alive_counts_parent1)
        total_alive_cells_parent2 = sum(block_alive_counts_parent2)

        # Assign probabilities for block selection for parent1
        if total_alive_cells_parent1 > 0:
            probabilities_parent1 = [
                (alive_count / total_alive_cells_parent1) if alive_count > 0 else (1 / total_cells) for alive_count in block_alive_counts_parent1
            ]
        else:
            # If no alive cells, assign equal probability to all blocks
            probabilities_parent1 = [1 / total_cells] * N

        # Assign probabilities for block selection for parent2
        if total_alive_cells_parent2 > 0:
            probabilities_parent2 = [
                (alive_count / total_alive_cells_parent2) if alive_count > 0 else (1 / total_cells) for alive_count in block_alive_counts_parent2
            ]
        else:
            # If no alive cells, assign equal probability to all blocks
            probabilities_parent2 = [1 / total_cells] * N

        # Select blocks from the parent with more living cells based on calculated probabilities
        selected_blocks_parent1 = random.choices(
            range(N), weights=probabilities_parent1, k=(N // 2) + reminder)

        # Now, select blocks from parent2 excluding the already selected blocks from parent1
        remaining_blocks_parent2 = [i for i in range(
            N) if i not in selected_blocks_parent1]
        selected_blocks_parent2 = random.choices(
            remaining_blocks_parent2,
            weights=[probabilities_parent2[i]
                     for i in remaining_blocks_parent2],
            k=N // 2
        )

        # Create the child configuration by combining selected blocks from both parents
        child_blocks = []
        for i in range(N):
            if i in selected_blocks_parent1:
                child_blocks.extend(blocks_parent1[i])
            elif i in selected_blocks_parent2:
                child_blocks.extend(blocks_parent2[i])
            else:
                # If the block is not selected from either parent, choose randomly between them
                selected_parent = random.choices(
                    [1, 2], weights=[0.5, 0.5], k=1)[0]
                if selected_parent == 1:
                    child_blocks.extend(blocks_parent1[i])
                else:
                    child_blocks.extend(blocks_parent2[i])

        # Ensure the child configuration has the correct number of cells (in case of rounding issues)
        if len(child_blocks) != total_cells:
            logging.info(f"""Child size mismatch, expected {
                         total_cells}, got {len(child_blocks)}""")
            # Ensure that we pad or adjust the final size to match the expected number of cells
            child_blocks = child_blocks + \
                [0] * (total_cells - len(child_blocks))  # Padding if necessary

        return tuple(child_blocks)

    def initialize(self):
        logging.info(f"""Generation {1} started.""")

        self.population = [self.random_configuration()
                           for _ in range(self.population_size)]
        generation = 0
        scores = []
        lifespans = []
        alive_growth_rates = []
        max_alive_cells_count = []
        stableness = []
        for configuration in self.population:
            self.evaluate(configuration)
            scores.append(
                self.configuration_cache[configuration]['fitness_score'])
            lifespans.append(
                self.configuration_cache[configuration]['lifespan'])
            alive_growth_rates.append(
                self.configuration_cache[configuration]['alive_growth'])
            max_alive_cells_count.append(
                self.configuration_cache[configuration]['max_alive_cells_count'])
            stableness.append(
                self.configuration_cache[configuration]['stableness'])

        self.calc_statistics(generation=generation, scores=scores, lifespans=lifespans,
                             alive_growth_rates=alive_growth_rates, max_alive_cells_count=max_alive_cells_count, stableness=stableness)

        self.generations_cache[generation]['avg_fitness'] = np.average(scores)
        self.generations_cache[generation]['avg_lifespan'] = np.average(
            lifespans)
        self.generations_cache[generation]['avg_alive_growth_rate'] = np.average(
            alive_growth_rates)
        self.generations_cache[generation]['avg_max_alive_cells_count'] = np.average(
            max_alive_cells_count)

        # Calculate the standard deviations for each metric
        self.generations_cache[generation]['std_fitness'] = np.std(scores)
        self.generations_cache[generation]['std_lifespan'] = np.std(lifespans)
        self.generations_cache[generation]['std_alive_growth_rate'] = np.std(
            alive_growth_rates)
        self.generations_cache[generation]['std_max_alive_cells_count'] = np.std(
            max_alive_cells_count)

    def random_configuration(self):
        # Size of the matrix (grid_size * grid_size)
        N = self.grid_size * self.grid_size
        configuration = [0] * N  # Initialize matrix with zeros

        # Calculate num_parts based on grid_size to match matrix dimensions
        num_parts = self.grid_size  # Number of parts is equal to the matrix dimension

        # Divide the matrix into equal parts
        part_size = self.grid_size  # Size of each part

        # Choose parts to assign live cells
        alive_blocks_indices = random.sample(
            range(self.grid_size), self.alive_blocks)

        max_alive_cells_count = 0  # This will track the number of alive cells assigned

        # Choose cells for each selected part
        for part_index in alive_blocks_indices:
            start_idx = part_index * part_size
            end_idx = start_idx + part_size

            # Choose the cells allocated for the selected part
            chosen_cells = random.sample(
                range(start_idx, end_idx), min(self.alive_cells_per_block, end_idx-start_idx))

            # Log the selected block information
            # logging.info(f"Block {part_index} chosen cells: {chosen_cells}")

            # Mark the chosen cells as alive
            for cell in chosen_cells:
                configuration[cell] = 1
                max_alive_cells_count += 1  # Update total alive cells

        # Log the final configuration
        # logging.info(f"""Generated configuration: {configuration} with {
        #     max_alive_cells_count} alive cells""")

        return tuple(configuration)

    def calc_statistics(self, generation, scores, lifespans, alive_growth_rates, stableness, max_alive_cells_count):
        scores = np.array(scores)
        lifespans = np.array(lifespans)
        alive_growth_rates = np.array(alive_growth_rates)
        stableness = np.array(stableness)
        self.generations_cache[generation]['avg_fitness'] = np.average(
            scores)
        self.generations_cache[generation]['avg_lifespan'] = np.average(
            lifespans)
        self.generations_cache[generation]['avg_alive_growth_rate'] = np.average(
            alive_growth_rates)
        self.generations_cache[generation]['avg_max_alive_cells_count'] = np.average(
            max_alive_cells_count)
        self.generations_cache[generation]['avg_stableness'] = np.average(
            stableness)
        # Calculate the standard deviations for each metric
        self.generations_cache[generation]['std_fitness'] = np.std(scores)
        self.generations_cache[generation]['std_lifespan'] = np.std(
            lifespans)
        self.generations_cache[generation]['std_alive_growth_rate'] = np.std(
            alive_growth_rates)
        self.generations_cache[generation]['std_max_alive_cells_count'] = np.std(
            max_alive_cells_count)

    def run(self):
        self.initialize()
        for generation in range(1, self.generations):
            logging.info(f"""Generation {generation + 1} started.""")

            self.populate()
            scores = []
            lifespans = []
            alive_growth_rates = []
            max_alive_cells_count = []
            stableness = []
            for configuration in self.population:
                self.evaluate(configuration)
                scores.append(
                    self.configuration_cache[configuration]['fitness_score'])
                lifespans.append(
                    self.configuration_cache[configuration]['lifespan'])
                alive_growth_rates.append(
                    self.configuration_cache[configuration]['alive_growth'])
                max_alive_cells_count.append(
                    self.configuration_cache[configuration]['max_alive_cells_count'])
                stableness.append(
                    self.configuration_cache[configuration]['stableness'])

            self.calc_statistics(generation=generation, scores=scores, lifespans=lifespans,
                                 alive_growth_rates=alive_growth_rates, max_alive_cells_count=max_alive_cells_count, stableness=stableness)
            prev_fitness = self.generations_cache[generation - 1]['avg_fitness']
            curr_fitness = self.generations_cache[generation]['avg_fitness']
            fitness_improvement_rate = curr_fitness / prev_fitness if prev_fitness != 0 else 1
            self.mutation_rate = max(0.2, self.mutation_rate * fitness_improvement_rate)


            if self.generations_cache[generation]['std_fitness'] < self.lifespan_threshold:  # low fitness standard deviation
                self.mutation_rate = self.mutation_rate / 1.5  # Reduce mutation rate significantly

        logging.info(f"""population size = {len(set(self.population))}""")
        fitness_scores = [(config, self.configuration_cache[config]['fitness_score'])
                          for config in self.population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        best_configs = fitness_scores[:5]
        logging.info(f"""fitness score length : {len(fitness_scores)}""")
        logging.info(f"""best configs length : {len(best_configs)}""")
        logging.info(f"""best configs: {best_configs}""")
        logging.info(f"""all current population: {self.population}""")

        for config, _ in best_configs:
            # Use history from the cache
            history = list(self.configuration_cache[config]['history'])
            self.best_histories.append(history)

            logging.info(f"""Top {config} Configuration:""")
            logging.info(f"""  Fitness Score: {
                         self.configuration_cache[config]['fitness_score']}""")
            logging.info(f"""  Lifespan: {
                         self.configuration_cache[config]['lifespan']}""")
            logging.info(f"""  Total Alive Cells: {
                         self.configuration_cache[config]['max_alive_cells_count']}""")
            logging.info(f"""  Alive Growth: {
                         self.configuration_cache[config]['alive_growth']}""")

        return best_configs

    # def safe_standardize(self,values, avg, std):
    #     if std == 0:
    #         logging.warning(f"Standard deviation is zero, returning zero for all values. Avg: {avg}, Std: {std}")
    #         return [0] * len(values)  # Return zero if std is zero to avoid division by zero
    #     return [(x - avg) / std for x in values]
