
import logging
from GameOfLife import GameOfLife
import random
import math
import numpy as np
import collections


class GeneticAlgorithm:
    def __init__(self, grid_size, population_size, generations, mutation_rate,
                 alive_cells_weight, lifespan_weight, alive_growth_weight,
                 alive_cells_per_block, alive_blocks, predefined_configurations=None, min_fitness_score=1):
        logging.info("Initializing GeneticAlgorithm.")
        self.grid_size = grid_size
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.alive_cells_weight = alive_cells_weight
        self.lifespan_weight = lifespan_weight
        self.alive_growth_weight = alive_growth_weight
        self.configuration_cache = collections.defaultdict(collections.defaultdict)
        self.generations_cache = collections.defaultdict(collections.defaultdict)
        self.predefined_configurations = predefined_configurations
        self.alive_cells_per_block = alive_cells_per_block
        self.alive_blocks = alive_blocks
        self.min_fitness_score = min_fitness_score
        self.best_histories = []
        self.population = []

    def calc_fitness(self,lifespan,total_alive_cells,alive_growth):
        return (lifespan * self.lifespan_weight +
                         total_alive_cells * self.alive_cells_weight +
                         alive_growth * self.alive_growth_weight)
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
        fitness_score = self.calc_fitness(lifespan=game.lifespan,total_alive_cells=game.total_alive_cells, alive_growth=game.alive_growth)
        

        self.configuration_cache[configuration_tuple] = {
                'fitness_score': fitness_score, 'history': tuple(game.history), 
                'lifespan': game.lifespan, 'alive_growth': game.alive_growth, 
                'total_alive_cells':game.total_alive_cells,
                'is_static':game.is_static,
                'is periodic':game._is_periodic,
                }
        
        return self.configuration_cache[configuration_tuple]
       
                

    def populate(self):
        new_population = []
        for i in range(self.population_size):
            parent1, parent2 = self.select_parents()
            child = self.crossover(parent1, parent2)
            if random.uniform(0, 1) < self.mutation_rate:
                child = self.mutate(child)
            new_population.append(child)
        self.population = new_population

    def mutate(self, configuration):
        N = self.grid_size
        expected_size = N * N
        # logging.info(f"""inside mutate, configuration arg = {configuration}\nconfiguration legnth:{len(configuration)}\nexpected size:{expected_size}""")

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
        new_configuration = tuple([cell for block in blocks for cell in block])

        # Log the new configuration
        # logging.info(f"""old_configuration : {configuration}""")
        # logging.info(f"""new_configuration : {new_configuration}""")
        # logging.info(f"""old_configuration == new_configuration : {
        #              new_configuration == configuration}""")

        return new_configuration

    def select_parents(self):
        # Calculate fitness scores for the entire population
        fitness_scores = []
        for config in self.population:
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
            return random.choices(self.population, k=2)

        probabilities = [score / total_fitness for score in fitness_scores]
        parents = random.choices(self.population, weights=probabilities, k=2)
        return parents

    def crossover(self, parent1, parent2):
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

    def initialize(self):
        logging.info(f"""Generation {1} started.""")

        self.population = [self.random_configuration() for _ in range(self.population_size)]
        generation = 0
        scores = []
        lifespans = []
        alive_growth_rates = []
        total_alive_cells = []
        for configuration in self.population:
            self.evaluate(configuration)
            scores.append(self.configuration_cache[configuration]['fitness_score'])
            lifespans.append(self.configuration_cache[configuration]['lifespan'])
            alive_growth_rates.append(self.configuration_cache[configuration]['alive_growth'])
            total_alive_cells.append(self.configuration_cache[configuration]['total_alive_cells'])



        self.generations_cache[generation]['avg_fitness'] = np.average(scores)
        self.generations_cache[generation]['avg_lifespan'] = np.average(lifespans)
        self.generations_cache[generation]['avg_alive_growth_rate'] = np.average(alive_growth_rates)
        self.generations_cache[generation]['avg_total_alive_cells'] = np.average(total_alive_cells)

            # Calculate the standard deviations for each metric
        self.generations_cache[generation]['std_fitness'] = np.std(scores)
        self.generations_cache[generation]['std_lifespan'] = np.std(lifespans)
        self.generations_cache[generation]['std_alive_growth_rate'] = np.std(alive_growth_rates)
        self.generations_cache[generation]['std_total_alive_cells'] = np.std(total_alive_cells)

            # Standardize the values for each metric (mean = avg, std = std)
        self.generations_cache[generation]['std_fitness_values'] = [(x - self.generations_cache[generation]['avg_fitness']) / self.generations_cache[generation]['std_fitness'] for x in scores]
        self.generations_cache[generation]['std_lifespan_values'] = [(x - self.generations_cache[generation]['avg_lifespan']) / self.generations_cache[generation]['std_lifespan'] for x in lifespans]
        self.generations_cache[generation]['std_alive_growth_rate_values'] = [(x - self.generations_cache[generation]['avg_alive_growth_rate']) / self.generations_cache[generation]['std_alive_growth_rate'] for x in alive_growth_rates]
        self.generations_cache[generation]['std_total_alive_cells_values'] = [(x - self.generations_cache[generation]['avg_total_alive_cells']) / self.generations_cache[generation]['std_total_alive_cells'] for x in total_alive_cells]


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

        total_alive_cells = 0  # This will track the number of alive cells assigned

        # Choose cells for each selected part
        for part_index in alive_blocks_indices:
            start_idx = part_index * part_size
            end_idx = start_idx + part_size

            # Choose the cells allocated for the selected part
            chosen_cells = random.sample(
                range(start_idx, end_idx), min(self.alive_cells_per_block,end_idx-start_idx))

            # Log the selected block information
            # logging.info(f"Block {part_index} chosen cells: {chosen_cells}")

            # Mark the chosen cells as alive
            for cell in chosen_cells:
                configuration[cell] = 1
                total_alive_cells += 1  # Update total alive cells

        # Log the final configuration
        # logging.info(f"""Generated configuration: {configuration} with {
        #     total_alive_cells} alive cells""")

        return tuple(configuration)

    def run(self):
        self.initialize()
        for generation in range(1,self.generations):
            logging.info(f"""Generation {generation + 1} started.""")
            self.populate()
            scores = []
            lifespans = []
            alive_growth_rates = []
            total_alive_cells = []

            for configuration in self.population:
                self.evaluate(configuration)
                scores.append(self.configuration_cache[configuration]['fitness_score'])
                lifespans.append(self.configuration_cache[configuration]['lifespan'])
                alive_growth_rates.append(
                    self.configuration_cache[configuration]['alive_growth'])
                total_alive_cells.append(self.configuration_cache[configuration]['total_alive_cells'])
                
                


            self.generations_cache[generation]['avg_fitness'] = np.average(scores)
            self.generations_cache[generation]['avg_lifespan'] = np.average(lifespans)
            self.generations_cache[generation]['avg_alive_growth_rate'] = np.average(alive_growth_rates)
            self.generations_cache[generation]['avg_total_alive_cells'] = np.average(total_alive_cells)

            # Calculate the standard deviations for each metric
            self.generations_cache[generation]['std_fitness'] = np.std(scores)
            self.generations_cache[generation]['std_lifespan'] = np.std(lifespans)
            self.generations_cache[generation]['std_alive_growth_rate'] = np.std(alive_growth_rates)
            self.generations_cache[generation]['std_total_alive_cells'] = np.std(total_alive_cells)


            # Then use the above function in the code as follows:
            self.generations_cache[generation]['std_fitness_values'] = self.safe_standardize(scores, self.generations_cache[generation]['avg_fitness'], self.generations_cache[generation]['std_fitness'])
            self.generations_cache[generation]['std_lifespan_values'] = self.safe_standardize(lifespans, self.generations_cache[generation]['avg_lifespan'], self.generations_cache[generation]['std_lifespan'])
            self.generations_cache[generation]['std_alive_growth_rate_values'] = self.safe_standardize(alive_growth_rates, self.generations_cache[generation]['avg_alive_growth_rate'], self.generations_cache[generation]['std_alive_growth_rate'])
            self.generations_cache[generation]['std_total_alive_cells_values'] = self.safe_standardize(total_alive_cells, self.generations_cache[generation]['avg_total_alive_cells'], self.generations_cache[generation]['std_total_alive_cells'])


        logging.info(f"""population size = {len(set(self.population))}""")
        fitness_scores = [(config, self.configuration_cache[config]['fitness_score'])
                          for config in self.population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        # logging.info(f"""fitness score sorted : {fitness_scores}""")

        best_configs = fitness_scores[:5]


        for config,_ in best_configs:
            # Use history from the cache
            history = self.configuration_cache[config]['history']
            self.best_histories.append(list(history))

            logging.info(f"""Top {config} Configuration:""")
            logging.info(f"""  Fitness Score: {self.configuration_cache[config]['fitness_score']}""")
            logging.info(f"""  Lifespan: {self.configuration_cache[config]['lifespan']}""")
            logging.info(f"""  Total Alive Cells: {self.configuration_cache[config]['total_alive_cells']}""")
            logging.info(f"""  Alive Growth: {self.configuration_cache[config]['alive_growth']}""")

        return best_configs
    
    def safe_standardize(self,values, avg, std):
        if std == 0:
            logging.warning(f"Standard deviation is zero, returning zero for all values. Avg: {avg}, Std: {std}")
            return [0] * len(values)  # Return zero if std is zero to avoid division by zero
        return [(x - avg) / std for x in values]
