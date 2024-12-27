"""
GeneticAlgorithm.py
-------------------

Implements a genetic algorithm to evolve initial configurations for Conway's Game of Life,
optimizing for extended lifespan, higher maximum living cells, greater growth ratio, etc.

Classes:
    GeneticAlgorithm: Manages population evolution through selection, crossover, and mutation
    to optimize Game of Life configurations.
"""

import logging
from GameOfLife import GameOfLife
import random
import math
import numpy as np
import collections


class GeneticAlgorithm:
    """
    GeneticAlgorithm manages the evolution of GameOfLife configurations using genetic principles.
    It iteratively optimizes configurations over multiple generations using fitness evaluation,
    selection, crossover, and mutation.

    Attributes:
        grid_size (int): The NxN grid size for GameOfLife configurations.
        population_size (int): Number of individuals in each population.
        generations (int): Total number of generations to simulate.
        initial_mutation_rate (float): Initial probability of mutation per cell.
        mutation_rate_lower_limit (float): Minimum mutation rate value.
        alive_cells_weight (float): Weight for the maximum number of alive cells in fitness.
        lifespan_weight (float): Weight for lifespan in the fitness score.
        alive_growth_weight (float): Weight for alive cell growth ratio in fitness.
        stableness_weight (float): Weight for stability of configurations in fitness.
        initial_living_cells_count_weight (float): Weight for penalizing large initial configurations.
        alive_cells_per_block (int): Maximum alive cells allowed per block in random initialization.
        alive_blocks (int): Number of blocks containing alive cells in random initialization.
        predefined_configurations (optional): Allows injecting pre-made Game of Life configurations.
        population (set[tuple]): Current population of configurations (unique).
        configuration_cache (dict): Stores previously evaluated configurations and results.
        generations_cache (dict): Tracks statistics (e.g., fitness) for each generation.
        best_histories (list): Histories of top configurations for each generation.
        mutation_rate_history (list): Tracks mutation rate changes across generations.
        lifespan_threshold (int): Threshold used for additional lifespan-based logic (unused).
    """

    def __init__(self, grid_size, population_size, generations, initial_mutation_rate, mutation_rate_lower_limit,
                 alive_cells_weight, lifespan_weight, alive_growth_weight, stableness_weight, initial_living_cells_count_weight,
                 alive_cells_per_block, alive_blocks, predefined_configurations=None):
        """
        Initialize the GeneticAlgorithm class with key parameters.

        Args:
            grid_size (int): NxN grid size.
            population_size (int): Number of individuals per generation.
            generations (int): Total generations to simulate.
            initial_mutation_rate (float): Initial probability of mutation.
            mutation_rate_lower_limit (float): Minimum mutation rate value.
            alive_cells_weight (float): Weight factor for alive cells in fitness.
            lifespan_weight (float): Weight factor for lifespan in fitness.
            alive_growth_weight (float): Weight factor for alive cell growth ratio in fitness.
            stableness_weight (float): Weight factor for configuration stability.
            initial_living_cells_count_weight (float): Penalizes larger initial configurations.
            alive_cells_per_block (int): Maximum alive cells per block for random initialization.
            alive_blocks (int): Number of blocks to initialize with alive cells.
            predefined_configurations (optional): Allows using predefined patterns for initialization.
        """
        print("Initializing GeneticAlgorithm.")
        self.grid_size = grid_size
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate_lower_limit = mutation_rate_lower_limit
        self.mutation_rate = initial_mutation_rate
        self.alive_cells_weight = alive_cells_weight
        self.lifespan_weight = lifespan_weight
        self.initial_living_cells_count_weight = initial_living_cells_count_weight
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
        self.best_histories = []
        self.best_params = []
        self.population = set()
        self.mutation_rate_history = [initial_mutation_rate]

    def calc_fitness(self, lifespan, max_alive_cells_count, alive_growth, stableness, initial_living_cells_count):
        """
        Calculates the fitness score of a configuration by combining key metrics
        weighted by their respective coefficients.

        Args:
            lifespan (int): Total number of unique states before stopping.
            max_alive_cells_count (int): Maximum living cells in any generation.
            alive_growth (float): Ratio of max to min living cells across generations.
            stableness (float): Stability score based on repeated static or periodic patterns.
            initial_living_cells_count (int): Number of alive cells in the initial configuration.

        Returns:
            float: A fitness score computed as a weighted sum of the provided metrics.
        """
        lifespan_score = lifespan * self.lifespan_weight
        alive_cells_score = max_alive_cells_count * self.alive_cells_weight
        growth_score = alive_growth * self.alive_growth_weight
        stableness_score = stableness * self.stableness_weight
        small_configuration_score = self.initial_living_cells_count_weight * \
            (1 / max(1, initial_living_cells_count))
        return (lifespan_score + alive_cells_score + growth_score + stableness_score) * small_configuration_score

    def evaluate(self, configuration):
        """
        Evaluates a configuration by simulating its evolution in GameOfLife
        and calculating its fitness score. Results are cached to avoid redundant calculations.

        Args:
            configuration (tuple[int]): Flattened 1D representation of NxN GameOfLife grid.

        Returns:
            dict: Simulation results including fitness, lifespan, and other statistics.
        """
        configuration_tuple = tuple(configuration)
        expected_size = self.grid_size * self.grid_size
        if len(configuration_tuple) != expected_size:
            raise ValueError(f"""Configuration size must be {
                             expected_size}, but got {len(configuration_tuple)}""")

        if configuration_tuple in self.configuration_cache:
            return self.configuration_cache[configuration_tuple]

        # Create and run a GameOfLife instance
        game = GameOfLife(self.grid_size, configuration_tuple)
        game.run()
        initial_living_cells_count = sum(configuration_tuple)
        fitness_score = self.calc_fitness(
            lifespan=game.lifespan,
            max_alive_cells_count=game.max_alive_cells_count,
            alive_growth=game.alive_growth,
            stableness=game.stableness,
            initial_living_cells_count=initial_living_cells_count
        )

        self.configuration_cache[configuration_tuple] = {
            'fitness_score': fitness_score,
            'history': tuple(game.history),
            'lifespan': game.lifespan,
            'alive_growth': game.alive_growth,
            'max_alive_cells_count': game.max_alive_cells_count,
            'is_static': game.is_static,
            'is periodic': game.is_periodic,
            'stableness': game.stableness,
            'initial_living_cells_count': initial_living_cells_count
        }
        return self.configuration_cache[configuration_tuple]

    def populate(self):
        """
        Generate a new generation of configurations for the population.

        Process:
            1. Select two parent configurations from the current population based on fitness.
            2. Create a child configuration using crossover between the two parents.
            3. Apply mutation to the child with a probability determined by the mutation rate.
            4. Combine the new children with the existing population.
            5. Evaluate all configurations and retain only the top `population_size` individuals.

        Ensures:
            - The population evolves towards higher fitness by keeping top performers.
            - The population remains diverse through mutation.

        Returns:
            None: Updates the `population` attribute in place.
        """
        new_population = set()
        for i in range(self.population_size):
            parent1, parent2 = self.select_parents()
            child = self.crossover(parent1, parent2)
            if random.uniform(0, 1) < self.mutation_rate:
                child = self.mutate(child)
            new_population.add(child)

        # Merge old + new, then select the best
        combined_population = list(self.population) + list(new_population)
        fitness_scores = [(config, self.evaluate(config)['fitness_score'])
                          for config in combined_population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        # Keep only the best population_size
        self.population = [config for config,
                           _ in fitness_scores[:self.population_size]]
        # Convert back to set for uniqueness
        self.population = set(self.population)

    def mutate(self, configuration):
        """
        Perform mutation on a given configuration by flipping some dead cells (0 -> 1).

        Mutation Process:
            - Each cell in the configuration has a chance to flip to 1 (alive) if it is currently 0.
            - The chance of flipping is determined by `self.mutation_rate`.
            - Live cells (1) are not flipped to 0 in this implementation but could be added as an extension.

        Args:
            configuration (tuple[int]): A flattened NxN grid of 0s and 1s representing the configuration.

        Returns:
            tuple[int]: A new configuration with mutations applied.
        """
        N = self.grid_size
        matrix = [configuration[i*N:(i+1)*N] for i in range(N)]
        new_configuration = []
        for row in matrix:
            for cell in row:
                # If cell is dead, there's a chance we flip it to alive
                if cell == 0 and random.uniform(0, 1) < self.mutation_rate:
                    new_configuration.append(1)
                else:
                    new_configuration.append(cell)
        return tuple(new_configuration)

    def select_parents(self):
        """
        Select two parent configurations from the current population for crossover.

        Selection Process:
            - Fitness scores are calculated for all individuals in the population.
            - A probability distribution is created based on the fitness scores, with higher scores
            having a greater chance of selection.
            - If the total fitness is 0 (e.g., all configurations have identical fitness), selection
            is done randomly.

        Returns:
            tuple: Two parent configurations selected from the population.
        """
        population = list(self.population)
        fitness_scores = []
        for config in population:
            score = self.evaluate(config)['fitness_score']
            if score is not None:
                fitness_scores.append(score)
            else:
                fitness_scores.append(0)

        total_fitness = sum(fitness_scores)
        avg_fitness = np.average(fitness_scores)
        fitness_variance = np.std(fitness_scores)
        if total_fitness == 0:
            # All zero => random
            logging.info("Total fitness is 0, selecting random parents.")
            return random.choices(population, k=2)

        # This probability model is somewhat ad-hoc and can be refined
        probabilities = [(1+avg_fitness)*(1 + fitness_variance)
                         for _score in fitness_scores]
        parents = random.choices(population, weights=probabilities, k=2)
        return parents

    def crossover(self, parent1, parent2):
        """
        Create a child configuration by combining blocks from two parent configurations.

        Crossover Process:
            - Divide each parent's configuration into blocks of size `block_size`.
            - Select blocks from each parent based on the ratio of living cells in each block.
            - Combine selected blocks to form a new child configuration.
            - If a block is not chosen from either parent, randomly select one parent for that block.

        Args:
            parent1 (tuple[int]): A flattened NxN configuration (first parent).
            parent2 (tuple[int]): A flattened NxN configuration (second parent).

        Returns:
            tuple[int]: A new child configuration created by combining blocks from both parents.
        """
        N = self.grid_size
        total_cells = N*N
        reminder = N % 2

        if len(parent1) != total_cells or len(parent2) != total_cells:
            logging.info(f"""Parent configurations must be {total_cells}, but got sizes: {
                         len(parent1)} and {len(parent2)}""")
            raise ValueError(f"""Parent configurations must be {
                             total_cells}, but got sizes: {len(parent1)} and {len(parent2)}""")

        block_size = total_cells // N
        blocks_parent1 = [
            parent1[i*block_size:(i+1)*block_size] for i in range(N)]
        blocks_parent2 = [
            parent2[i*block_size:(i+1)*block_size] for i in range(N)]

        block_alive_counts_parent1 = [sum(block) for block in blocks_parent1]
        block_alive_counts_parent2 = [sum(block) for block in blocks_parent2]
        max_alive_cells_parent1 = sum(block_alive_counts_parent1)
        max_alive_cells_parent2 = sum(block_alive_counts_parent2)

        # Probability assignment
        if max_alive_cells_parent1 > 0:
            probabilities_parent1 = [(alive_count / max_alive_cells_parent1) if alive_count > 0 else (1/total_cells)
                                     for alive_count in block_alive_counts_parent1]
        else:
            probabilities_parent1 = [1/total_cells]*N

        if max_alive_cells_parent2 > 0:
            probabilities_parent2 = [(alive_count / max_alive_cells_parent2) if alive_count > 0 else (1/total_cells)
                                     for alive_count in block_alive_counts_parent2]
        else:
            probabilities_parent2 = [1/total_cells]*N

        selected_blocks_parent1 = random.choices(
            range(N), weights=probabilities_parent1, k=(N//2)+reminder)
        remaining_blocks_parent2 = [i for i in range(
            N) if i not in selected_blocks_parent1]
        selected_blocks_parent2 = random.choices(
            remaining_blocks_parent2,
            weights=[probabilities_parent2[i]
                     for i in remaining_blocks_parent2],
            k=N//2
        )

        child_blocks = []
        for i in range(N):
            if i in selected_blocks_parent1:
                child_blocks.extend(blocks_parent1[i])
            elif i in selected_blocks_parent2:
                child_blocks.extend(blocks_parent2[i])
            else:
                # If not chosen from either, pick randomly
                selected_parent = random.choices(
                    [1, 2], weights=[0.5, 0.5], k=1)[0]
                if selected_parent == 1:
                    child_blocks.extend(blocks_parent1[i])
                else:
                    child_blocks.extend(blocks_parent2[i])

        # Fix length if needed
        if len(child_blocks) != total_cells:
            logging.info(f"""Child size mismatch, expected {
                         total_cells}, got {len(child_blocks)}""")
            child_blocks = child_blocks + [0]*(total_cells - len(child_blocks))
        return tuple(child_blocks)

    def initialize(self):
        """
        Initialize the population with random configurations and evaluate their fitness.

        Initialization Process:
            - Create `population_size` random configurations.
            - Evaluate the fitness of each configuration using the `evaluate` method.
            - Calculate and store initial statistics (average fitness, lifespan, etc.) for generation 0.

        Returns:
            None: Updates the `population` attribute and initializes the first generation's statistics.
        """

        print(f"Generation 1 started.")
        self.population = [self.random_configuration()
                           for _ in range(self.population_size)]

        generation = 0
        scores = []
        lifespans = []
        alive_growth_rates = []
        max_alive_cells_count = []
        stableness = []
        initial_living_cells_count=[]
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
            initial_living_cells_count.append(self.configuration_cache[configuration]['initial_living_cells_count'])

        self.calc_statistics(generation=generation,
                             scores=scores,
                             lifespans=lifespans,
                             alive_growth_rates=alive_growth_rates,
                             max_alive_cells_count=max_alive_cells_count,
                             stableness=stableness,
                             initial_living_cells_count = initial_living_cells_count
                             )
    def random_configuration(self):
        """
        Randomly generate an NxN configuration (flattened to length N*N).
        The grid is divided conceptually into `grid_size` parts. We randomly select
        `alive_blocks` of those parts, and within each part, we randomly turn
        `alive_cells_per_block` cells into 1 (alive).

        Returns:
            tuple[int]: A new random configuration (flattened).
        """
        N = self.grid_size * self.grid_size
        configuration = [0]*N

        partition_size = self.grid_size
        alive_blocks_indices = random.sample(
            range(self.alive_blocks), k=self.alive_blocks)
        total_taken_cells = self.alive_blocks * self.alive_cells_per_block
        for part_index in alive_blocks_indices:
            start_idx = part_index*partition_size
            end_idx = start_idx + partition_size
            chosen_cells = random.sample(
                range(start_idx, end_idx), k=self.alive_cells_per_block)
            for cell in chosen_cells:
                configuration[cell] = 1
                total_taken_cells -= 1
        return tuple(configuration)

    def calc_statistics(self, generation, scores, lifespans, alive_growth_rates, stableness, max_alive_cells_count, initial_living_cells_count):
        """
        Record the average and standard deviation of each metric for the population at this generation.

        Args:
            generation (int): Which generation we're recording.
            scores (list[float]): Fitness values of all individuals in the population.
            lifespans (list[int]): Lifespan (unique states) for each individual.
            alive_growth_rates (list[float]): alive_growth metric for each individual.
            stableness (list[float]): how stable or unstable each individual ended up.
            max_alive_cells_count (list[int]): maximum number of living cells encountered for each.
        """
        scores = np.array(scores)
        lifespans = np.array(lifespans)
        alive_growth_rates = np.array(alive_growth_rates)
        stableness = np.array(stableness)
        max_alive_cells_count = np.array(max_alive_cells_count)

        self.generations_cache[generation]['avg_fitness'] = np.mean(scores)
        self.generations_cache[generation]['avg_lifespan'] = np.mean(lifespans)
        self.generations_cache[generation]['avg_alive_growth_rate'] = np.mean(
            alive_growth_rates)
        self.generations_cache[generation]['avg_max_alive_cells_count'] = np.mean(
            max_alive_cells_count)
        self.generations_cache[generation]['avg_stableness'] = np.mean(
            stableness)
        self.generations_cache[generation]['avg_initial_living_cells_count'] = np.mean(initial_living_cells_count)

        self.generations_cache[generation]['std_fitness'] = np.std(scores)
        self.generations_cache[generation]['std_lifespan'] = np.std(lifespans)
        self.generations_cache[generation]['std_alive_growth_rate'] = np.std(
            alive_growth_rates)
        self.generations_cache[generation]['std_max_alive_cells_count'] = np.std(
            max_alive_cells_count)
        self.generations_cache[generation]['std_initial_living_cells_count'] = np.std(initial_living_cells_count)

    def run(self):
        """
        Execute the genetic algorithm over the specified number of generations.

        Process:
            1. Initialize the population with random configurations.
            2. For each generation:
                - Generate a new population using `populate`.
                - Evaluate the fitness of all configurations.
                - Record statistics for the current generation.
                - Adjust the mutation rate dynamically based on fitness trends.
                - Check for stagnation and adjust parameters if necessary.
            3. At the end, select the top 5 configurations based on fitness and store their histories.

        Returns:
            tuple:
                - List of top 5 configurations with the highest fitness.
                - List of dictionaries containing detailed metrics for each top configuration.
        """
        self.initialize()
        for generation in range(1, self.generations):
            print(f"""Computing Generation {generation+1} started.""")
            self.populate()

            scores = []
            lifespans = []
            alive_growth_rates = []
            max_alive_cells_count = []
            stableness = []
            initial_living_cells_count = []

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
                initial_living_cells_count.append(self.configuration_cache[configuration]['initial_living_cells_count'])

            self.mutation_rate_history.append(self.mutation_rate)
            self.calc_statistics(generation=generation,
                                 scores=scores,
                                 lifespans=lifespans,
                                 alive_growth_rates=alive_growth_rates,
                                 max_alive_cells_count=max_alive_cells_count,
                                 stableness=stableness,
                                 initial_living_cells_count=initial_living_cells_count
                                 
                                 )
            self.check_for_stagnation(last_generation=generation)
            self.adjust_mutation_rate(generation)

        # Final selection of best configurations
        fitness_scores = [(config, self.configuration_cache[config]['fitness_score'])
                          for config in self.population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        best_configs = fitness_scores[:5]

        # Store their histories for later viewing
        for config, _ in best_configs:
            history = list(self.configuration_cache[config]['history'])
            self.best_histories.append(history)
            logging.info("Top Configuration:")
            logging.info(f"  Configuration: {config}")
            logging.info(f"""Fitness Score: {
                         self.configuration_cache[config]['fitness_score']}""")
            logging.info(f"""Lifespan: {
                         self.configuration_cache[config]['lifespan']}""")
            logging.info(f"""Total Alive Cells: {
                         self.configuration_cache[config]['max_alive_cells_count']}""")
            logging.info(f"""Alive Growth: {
                         self.configuration_cache[config]['alive_growth']}""")
            logging.info(f"""Initial Configuration Living Cells Count: {
                         self.configuration_cache[config]['initial_living_cells_count']}""")
        best_params = []
        for config, _ in best_configs:
            params_dict = {
                'lifespan': self.configuration_cache[config]['lifespan'],
                'max_alive_cells_count': self.configuration_cache[config]['max_alive_cells_count'],
                'alive_growth': self.configuration_cache[config]['alive_growth'],
                'stableness': self.configuration_cache[config]['stableness'],
                'initial_living_cells_count': self.configuration_cache[config]['initial_living_cells_count']
            }
            best_params.append(params_dict)
        self.best_params = best_params
        return best_configs, best_params

    def adjust_mutation_rate(self, generation):
        """
        Dynamically adjust the mutation rate based on changes in average fitness between generations.

        Purpose:
            - Increase mutation rate if no improvement in average fitness is observed, 
            to encourage exploration and escape local minima.
            - Reduce mutation rate gently if improvement is observed, promoting stability in the evolution.

        Process:
            - Compare the average fitness of the current generation with the previous generation.
            - If there is no improvement in average fitness for more than 10 generations, the mutation rate is increased.
            - If fitness improves, the mutation rate is decreased, but it is always kept above `mutation_rate_lower_limit`.

        Args:
            generation (int): The current generation index.

        Adjustments:
            - Increase mutation rate: Mutation rate is multiplied by 1.2, but capped at the mutation rate's lower limit.
            - Decrease mutation rate: Mutation rate is multiplied by 0.9, but not reduced below `mutation_rate_lower_limit`.
        """
        if generation > 10 and self.generations_cache[generation]['avg_fitness'] == self.generations_cache[generation - 1]['avg_fitness']:
            self.mutation_rate = min(
                self.mutation_rate_lower_limit, self.mutation_rate * 1.2)
        elif self.generations_cache[generation]['avg_fitness'] > self.generations_cache[generation - 1]['avg_fitness']:
            self.mutation_rate = max(
                self.mutation_rate_lower_limit, self.mutation_rate * 0.9)

    def check_for_stagnation(self, last_generation):
        """
        Detect stagnation in the evolution process over the last 10 generations.

        Purpose:
            - Identify if the population has stopped improving in terms of average fitness.
            - If stagnation is detected, increase the mutation rate significantly to encourage diversity and escape local optima.

        Process:
            - Retrieve the average fitness scores for the last 10 generations.
            - If all 10 generations have identical average fitness, classify it as stagnation.
            - Increase the mutation rate by 50% to encourage exploration.

        Args:
            last_generation (int): The index of the most recent generation.

        Adjustments:
            - Mutation rate: Increased by 50% if stagnation is detected but capped at the mutation rate's lower limit.

        Logs:
            - A warning is logged if stagnation is detected.
        """
        if last_generation >= 10:

            avg_fitness = [int(self.generations_cache[g]['avg_fitness'])
                           for g in range(last_generation -10, last_generation)]
            list_size = len(avg_fitness)
            set_size = len(set(avg_fitness))
            if set_size < list_size:
                logging.warning(
                    "Stagnation detected in the last 10 generations!")
                self.mutation_rate = min(1,min(self.mutation_rate_lower_limit, self.mutation_rate) * min((list_size/set_size), 1.5))
