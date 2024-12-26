"""
GeneticAlgorithm.py
-------------------

Implements a genetic algorithm to evolve initial configurations for Conway's Game of Life,
optimizing for extended lifespan, higher maximum living cells, greater growth ratio, etc.
"""

import logging
from GameOfLife import GameOfLife
import random
import math
import numpy as np
import collections


class GeneticAlgorithm:
    """
    The GeneticAlgorithm class handles population-based evolution of GameOfLife configurations.
    It maintains a population of binary NxN grids (flattened to 1D), calculates fitness for each,
    and performs selection, crossover, and mutation across multiple generations.

    Attributes:
        grid_size (int): Dimension N for NxN grids.
        population_size (int): Number of individuals in the population.
        generations (int): Number of generations to simulate.
        initial_mutation_rate (float): The starting probability of mutation per cell.
        mutation_rate_lower_limit (float): The lower bound for mutation rate adjustments.
        alive_cells_weight (float): Weight factor for max alive cells in the fitness function.
        lifespan_weight (float): Weight factor for lifespan in the fitness function.
        alive_growth_weight (float): Weight factor for growth ratio in the fitness function.
        stableness_weight (float): Weight factor for stableness in the fitness function.
        alive_cells_per_block (int): Used in random initialization (max living cells per block).
        alive_blocks (int): Number of blocks in which to place living cells in random init.
        predefined_configurations (optional): For injecting known patterns (not used by default).
        population (set[tuple]): Current set of configurations in the population.
        configuration_cache (dict): Cache mapping configurations -> their simulation results.
        generations_cache (dict): Statistics about each generation (avg fitness, etc.).
        best_histories (list): Holds the game-of-life state histories of top individuals at the end.
        mutation_rate_history (list): Tracks how the mutation rate changed across generations.
        lifespan_threshold (int): A threshold used for potential advanced logic (unused).
    """

    def __init__(self, grid_size, population_size, generations, initial_mutation_rate, mutation_rate_lower_limit,
                 alive_cells_weight, lifespan_weight, alive_growth_weight, stableness_weight,
                 alive_cells_per_block, alive_blocks, predefined_configurations=None):
        """
        Initialize the genetic algorithm.

        Args:
            grid_size (int): Dimension of the NxN grid.
            population_size (int): Size of the population per generation.
            generations (int): Number of generations to run.
            initial_mutation_rate (float): Starting mutation probability [0..1].
            mutation_rate_lower_limit (float): Minimum mutation rate (never goes below this).
            alive_cells_weight (float): Fitness weight for maximum alive cells encountered.
            lifespan_weight (float): Fitness weight for how many unique states appeared (lifespan).
            alive_growth_weight (float): Fitness weight for ratio between min and max alive cells.
            stableness_weight (float): Fitness weight for how stable or unstable the pattern becomes.
            alive_cells_per_block (int): Number of cells assigned as 'alive' in each randomly chosen block.
            alive_blocks (int): How many blocks are chosen to have 'alive' cells in random initialization.
            predefined_configurations (iterable, optional): Pre-made patterns (if desired).
        """
        print("Initializing GeneticAlgorithm.")
        self.grid_size = grid_size
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate_lower_limit = mutation_rate_lower_limit
        self.mutation_rate = initial_mutation_rate
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
        self.best_histories = []
        self.best_params = []
        self.population = set()
        self.mutation_rate_history = [initial_mutation_rate]

    def calc_fitness(self, lifespan, max_alive_cells_count, alive_growth, stableness,initial_living_cells_count):
        """
        Combine multiple factors (lifespan, alive cell count, growth, and stableness)
        into one final fitness score based on their respective weights.
        """
        lifespan_score = lifespan * self.lifespan_weight
        alive_cells_score = max_alive_cells_count * self.alive_cells_weight
        growth_score = alive_growth * self.alive_growth_weight
        stableness_score = stableness * self.stableness_weight
        return (lifespan_score + alive_cells_score + growth_score + stableness_score) * 1 / max(1,initial_living_cells_count)

    def evaluate(self, configuration):
        """
        Evaluate a single configuration by running GameOfLife on it and computing its fitness.
        Results are cached to avoid re-simulating the same configuration.

        Args:
            configuration (tuple[int]): A flattened NxN arrangement of 0s and 1s.

        Returns:
            dict: {
                'fitness_score': float,
                'history': tuple of states,
                'lifespan': int,
                'alive_growth': float,
                'max_alive_cells_count': int,
                'is_static': int,
                'is periodic': int,
                'stableness': float
            }
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
        initial_living_cells_count = sum(configuration)
        fitness_score = self.calc_fitness(
            lifespan=game.lifespan,
            max_alive_cells_count=game.max_alive_cells_count,
            alive_growth=game.alive_growth,
            stableness=game.stableness,
            initial_living_cells_count = initial_living_cells_count
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
            'initial_living_cells_count':initial_living_cells_count
        }
        return self.configuration_cache[configuration_tuple]

    def populate(self):
        """
        Produce a new generation of individuals by:
            1. Selecting parents from the current population.
            2. Performing crossover to create children.
            3. Applying mutation to some children.
            4. Combining old and new individuals, then picking the top performers.
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
        Mutation operator: flips some dead cells (0->1) randomly, influenced by mutation_rate.
        Currently does NOT kill living cells, but this can be changed if desired.

        Args:
            configuration (tuple[int]): A flattened NxN arrangement.

        Returns:
            tuple[int]: A new configuration with random mutations applied.
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
        Select two parents from the population using a custom probability distribution
        that depends on average fitness and standard deviation. If total fitness is 0,
        select randomly.

        Returns:
            (tuple[int], tuple[int]): Two parent configurations from the population.
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
        Creates a child from two parents by dividing each parent's chromosome into N blocks
        and selecting blocks according to their ratio of living cells.

        Args:
            parent1 (tuple[int]): 1D grid of length N*N.
            parent2 (tuple[int]): 1D grid of length N*N.

        Returns:
            tuple[int]: A new child configuration combining blocks from both.
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
        Create the initial random population of configurations, evaluate them all,
        and record basic statistics for generation 0.
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

        self.calc_statistics(generation=generation,
                             scores=scores,
                             lifespans=lifespans,
                             alive_growth_rates=alive_growth_rates,
                             max_alive_cells_count=max_alive_cells_count,
                             stableness=stableness)

        self.generations_cache[generation]['avg_fitness'] = np.average(scores)
        self.generations_cache[generation]['avg_lifespan'] = np.average(
            lifespans)
        self.generations_cache[generation]['avg_alive_growth_rate'] = np.average(
            alive_growth_rates)
        self.generations_cache[generation]['avg_max_alive_cells_count'] = np.average(
            max_alive_cells_count)
        self.generations_cache[generation]['std_fitness'] = np.std(scores)
        self.generations_cache[generation]['std_lifespan'] = np.std(lifespans)
        self.generations_cache[generation]['std_alive_growth_rate'] = np.std(
            alive_growth_rates)
        self.generations_cache[generation]['std_max_alive_cells_count'] = np.std(
            max_alive_cells_count)

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
        alive_blocks_indices = random.sample(range(self.alive_blocks), k=self.alive_blocks)
        total_taken_cells = self.alive_blocks * self.alive_cells_per_block
        logging.info(f"""alive_blocks_indices : {alive_blocks_indices}""")
        for part_index in alive_blocks_indices:
            start_idx = part_index*partition_size
            end_idx = start_idx + partition_size
            chosen_cells = random.sample(range(start_idx, end_idx), k = self.alive_cells_per_block)
            for cell in chosen_cells:
                configuration[cell] = 1
                total_taken_cells-=1
        return tuple(configuration)

    def calc_statistics(self, generation, scores, lifespans, alive_growth_rates, stableness, max_alive_cells_count):
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

        self.generations_cache[generation]['std_fitness'] = np.std(scores)
        self.generations_cache[generation]['std_lifespan'] = np.std(lifespans)
        self.generations_cache[generation]['std_alive_growth_rate'] = np.std(
            alive_growth_rates)
        self.generations_cache[generation]['std_max_alive_cells_count'] = np.std(
            max_alive_cells_count)

    def run(self):
        """
        Main loop of the GA:
        1. Initialize the population.
        2. For each generation up to self.generations:
            - Produce a new population via populate().
            - Evaluate them, record stats, manage mutation rate.
        3. Sort final population by fitness and select top 5, storing their histories.
        4. Return top 5 configurations.
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

            self.mutation_rate_history.append(self.mutation_rate)
            self.calc_statistics(generation=generation,
                                 scores=scores,
                                 lifespans=lifespans,
                                 alive_growth_rates=alive_growth_rates,
                                 max_alive_cells_count=max_alive_cells_count,
                                 stableness=stableness)
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
            logging
        best_params = []
        for config, _ in best_configs:
            params_dict = {
                'lifespan': self.configuration_cache[config]['lifespan'],
                'max_alive_cells_count': self.configuration_cache[config]['max_alive_cells_count'],
                'alive_growth': self.configuration_cache[config]['alive_growth'],
                'stableness': self.configuration_cache[config]['stableness'],
                'initial_living_cells_count' : self.configuration_cache[config]['initial_living_cells_count']
            }
            best_params.append(params_dict)
        self.best_params = best_params
        return best_configs, best_params

    def adjust_mutation_rate(self, generation):
        """
        Dynamically adjust the mutation rate if we detect improvement or stagnation.
        If there's no improvement in average fitness, attempt to increase the rate.
        Otherwise, if there's improvement, gently reduce it, but never below mutation_rate_lower_limit.
        """
        if generation > 10 and self.generations_cache[generation]['avg_fitness'] == self.generations_cache[generation - 1]['avg_fitness']:
            logging.info("Mutation rate increased due to stagnation.")
            self.mutation_rate = min(
                self.mutation_rate_lower_limit, self.mutation_rate * 1.2)
        elif self.generations_cache[generation]['avg_fitness'] > self.generations_cache[generation - 1]['avg_fitness']:
            self.mutation_rate = max(
                self.mutation_rate_lower_limit, self.mutation_rate * 0.9)

    def check_for_stagnation(self, last_generation):
        """
        Checks the last 10 generations for identical average fitness, indicating stagnation.
        If found, forcibly raises the mutation rate to try and escape local minima.

        Args:
            last_generation (int): Index of the current generation in the loop.
        """
        avg_fitness = [int(self.generations_cache[g]['avg_fitness'])
                       for g in range(last_generation)]
        if len(avg_fitness) >= 10 and len(set(avg_fitness[-10:])) == 1:
            logging.warning("Stagnation detected in the last 10 generations!")
            self.mutation_rate = min(
                self.mutation_rate_lower_limit, self.mutation_rate)*1.5
