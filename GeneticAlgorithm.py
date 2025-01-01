import logging
from GameOfLife import GameOfLife
import random
import math
import numpy as np
import collections
from itertools import combinations


class GeneticAlgorithm:
    """
    A genetic algorithm implementation for optimizing Conway's Game of Life configurations.
    Uses evolutionary principles to evolve configurations that maximize desired properties 
    like lifespan, cell growth, and pattern stability.

    Attributes:
        grid_size (int): Dimensions of the NxN grid for Game of Life configurations.
        population_size (int): Number of configurations in each generation.
        generations (int): Total number of generations to simulate.
        initial_mutation_rate (float): Starting mutation probability.
        mutation_rate_lower_limit (float): Minimum allowed mutation rate.
        alive_cells_weight (float): Fitness weight for maximum number of living cells.
        lifespan_weight (float): Fitness weight for configuration lifespan.
        alive_growth_weight (float): Fitness weight for cell growth ratio.
        initial_living_cells_count_penalty_weight (float): Penalty weight for large initial configurations.
        predefined_configurations (optional): Pre-made Game of Life patterns to include.
        population (set[tuple]): Current generation's configurations.
        configuration_cache (dict): Cache of evaluated configurations and their metrics.
        generations_cache (dict): Statistics for each generation.
        mutation_rate_history (list): Track mutation rate changes over time.
        diversity_history (list): Track diversity metrics over time.
    """

    def __init__(self, grid_size, population_size, generations, initial_mutation_rate, mutation_rate_lower_limit,
                 alive_cells_weight, lifespan_weight, alive_growth_weight,
                 initial_living_cells_count_penalty_weight, predefined_configurations=None):
        """
        Initialize the genetic algorithm with configuration parameters.

        Args:
            grid_size (int): Size of the NxN grid.
            population_size (int): Number of configurations per generation.
            generations (int): Number of generations to evolve.
            initial_mutation_rate (float): Starting mutation probability.
            mutation_rate_lower_limit (float): Minimum mutation rate.
            alive_cells_weight (float): Weight for maximum living cells in fitness.
            lifespan_weight (float): Weight for configuration lifespan in fitness.
            alive_growth_weight (float): Weight for cell growth ratio in fitness.
            initial_living_cells_count_penalty_weight (float): Weight for penalizing large initial patterns.
            predefined_configurations (optional): Predefined Game of Life patterns to include.
        """
        logging.info("""Initializing GeneticAlgorithm.""")
        self.grid_size = grid_size
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate_lower_limit = mutation_rate_lower_limit
        self.mutation_rate = initial_mutation_rate
        self.initial_mutation_rate = initial_mutation_rate
        self.alive_cells_weight = alive_cells_weight
        self.lifespan_weight = lifespan_weight
        self.initial_living_cells_count_penalty_weight = initial_living_cells_count_penalty_weight
        self.alive_growth_weight = alive_growth_weight
        self.configuration_cache = collections.defaultdict(dict)
        self.generations_cache = collections.defaultdict(dict)
        self.canonical_forms_cache = collections.defaultdict(
            tuple)  # Cache for canonical forms
        self.block_frequencies_cache = collections.defaultdict(
            dict)  # Cache for block frequencies
        self.population = set()
        self.initial_population = set()
        self.mutation_rate_history = []
        self.diversity_history = []  # Track diversity metrics over generations
        self.min_fitness = float('inf')  # Minimum fitness score
        self.max_fitness = float('-inf')  # Maximum fitness score
        self.min_uniqueness_score = float('inf')  # Minimum uniqueness score
        self.max_uniqueness_score = float('-inf')  # Maximum uniqueness score
        self.predefined_configurations = predefined_configurations

    def enrich_population_with_variety(self, clusters_type_amount, scatter_type_amount, basic_patterns_type_amount):
        """
        Generate diverse configurations using three distinct pattern types:

        1. Clustered: Groups of adjacent living cells
        - Creates naturalistic patterns
        - Variable cluster sizes based on grid dimensions

        2. Scattered: Randomly distributed living cells
        - Ensures broad pattern coverage
        - Controlled density distribution

        3. Basic patterns: Simple geometric arrangements
        - Creates structured initial patterns
        - Balanced random variations

        Args:
            clusters_type_amount (int): Number of cluster-based configurations
            scatter_type_amount (int): Number of scattered configurations
            basic_patterns_type_amount (int): Number of basic pattern configurations

        Returns:
            list[tuple[int]]: Collection of diverse initial configurations
        """
        total_cells = self.grid_size * self.grid_size
        max_cluster_size = self.grid_size ** 2
        min_cluster_size = self.grid_size
        max_scattered_cells = total_cells
        min_scattered_cells = self.grid_size
        max_pattern_cells = total_cells
        min_pattern_cells = self.grid_size
        population_pool = []

        # Generate Cluster Configurations
        for _ in range(clusters_type_amount):
            configuration = [0] * total_cells
            cluster_size = random.randint(min_cluster_size, max_cluster_size)
            center_row = random.randint(0, self.grid_size - 1)
            center_col = random.randint(0, self.grid_size - 1)
            added_cells = set()  # to ensure no duplicate cells
            for _ in range(cluster_size):
                while True:
                    offset_row = random.randint(-1, 1)
                    offset_col = random.randint(-1, 1)
                    row = (center_row + offset_row) % self.grid_size
                    col = (center_col + offset_col) % self.grid_size
                    index = row * self.grid_size + col
                    if index not in added_cells:
                        added_cells.add(index)
                        configuration[index] = 1
                        break
            population_pool.append(tuple(configuration))

        # Generate Scattered Configurations
        for _ in range(scatter_type_amount):
            configuration = [0] * total_cells
            scattered_cells = random.randint(min_scattered_cells, max_scattered_cells)
            scattered_indices = random.sample(range(total_cells), scattered_cells)
            for index in scattered_indices:
                configuration[index] = 1
            population_pool.append(tuple(configuration))

        # Generate Simple Patterns Configuration
        for _ in range(basic_patterns_type_amount):
            configuration = [0] * total_cells
            pattern_cells = random.randint(min_pattern_cells, max_pattern_cells)
            start_row = random.randint(0, self.grid_size - 3)
            start_col = random.randint(0, self.grid_size - 3)

            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if random.uniform(0, 1) < 0.5:
                        row = (start_row + i) % self.grid_size
                        col = (start_col + j) % self.grid_size
                        index = row * self.grid_size + col
                        configuration[index] = 1

            current_live_cells = sum(configuration)
            if current_live_cells < pattern_cells:
                additional_cells = random.sample([i for i in range(total_cells) if configuration[i] == 0],
                                                pattern_cells - current_live_cells)
                for index in additional_cells:
                    configuration[index] = 1

            population_pool.append(tuple(configuration))

        logging.debug("""Enriched population with variety.""")
        return population_pool


    def generate_new_population_pool(self, amount):
        clusters_type_amount = amount // 2 + amount % 2
        scatter_type_amount = amount // 4
        basic_patterns_type_amount = amount // 4

        population = self.enrich_population_with_variety(
            clusters_type_amount=clusters_type_amount,
            scatter_type_amount=scatter_type_amount,
            basic_patterns_type_amount=basic_patterns_type_amount
        )

        return population

    def calc_fitness(self, lifespan, max_alive_cells_count, alive_growth, stableness, initial_living_cells_count):
        """
        Calculate weighted fitness score combining multiple optimization objectives.

        Combines multiple metrics into a single fitness value that balances:
        - Configuration longevity through lifespan
        - Peak population through maximum alive cells
        - Growth dynamics through alive growth ratio
        - Efficiency through initial size penalty

        Args:
            lifespan (int): Number of unique states before stopping.
            max_alive_cells_count (int): Maximum living cells in any generation.
            alive_growth (float): Ratio of max to min living cells.
            stableness (int): 1 for configuration that stablizes and 0 for one who do not
            initial_living_cells_count (int): Starting number of living cells.

        Returns:
            float: Combined fitness score weighted by configuration parameters.
        """
        lifespan_score = lifespan * self.lifespan_weight
        alive_cells_score = max_alive_cells_count * self.alive_cells_weight
        growth_score = alive_growth * self.alive_growth_weight
        large_configuration_penalty = (
            1 / max(1, initial_living_cells_count * self.initial_living_cells_count_penalty_weight))
        fitness = ((lifespan_score + alive_cells_score + growth_score)
                   * (large_configuration_penalty) * stableness)
        return fitness

    def evaluate(self, configuration):
        """
        Evaluate a configuration by simulating its evolution and calculating fitness.

        Simulates the configuration through Conway's Game of Life and computes
        various metrics including lifespan, population dynamics, and stability.
        Results are cached to avoid redundant calculations.

        Args:
            configuration (tuple[int]): Flattened 1D representation of Game of Life grid.

        Returns:
            dict: Configuration evaluation results including:
                - fitness_score: Overall fitness value
                - history: Complete evolution history
                - lifespan: Number of unique states
                - alive_growth: Cell growth ratio
                - max_alive_cells_count: Peak population
                - is_static: Whether pattern becomes static
                - is_periodic: Whether pattern becomes periodic
                - stableness: Stability measure
                - initial_living_cells_count: Starting population
        """
        configuration_tuple = tuple(configuration)

        # Check if the configuration is already cached
        if configuration_tuple in self.configuration_cache:
            # Recalculate normalized fitness if min/max fitness changed
            if self.max_fitness != self.min_fitness:
                self.configuration_cache[configuration_tuple]['normalized_fitness_score'] = (
                    (self.configuration_cache[configuration_tuple]['fitness_score'] - self.min_fitness) /
                    (self.max_fitness - self.min_fitness)
                )
            else:
                self.configuration_cache[configuration_tuple]['normalized_fitness_score'] = 1.0
            logging.debug(
                """Configuration already evaluated. Retrieved from cache.""")
            return self.configuration_cache[configuration_tuple]

        expected_size = self.grid_size * self.grid_size
        if len(configuration_tuple) != expected_size:
            raise ValueError("""Configuration size must be {}, but got {}""".format(
                expected_size, len(configuration_tuple)))

        def max_difference_with_distance(lst):
            max_value = float('-inf')
            dis = 0
            min_index = 0
            for j in range(1, len(lst)):
                diff = (lst[j] - lst[min_index]) * (j - min_index)
                if diff > max_value:
                    dis = j - min_index
                    max_value = diff
                if lst[j] < lst[min_index]:
                    min_index = j
            return max(max_value, 0) / dis if dis != 0 else 0

        game = GameOfLife(self.grid_size, configuration_tuple)
        game.run()
        max_alive_cells_count = max(
            game.alive_history) if game.alive_history else 0
        initial_living_cells_count = sum(configuration_tuple)
        alive_growth = max_difference_with_distance(
            game.alive_history) if game.alive_history else 0
        stableness = 1 if game.is_methuselah else 0
        fitness_score = self.calc_fitness(
            lifespan=game.lifespan,
            max_alive_cells_count=max_alive_cells_count,
            alive_growth=alive_growth,
            stableness=stableness,
            initial_living_cells_count=initial_living_cells_count
        )

        # Update global min/max fitness values
        self.min_fitness = min(self.min_fitness, fitness_score)
        self.max_fitness = max(self.max_fitness, fitness_score)

        # Calculate normalized fitness for this configuration
        normalized_fitness = (
            (fitness_score - self.min_fitness) /
            (self.max_fitness - self.min_fitness)
            if self.max_fitness != self.min_fitness else 1.0
        )

        self.configuration_cache[configuration_tuple] = {
            'fitness_score': fitness_score,
            'normalized_fitness_score': normalized_fitness,
            'history': tuple(game.history[:]),
            'lifespan': game.lifespan,
            'alive_growth': alive_growth,
            'max_alive_cells_count': max_alive_cells_count,
            'is_static': game.is_static,
            'is_periodic': game.is_periodic,
            'stableness': stableness,
            'initial_living_cells_count': initial_living_cells_count
        }

        logging.debug("""Evaluated configuration and cached results.""")
        return self.configuration_cache[configuration_tuple]

    def initialize(self):
        """
        Initialize the population with diverse configurations.

        Creates initial population using three equal parts:
        - Clustered configurations
        - Scattered configurations
        - Basic pattern configurations

        Evaluates and caches fitness for all initial configurations.
        Records initial generation statistics.
        """
        logging.info(
            """Initializing population with diverse configurations.""")

        population_pool = self.generate_new_population_pool(
            amount=self.population_size)

        self.initial_population.update(population_pool)
        self.population.update(population_pool)
        self.compute_generation(generation=0)

    def populate(self, generation):
        """
        Generate the next generation of configurations using one of two strategies.

        Every 10th generation: Creates fresh configurations using three pattern types:
        - Clustered cells
        - Scattered cells  
        - Basic geometric patterns

        Other generations: Performs genetic operations:
        - Selects parents based on normalized fitness
        - Applies crossover
        - Mutates offspring
        - Preserves top performers

        Args:
            generation (int): Current generation number.
        """
        new_population = set()

        # **Elitism**: Preserve top 5% configurations unchanged
        elitism_count = max(1, int(0.05 * self.population_size))
        sorted_population = sorted(
            self.population,
            key=lambda config: self.configuration_cache[config]['fitness_score'],
            reverse=True
        )
        elites = sorted_population[:elitism_count]
        new_population.update(elites)
        logging.debug(f"""Elitism: Preserved top {
                      elitism_count} configurations.""")

        if generation % 10 != 0:
            amount = self.population_size // 4
            for _ in range(amount):
                try:
                    parent1, parent2 = self.select_parents(
                        generation=generation)
                    child = self.crossover(parent1, parent2)
                    if random.uniform(0, 1) < self.mutation_rate:
                        child = self.mutate(child)
                    new_population.add(child)
                except ValueError as ve:
                    logging.error(
                        """Error during selection and crossover: {}""".format(ve))
                    continue
        else:
            # Introduce fresh diversity
            logging.debug(
                """Introducing fresh diversity for generation {}.""".format(generation + 1))
            new_population = set(self.generate_new_population_pool(
                amount=self.population_size))

        # Combine new and existing population
        combined = list(new_population) + list(self.population)
        # Evaluate and retrieve normalized fitness scores
        combined = [(config, self.evaluate(config)['normalized_fitness_score'])
                    for config in combined]
        # Sort based on normalized fitness
        combined.sort(key=lambda x: x[1], reverse=True)

        # Update population with top performers
        self.population = set(
            [config for config, _ in combined[:self.population_size]]
        )

        # Fill remaining slots if any
        i = 0
        while len(self.population) < self.population_size and i < len(combined) - self.population_size:
            self.population.add(combined[self.population_size + i][0])
            i += 1

    def select_parents(self, generation):
        """
        main parent selection integrating all selection strategies.

        Args:
            generation (int): Current generation number.

        Returns:
            tuple: Two parent configurations for crossover.
        """
        if generation % 10 == 0 and generation != 0:
            # Every 10th generation, use corrected scores with penalties
            corrected_scores = self.calculate_corrected_scores()
        else:
            # Use normalized_fitness_score for selection
            corrected_scores = [(config, self.configuration_cache[config]
                                 ['normalized_fitness_score']) for config in self.population]

        parents = self.select_parents_method(corrected_scores)
        return parents

    def crossover(self, parent1, parent2):
        """
        Create child configurations using one of three crossover methods:

        1. Basic crossover (30% chance):
           - Alternates cells from parents
           - Simple but effective mixing strategy

        2. Simple crossover (30% chance):
           - Alternates rows from parents
           - Preserves horizontal patterns

        3. Complex crossover (40% chance):
           - Selects blocks based on living cell density
           - Intelligently combines high-fitness regions

        Args:
            parent1 (tuple[int]): First parent configuration
            parent2 (tuple[int]): Second parent configuration

        Returns:
            tuple[int]: Child configuration created through crossover
        """
        crossover_methods = [self.crossover_basic,
                             self.crossover_simple, self.crossover_complex]
        crossover_probabilities = [0.3, 0.3, 0.4]
        selected_crossover_method = random.choices(
            crossover_methods, weights=crossover_probabilities, k=1)[0]
        child_configuration = selected_crossover_method(parent1, parent2)
        logging.debug("""Crossover created child configuration: {}""".format(
            child_configuration))
        return child_configuration

    def mutate(self, configuration):
        """
        Apply one of three mutation strategies to a configuration:

        1. Basic mutation (40% chance):
           - Random cell flips with mutation rate probability
           - Uniform distribution of changes

        2. Cluster mutation (40% chance):
           - Flips cells in random 3x3 neighborhoods
           - Creates localized pattern changes

        3. Harsh mutation (20% chance):
           - Flips large contiguous blocks of cells
           - Enables major pattern alterations

        Args:
            configuration (tuple[int]): Configuration to mutate.

        Returns:
            tuple[int]: Mutated configuration.
        """
        mutation_methods = [self.mutate_basic,
                            self.mutate_clusters, self.mutate_harsh]
        mutation_probabilities = [0.4, 0.4, 0.2]
        selected_mutation_method = random.choices(
            mutation_methods, weights=mutation_probabilities, k=1)[0]
        mutated_configuration = selected_mutation_method(configuration)
        return mutated_configuration

    def crossover_basic(self, parent1, parent2):
        """
        Perform basic crossover by alternating cells from each parent.

        Args:
            parent1 (tuple[int]): First parent configuration.
            parent2 (tuple[int]): Second parent configuration.

        Returns:
            tuple[int]: Child configuration.
        """
        N = self.grid_size
        total_cells = N * N
        child = []
        for i in range(total_cells):
            child.append(parent1[i] if i % 2 == 0 else parent2[i])
        child_configuration = tuple(child)

        # Assert child size
        assert len(child_configuration) == total_cells, """Child size mismatch in basic crossover: expected {}, got {}""".format(
            total_cells, len(child_configuration))

        logging.debug("""Crossover applied (Basic): {}""".format(
            child_configuration))
        return child_configuration

    def crossover_simple(self, parent1, parent2):
        """
        Perform simple crossover by alternating entire rows from each parent.

        Args:
            parent1 (tuple[int]): First parent configuration.
            parent2 (tuple[int]): Second parent configuration.

        Returns:
            tuple[int]: Child configuration.

        Raises:
            ValueError: If parent configurations do not match the expected grid size.
        """
        N = self.grid_size
        total_cells = N * N

        if len(parent1) != total_cells or len(parent2) != total_cells:
            logging.error("""Parent configurations must be {}., but got sizes: {} and {}""".format(
                total_cells, len(parent1), len(parent2)))
            raise ValueError("""Parent configurations must be {}, but got sizes: {} and {}""".format(
                total_cells, len(parent1), len(parent2)))

        blocks_parent1 = [parent1[i * N: (i + 1) * N] for i in range(N)]
        blocks_parent2 = [parent2[i * N: (i + 1) * N] for i in range(N)]

        child_blocks = []
        for i in range(N):
            if i % 2 == 0:
                child_blocks.extend(blocks_parent2[i])
            else:
                child_blocks.extend(blocks_parent1[i])

        child_configuration = tuple(child_blocks)

        # Assert child size
        assert len(child_configuration) == total_cells, """Child size mismatch in simple crossover: expected {}, got {}""".format(
            total_cells, len(child_configuration))

        logging.debug("""Crossover applied (Simple): {}""".format(
            child_configuration))
        return child_configuration

    def crossover_complex(self, parent1, parent2):
        """
        Perform complex crossover by selecting blocks based on living cell density.

        Args:
            parent1 (tuple[int]): First parent configuration.
            parent2 (tuple[int]): Second parent configuration.

        Returns:
            tuple[int]: Child configuration.

        Raises:
            ValueError: If parent configurations do not match the expected grid size.
        """
        N = self.grid_size
        total_cells = N * N
        number_of_blocks = N  # Each block is a row
        block_size = N
        remainder = 0  # Since block_size * number_of_blocks == total_cells

        if len(parent1) != total_cells or len(parent2) != total_cells:
            logging.error("""Parent configurations must be {}., but got sizes: {} and {}""".format(
                total_cells, len(parent1), len(parent2)))
            raise ValueError("""Parent configurations must be {}, but got sizes: {} and {}""".format(
                total_cells, len(parent1), len(parent2)))

        blocks_parent1 = [
            parent1[i * block_size:(i + 1) * block_size] for i in range(number_of_blocks)]
        blocks_parent2 = [
            parent2[i * block_size:(i + 1) * block_size] for i in range(number_of_blocks)]

        block_alive_counts_parent1 = [sum(block) for block in blocks_parent1]
        block_alive_counts_parent2 = [sum(block) for block in blocks_parent2]
        max_alive_cells_parent1 = sum(block_alive_counts_parent1)
        max_alive_cells_parent2 = sum(block_alive_counts_parent2)

        if max_alive_cells_parent1 > 0:
            probabilities_parent1 = [(alive_count / max_alive_cells_parent1) if alive_count > 0 else (1 / number_of_blocks)
                                     for alive_count in block_alive_counts_parent1]
        else:
            probabilities_parent1 = [1 / number_of_blocks] * number_of_blocks

        if max_alive_cells_parent2 > 0:
            probabilities_parent2 = [(alive_count / max_alive_cells_parent2) if alive_count > 0 else (1 / number_of_blocks)
                                     for alive_count in block_alive_counts_parent2]
        else:
            probabilities_parent2 = [1 / number_of_blocks] * number_of_blocks

        # Ensure probabilities sum to 1
        sum_prob_parent1 = sum(probabilities_parent1)
        probabilities_parent1 = [
            p / sum_prob_parent1 for p in probabilities_parent1]

        sum_prob_parent2 = sum(probabilities_parent2)
        probabilities_parent2 = [
            p / sum_prob_parent2 for p in probabilities_parent2]

        selected_blocks_parent1 = random.choices(
            range(number_of_blocks), weights=probabilities_parent1, k=(number_of_blocks // 2) + remainder)
        remaining_blocks_parent2 = [i for i in range(
            number_of_blocks) if i not in selected_blocks_parent1]
        selected_blocks_parent2 = random.choices(
            remaining_blocks_parent2,
            weights=[probabilities_parent2[i]
                     for i in remaining_blocks_parent2],
            k=number_of_blocks // 2
        )

        child_blocks = []
        for i in range(number_of_blocks):
            if i in selected_blocks_parent1:
                child_blocks.extend(blocks_parent1[i])
            elif i in selected_blocks_parent2:
                child_blocks.extend(blocks_parent2[i])
            else:
                # This case should not occur, but handle it just in case
                selected_parent = random.choice([1, 2])
                if selected_parent == 1:
                    child_blocks.extend(blocks_parent1[i])
                else:
                    child_blocks.extend(blocks_parent2[i])

        child_configuration = tuple(child_blocks)

        # Assert child size
        assert len(child_configuration) == total_cells, """Child size mismatch in complex crossover: expected {}, got {}""".format(
            total_cells, len(child_configuration))

        logging.debug("""Crossover applied (Complex): {}""".format(
            child_configuration))
        return child_configuration

    def mutate_basic(self, configuration):
        """
        Perform basic mutation by flipping cells with a probability proportional to the mutation rate.

        Args:
            configuration (tuple[int]): Configuration to mutate.

        Returns:
            tuple[int]: Mutated configuration.
        """
        new_configuration = list(configuration)
        for i in range(len(configuration)):
            if random.uniform(0, 1) < min(0.5, self.mutation_rate * 5):
                new_configuration[i] = 0 if configuration[i] else 1
        mutated_configuration = tuple(new_configuration)

        # Assert child size
        expected_size = self.grid_size * self.grid_size
        assert len(mutated_configuration) == expected_size, """Mutated configuration size mismatch: expected {}, got {}""".format(
            expected_size, len(mutated_configuration))

        logging.debug("""Mutation applied (Basic): {}""".format(
            mutated_configuration))
        return mutated_configuration

    def mutate_harsh(self, configuration):
        """
        Perform harsh mutation by flipping large contiguous blocks of cells.

        Args:
            configuration (tuple[int]): Configuration to mutate.

        Returns:
            tuple[int]: Mutated configuration.
        """
        new_configuration = list(configuration)
        cluster_size = random.randint(1, len(new_configuration))
        start = random.randint(0, len(new_configuration) - 1)
        value = random.randint(0, 1)
        for j in range(cluster_size):
            idx = (start + j) % len(new_configuration)
            new_configuration[idx] = value
        mutated_configuration = tuple(new_configuration)

        # Assert child size
        expected_size = self.grid_size * self.grid_size
        assert len(mutated_configuration) == expected_size, """Mutated configuration size mismatch: expected {}, got {}""".format(
            expected_size, len(mutated_configuration))

        logging.debug("""Mutation applied (Harsh): {}""".format(
            mutated_configuration))
        return mutated_configuration

    def mutate_clusters(self, configuration, mutation_rate=0.1, cluster_size=3):
        """
        Perform cluster mutation by flipping cells in random neighborhoods.

        Args:
            configuration (tuple[int]): Configuration to mutate.
            mutation_rate (float): Probability of mutating a cluster.
            cluster_size (int): Size of the cluster to mutate.

        Returns:
            tuple[int]: Mutated configuration.
        """
        N = self.grid_size
        mutated = list(configuration)
        for _ in range(cluster_size):
            if random.uniform(0, 1) < mutation_rate:
                center_row = random.randint(0, N - 1)
                center_col = random.randint(0, N - 1)
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        row = (center_row + i) % N
                        col = (center_col + j) % N
                        index = row * N + col
                        mutated[index] = 1 if mutated[index] == 0 else 0
        mutated_configuration = tuple(mutated)

        # Assert child size
        expected_size = self.grid_size * self.grid_size
        assert len(mutated_configuration) == expected_size, """Mutated configuration size mismatch: expected {}, got {}""".format(
            expected_size, len(mutated_configuration))

        logging.debug("""Mutation applied (Cluster): {}""".format(
            mutated_configuration))
        return mutated_configuration

    def select_parents_normalized_probability(self, corrected_scores, num_parents=2):
        """
        Select parents using Normalized Probability (Roulette Wheel) Selection.

        Args:
            corrected_scores (list[tuple[int, float]]): List of tuples containing configurations and their corrected scores.
            num_parents (int): Number of parents to select.

        Returns:
            list[tuple[int]]: Selected parent configurations.
        """
        configs, scores = zip(*corrected_scores)
        total_score = sum(scores)
        if total_score == 0:
            logging.debug("""Total score is 0, selecting parents randomly.""")
            return random.choices(configs, k=num_parents)

        probabilities = [score / total_score for score in scores]
        selected_parents = random.choices(
            configs, weights=probabilities, k=num_parents)
        logging.debug(
            """Selected parents (Normalized Probability): {}""".format(selected_parents))
        return selected_parents

    def select_parents_tournament(self, corrected_scores, tournament_size=3, num_parents=2):
        """
        Select parents using Tournament Selection.

        Args:
            corrected_scores (list[tuple[int, float]]): List of tuples containing configurations and their corrected scores.
            tournament_size (int): Number of individuals competing in each tournament.
            num_parents (int): Number of parents to select.

        Returns:
            list[tuple[int]]: Selected parent configurations.
        """
        selected_parents = []
        population_size = len(corrected_scores)

        for _ in range(num_parents):
            # Randomly select tournament_size individuals
            tournament = random.sample(corrected_scores, k=min(
                tournament_size, population_size))
            parent = max(tournament, key=lambda x: x[1])[0]
            selected_parents.append(parent)
            logging.debug(
                """Selected parent from tournament: {}""".format(parent))

        logging.debug(
            """Selected parents (Tournament): {}""".format(selected_parents))
        return selected_parents

    def select_parents_rank_based(self, corrected_scores, num_parents=2):
        """
        Select parents using Rank-Based Selection.

        Args:
            corrected_scores (list[tuple[int, float]]): List of tuples containing configurations and their corrected scores.
            num_parents (int): Number of parents to select.

        Returns:
            list[tuple[int]]: Selected parent configurations.
        """
        sorted_scores = sorted(
            corrected_scores, key=lambda x: x[1], reverse=True)
        ranks = range(1, len(sorted_scores) + 1)
        total_rank = sum(ranks)
        probabilities = [rank / total_rank for rank in ranks]
        selected_parents = random.choices(
            [config for config, _ in sorted_scores],
            weights=probabilities,
            k=num_parents
        )
        logging.debug(
            """Selected parents (Rank-Based): {}""".format(selected_parents))
        return selected_parents

    def select_parents_method(self, corrected_scores):
        """
        Choose a selection method based on predefined probabilities and return selected parents.

        Args:
            corrected_scores (list[tuple[int, float]]): List of tuples containing configurations and their corrected scores.

        Returns:
            list[tuple[int]]: Selected parent configurations.
        """
        selection_methods = [
            self.select_parents_normalized_probability,
            self.select_parents_tournament,
            self.select_parents_rank_based
        ]
        selected_method = random.choices(
            selection_methods, weights=[0.5, 0.25, 0.25], k=1
        )[0]
        parents = selected_method(corrected_scores, num_parents=2)
        return parents

    def get_canonical_form(self, config):
        """
        Compute the canonical form of a configuration by normalizing its position and rotation.

        Args:
            config (tuple[int]): Flattened 1D representation of the grid.

        Returns:
            tuple[int]: Canonical form of the configuration.
        """
        if config in self.canonical_forms_cache:
            return self.canonical_forms_cache[config]

        grid = np.array(config).reshape(self.grid_size, self.grid_size)
        live_cells = np.argwhere(grid == 1)

        if live_cells.size == 0:
            canonical = tuple(grid.flatten())  # Return empty grid as-is
        else:
            min_row, min_col = live_cells.min(axis=0)
            translated_grid = np.roll(grid, shift=-min_row, axis=0)
            translated_grid = np.roll(
                translated_grid, shift=-min_col, axis=1)

            # Generate all rotations and find the lexicographically smallest
            rotations = [np.rot90(translated_grid, k).flatten()
                         for k in range(4)]
            canonical = tuple(min(rotations, key=lambda x: tuple(x)))

        self.canonical_forms_cache[config] = canonical
        return canonical

    def detect_recurrent_blocks(self, config):
        """
        Detect recurring canonical blocks within the configuration, considering rotations.

        Args:
            config (tuple[int]): Flattened 1D representation of the grid.

        Returns:
            dict: Frequency of each canonical block in the configuration.
        """
        if config in self.block_frequencies_cache:
            return self.block_frequencies_cache[config]

        block_size = self.grid_size // 2  # Define the block size
        grid = np.array(config).reshape(self.grid_size, self.grid_size)
        block_frequency = {}

        for row in range(0, self.grid_size, block_size):
            for col in range(0, self.grid_size, block_size):
                block = grid[row:row + block_size, col:col + block_size]
                if block.size == 0:
                    continue
                block_canonical = self.get_canonical_form(
                    tuple(block.flatten()))
                if block_canonical not in block_frequency:
                    block_frequency[block_canonical] = 0
                block_frequency[block_canonical] += 1

        self.block_frequencies_cache[config] = block_frequency
        return block_frequency

    def calculate_corrected_scores(self):
        """
        Calculate corrected scores by combining canonical form and cell frequency penalties.

        Returns:
            list[tuple[int, float]]: List of configurations with corrected scores.
        """
        total_cells = self.grid_size * self.grid_size
        frequency_vector = np.zeros(total_cells)
        canonical_frequency = {}
        block_frequencies = {}

        uniqueness_scores = []

        for config in self.population:
            frequency_vector += np.array(config)
            canonical = self.get_canonical_form(config)
            if canonical not in canonical_frequency:
                canonical_frequency[canonical] = 0
            canonical_frequency[canonical] += 1

            # Detect and update block frequencies
            block_frequency = self.detect_recurrent_blocks(config)
            for block, count in block_frequency.items():
                if block not in block_frequencies:
                    block_frequencies[block] = 0
                block_frequencies[block] += count

        corrected_scores = []

        for config in self.population:
            # Use normalized_fitness_score
            normalized_fitness = self.configuration_cache[config]['normalized_fitness_score']
            active_cells = [
                i for i, cell in enumerate(config) if cell == 1]

            # Canonical form penalty
            canonical = self.get_canonical_form(config)
            canonical_penalty = canonical_frequency.get(canonical, 1)

            # Cell frequency penalty
            if len(active_cells) == 0:
                cell_frequency_penalty = 1  # Avoid division by zero
            else:
                total_frequency = sum(
                    frequency_vector[i] for i in active_cells)
                cell_frequency_penalty = (
                    total_frequency / len(active_cells)) ** 3

            # Block recurrence penalty
            block_frequency_penalty = 1
            block_frequency = self.detect_recurrent_blocks(config)
            for block, count in block_frequency.items():
                block_frequency_penalty *= block_frequencies.get(block, 1)

            # Combine penalties
            uniqueness_score = (
                canonical_penalty * cell_frequency_penalty * block_frequency_penalty) ** 3
            uniqueness_scores.append(uniqueness_score)

        # Update min/max uniqueness scores globally
        self.min_uniqueness_score = min(
            uniqueness_scores) if uniqueness_scores else 0
        self.max_uniqueness_score = max(
            uniqueness_scores) if uniqueness_scores else 1

        for config, uniqueness_score in zip(self.population, uniqueness_scores):
            # Normalize uniqueness score
            normalized_uniqueness = (uniqueness_score - self.min_uniqueness_score) / \
                                    (self.max_uniqueness_score - self.min_uniqueness_score) \
                if self.max_uniqueness_score != self.min_uniqueness_score else 1.0

            # Use normalized_fitness in corrected_score
            corrected_score = (
                normalized_fitness if normalized_fitness is not None else 0) / max(1, normalized_uniqueness)
            corrected_scores.append((config, corrected_score))

        logging.debug("""Calculated corrected scores for parent selection.""")
        return corrected_scores

    def adjust_mutation_rate(self, generation):
        """
        Dynamically adjust mutation rate based on population fitness trends.

        Updates mutation rate to balance exploration and exploitation:
        - Increases rate when fitness plateaus to encourage exploration
        - Decreases rate when fitness improves to fine-tune solutions
        - Maintains rate within specified bounds

        Args:
            generation (int): Current generation number
        """
        if generation < 10:
            if generation == 0:
                return  # No previous generation to compare
            # Use basic improvement ratio
            improvement_ratio = self.generations_cache[generation]['avg_fitness'] / max(
                1, self.generations_cache[generation - 1]['avg_fitness'])
            self.mutation_rate = max(self.mutation_rate_lower_limit, min(
                self.initial_mutation_rate, improvement_ratio * self.mutation_rate))
            logging.debug("""Generation {}: Mutation rate adjusted to {} based on improvement ratio.""".format(
                generation + 1, self.mutation_rate))
        else:
            # Calculate improvement over the last 10 generations
            last_10_fitness = [
                self.generations_cache[g]['avg_fitness'] for g in range(generation - 10, generation)
            ]
            improvement_ratio = last_10_fitness[-1] / \
                max(1, last_10_fitness[0])

            if improvement_ratio < 1.01:
                # Plateau detected, increase mutation rate
                self.mutation_rate = min(
                    self.initial_mutation_rate, self.mutation_rate * 1.2)
                logging.debug("""Generation {}: Plateau detected. Increasing mutation rate to {}.""".format(
                    generation + 1, self.mutation_rate))
            else:
                # Fitness improving, decrease mutation rate
                self.mutation_rate = max(
                    self.mutation_rate_lower_limit, self.mutation_rate * 0.9)
                logging.debug("""Generation {}: Fitness improving. Decreasing mutation rate to {}.""".format(
                    generation + 1, self.mutation_rate))

    def check_for_stagnation(self, generation):
        """
        Monitor evolution progress and detect stagnation patterns.

        Analyzes recent generations to identify:
        - Complete stagnation (identical fitness scores)
        - Partial stagnation (low fitness diversity)
        - Adjusts mutation rate to escape local optima

        Args:
            generation (int): Current generation number
        """
        if generation < 10:
            return

        avg_fitness = [
            self.generations_cache[g]['avg_fitness']
            for g in range(generation - 10, generation)
        ]

        unique_fitness_scores = len(set(avg_fitness))
        total_generations = len(avg_fitness)
        stagnation_score = total_generations / \
            unique_fitness_scores if unique_fitness_scores > 0 else float(
                'inf')

        if stagnation_score > 5:
            self.mutation_rate = min(
                self.initial_mutation_rate, self.mutation_rate * 1.5)
            logging.debug("""Generation {}: High stagnation detected (score: {}). Increasing mutation rate to {}.""".format(
                generation + 1, stagnation_score, self.mutation_rate))
        elif stagnation_score > 2:
            self.mutation_rate = min(
                self.initial_mutation_rate, self.mutation_rate * 1.2)
            logging.debug("""Generation {}: Moderate stagnation detected (score: {}). Increasing mutation rate to {}.""".format(
                generation + 1, stagnation_score, self.mutation_rate))

    def compute_generation(self, generation):
        """
        Evaluate current generation and record population statistics.

        Processes current generation by:
        - Calculating fitness metrics for all configurations
        - Updating generation cache with statistics
        - Tracking mutation rate history
        - Tracking diversity metrics

        Args:
            generation (int): Current generation number.
        """
        print(
            """Computing Generation {} started.""".format(generation + 1))
        scores = []
        lifespans = []
        alive_growth_rates = []
        max_alive_cells_count = []
        stableness = []
        initial_living_cells_count = []

        for configuration in self.population:
            evaluation = self.evaluate(configuration)
            scores.append(evaluation['fitness_score'])
            lifespans.append(evaluation['lifespan'])
            alive_growth_rates.append(evaluation['alive_growth'])
            max_alive_cells_count.append(evaluation['max_alive_cells_count'])
            stableness.append(evaluation['stableness'])
            initial_living_cells_count.append(
                evaluation['initial_living_cells_count'])

        self.mutation_rate_history.append(self.mutation_rate)
        self.calc_statistics(
            generation=generation,
            scores=scores,
            lifespans=lifespans,
            alive_growth_rates=alive_growth_rates,
            max_alive_cells_count=max_alive_cells_count,
            stableness=stableness,
            initial_living_cells_count=initial_living_cells_count
        )
        # Track diversity after statistics calculation
        self.track_diversity()

    def calc_statistics(self, generation, scores, lifespans, alive_growth_rates, max_alive_cells_count, stableness, initial_living_cells_count):
        """
        Calculate and store population statistics for the current generation.

        Computes and records:
        - Mean and standard deviation for all fitness metrics
        - Population dynamics measures
        - Configuration characteristics

        Args:
            generation (int): Current generation number
            scores (list[float]): Fitness scores for all configurations
            lifespans (list[int]): Lifespan values for all configurations
            alive_growth_rates (list[float]): Growth rates for all configurations
            stableness (list[float]): Stability measures for all configurations
            max_alive_cells_count (list[int]): Peak populations for all configurations
            initial_living_cells_count (list[int]): Initial sizes for all configurations
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
        self.generations_cache[generation]['avg_initial_living_cells_count'] = np.mean(
            initial_living_cells_count)

        self.generations_cache[generation]['std_fitness'] = np.std(scores)
        self.generations_cache[generation]['std_lifespan'] = np.std(lifespans)
        self.generations_cache[generation]['std_alive_growth_rate'] = np.std(
            alive_growth_rates)
        self.generations_cache[generation]['std_max_alive_cells_count'] = np.std(
            max_alive_cells_count)
        self.generations_cache[generation]['std_initial_living_cells_count'] = np.std(
            initial_living_cells_count)

        logging.debug(
            """Calculated statistics for generation {}.""".format(generation + 1))

    def track_diversity(self):
        """
        Track genetic diversity within the population.

        Calculates diversity metrics such as average Hamming distance between configurations.
        """
        if not self.population:
            self.diversity_history.append(0)
            return

        population_list = list(self.population)
        total_pairs = len(population_list) * (len(population_list) - 1) / 2
        if total_pairs == 0:
            average_hamming_distance = 0
        else:
            hamming_distances = [
                self.hamming_distance(population_list[i], population_list[j])
                for i in range(len(population_list))
                for j in range(i + 1, len(population_list))
            ]
            average_hamming_distance = sum(hamming_distances) / total_pairs

        self.diversity_history.append(average_hamming_distance)
        logging.debug("""Tracked diversity: Average Hamming Distance = {}""".format(
            average_hamming_distance))

    @staticmethod
    def hamming_distance(config1, config2):
        """
        Calculate the Hamming distance between two configurations.

        Args:
            config1 (tuple[int]): First configuration.
            config2 (tuple[int]): Second configuration.

        Returns:
            int: Hamming distance.
        """
        return sum(c1 != c2 for c1, c2 in zip(config1, config2))

    def get_diversity_trend(self):
        """
        Retrieve the history of diversity metrics.

        Returns:
            list[float]: List of average Hamming distances per generation.
        """
        return self.diversity_history

    def run(self):
        """
        Execute the complete evolutionary process.

        Performs the following steps:
        1. Initializes random population
        2. Iterates through specified generations:
           - Generates new configurations
           - Evaluates fitness
           - Updates statistics
           - Adjusts parameters
           - Checks for stagnation
        3. Returns results of evolution

        Returns:
            tuple: (results list, initial_configurations_start_index)
        """
        self.initialize()
        for generation in range(1, self.generations):
            self.populate(generation=generation)
            self.compute_generation(generation=generation)
            self.adjust_mutation_rate(generation)
            self.check_for_stagnation(generation)

        return self.get_experiment_results()

    def get_experiment_results(self):
        """
        Compile and return the results of the GA experiment.

        Returns:
            tuple: (results list, initial_configurations_start_index)
        """
        # Final selection of best configurations
        fitness_scores = [(config, self.configuration_cache[config]['fitness_score'])
                          for config in self.population]

        fitness_scores_initial_population = [(config, self.configuration_cache[config]['fitness_score'])
                                             for config in self.initial_population]

        logging.debug("""Initial population size: {}""".format(
            len(fitness_scores_initial_population)))
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        fitness_scores_initial_population.sort(
            key=lambda x: x[1], reverse=True)

        top_configs = fitness_scores[:self.population_size]

        results = []

        for config, _ in top_configs:
            params_dict = {
                'fitness_score': self.configuration_cache[config]['fitness_score'],
                'normalized_fitness_score': self.configuration_cache[config]['normalized_fitness_score'],
                'lifespan': self.configuration_cache[config]['lifespan'],
                'max_alive_cells_count': self.configuration_cache[config]['max_alive_cells_count'],
                'alive_growth': self.configuration_cache[config]['alive_growth'],
                'stableness': self.configuration_cache[config]['stableness'],
                'initial_living_cells_count': self.configuration_cache[config]['initial_living_cells_count'],
                'history': list(self.configuration_cache[config]['history']),
                'config': config,
                'is_first_generation': False
            }

            logging.info("""Top Configuration:
Configuration: {}
Fitness Score: {}
Normalized Fitness Score: {}
Lifespan: {}
Total Alive Cells: {}
Alive Growth: {}
Initial Configuration Living Cells Count: {}""".format(
                config,
                self.configuration_cache[config]['fitness_score'],
                self.configuration_cache[config]['normalized_fitness_score'],
                self.configuration_cache[config]['lifespan'],
                self.configuration_cache[config]['max_alive_cells_count'],
                self.configuration_cache[config]['alive_growth'],
                self.configuration_cache[config]['initial_living_cells_count']
            ))
            results.append(params_dict)

        for config, _ in fitness_scores_initial_population:
            params_dict = {
                'fitness_score': self.configuration_cache[config]['fitness_score'],
                'normalized_fitness_score': self.configuration_cache[config]['normalized_fitness_score'],
                'lifespan': self.configuration_cache[config]['lifespan'],
                'max_alive_cells_count': self.configuration_cache[config]['max_alive_cells_count'],
                'alive_growth': self.configuration_cache[config]['alive_growth'],
                'stableness': self.configuration_cache[config]['stableness'],
                'initial_living_cells_count': self.configuration_cache[config]['initial_living_cells_count'],
                'history': list(self.configuration_cache[config]['history']),
                'config': config,
                'is_first_generation': True
            }

            logging.info("""Initial Configuration:
Configuration: {}
Fitness Score: {}
Normalized Fitness Score: {}
Lifespan: {}
Total Alive Cells: {}
Alive Growth: {}
Initial Configuration Living Cells Count: {}""".format(
                config,
                self.configuration_cache[config]['fitness_score'],
                self.configuration_cache[config]['normalized_fitness_score'],
                self.configuration_cache[config]['lifespan'],
                self.configuration_cache[config]['max_alive_cells_count'],
                self.configuration_cache[config]['alive_growth'],
                self.configuration_cache[config]['initial_living_cells_count']
            ))

            results.append(params_dict)

        initial_configurations_start_index = len(top_configs)
        return results, initial_configurations_start_index
