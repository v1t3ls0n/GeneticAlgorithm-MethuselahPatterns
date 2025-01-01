import logging
from GameOfLife import GameOfLife
import random
import math
import numpy as np
import collections


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
        stableness_weight (float): Fitness weight for pattern stability.
        initial_living_cells_count_penalty_weight (float): Penalty weight for large initial configurations.
        predefined_configurations (optional): Pre-made Game of Life patterns to include.
        population (set[tuple]): Current generation's configurations.
        configuration_cache (dict): Cache of evaluated configurations and their metrics.
        generations_cache (dict): Statistics for each generation.
        mutation_rate_history (list): Track mutation rate changes over time.
    """

    def __init__(self, grid_size, population_size, generations, initial_mutation_rate, mutation_rate_lower_limit,
                 alive_cells_weight, lifespan_weight, alive_growth_weight, stableness_weight,
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
            stableness_weight (float): Weight for pattern stability in fitness.
            initial_living_cells_count_penalty_weight (float): Weight for penalizing large initial patterns.
            predefined_configurations (optional): Predefined Game of Life patterns to include.
        """
        print("Initializing GeneticAlgorithm.")
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
        self.stableness_weight = stableness_weight
        self.configuration_cache = collections.defaultdict(
            collections.defaultdict)
        self.generations_cache = collections.defaultdict(
            collections.defaultdict)
        self.canonical_forms_cache = collections.defaultdict(
            tuple)  # Cache for canonical forms
        self.block_frequencies_cache = collections.defaultdict(
            tuple)  # Cache for block frequencies
        self.population = set()
        self.initial_population = []
        self.mutation_rate_history = []
        self.min_fitness = float('inf')  # Minimum fitness score
        self.max_fitness = float('-inf')  # Maximum fitness score
        self.min_uniqueness_score = float('inf')  # Minimum uniqueness score
        self.max_uniqueness_score = float('-inf')  # Maximum uniqueness score
        self.predefined_configurations = predefined_configurations

    def calc_fitness(self, lifespan, max_alive_cells_count, alive_growth, stableness, initial_living_cells_count):
        """
        Calculate weighted fitness score combining multiple optimization objectives.

        Combines multiple metrics into a single fitness value that balances:
        - Configuration longevity through lifespan
        - Peak population through maximum alive cells
        - Growth dynamics through alive growth ratio
        - Pattern stability through stableness score
        - Efficiency through initial size penalty

        Args:
            lifespan (int): Number of unique states before stopping.
            max_alive_cells_count (int): Maximum living cells in any generation.
            alive_growth (float): Ratio of max to min living cells.
            stableness (float): Measure of pattern stability.
            initial_living_cells_count (int): Starting number of living cells.

        Returns:
            float: Combined fitness score weighted by configuration parameters.
        """
        lifespan_score = lifespan * self.lifespan_weight
        alive_cells_score = max_alive_cells_count * self.alive_cells_weight
        growth_score = alive_growth * self.alive_growth_weight
        stableness_score = stableness * self.stableness_weight
        large_configuration_penalty = (
            1 / max(1, initial_living_cells_count * self.initial_living_cells_count_penalty_weight))
        return ((lifespan_score + alive_cells_score + growth_score + stableness_score) * (large_configuration_penalty))

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
            self.configuration_cache[configuration_tuple]['normalized_fitness_score'] = (
                (self.configuration_cache[configuration_tuple]['fitness_score'] - self.min_fitness) /
                (self.max_fitness - self.min_fitness)
                if self.max_fitness != self.min_fitness else 1.0
            )
            return self.configuration_cache[configuration_tuple]

        expected_size = self.grid_size * self.grid_size
        if len(configuration_tuple) != expected_size:
            raise ValueError(f"""Configuration size must be {
                             expected_size}, but got {len(configuration_tuple)}""")

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
            return max(max_value, 0) / dis

        game = GameOfLife(self.grid_size, configuration_tuple)
        game.run()
        max_alive_cells_count = max(game.alive_history)
        initial_living_cells_count = sum(configuration_tuple)
        alive_growth = max_difference_with_distance(game.alive_history)
        stableness = game.stable_count / game.max_stable_generations
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

        return self.configuration_cache[configuration_tuple]

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
        if generation % 10:
            amount = self.population_size // 4
            for _ in range(amount):
                parent1, parent2 = self.select_parents(generation=generation)
                child = self.crossover(parent1, parent2)
                if random.uniform(0, 1) < self.mutation_rate:
                    child = self.mutate(child)
                new_population.add(child)
        else:
            uniform_amount = self.population_size // 3
            rem_amount = self.population_size % 3
            new_population = set(self.enrich_population_with_variety(
                clusters_type_amount=uniform_amount+rem_amount,
                scatter_type_amount=uniform_amount,
                basic_patterns_type_amount=uniform_amount
            ))

        # **Use normalized_fitness_score instead of fitness_score for sorting**
        combined = list(new_population) + list(self.population)
        combined = [(config, self.evaluate(config)['normalized_fitness_score'])
                    for config in combined]
        combined.sort(key=lambda x: x[1], reverse=True)

        self.population = set(
            [config for config, _ in combined[:self.population_size]]
        )

        i = 0
        while len(self.population) < self.population_size and i < len(combined) - self.population_size:
            self.population.add(combined[self.population_size + i][0])
            i += 1

    def select_parents(self, generation):
        """
        Select two parent configurations using one of three selection methods.

        Combines canonical forms and cell frequency analysis to adjust fitness scores dynamically.
        Penalizes configurations that are both frequent in canonical form and have highly common active cells.

        Additionally considers recurring canonical blocks across configurations with rotational equivalence.

        Methods:
            1. Normalized probability (50% chance): Fitness proportional to adjusted score.
            2. Tournament selection (25% chance): Random subsets competition.
            3. Rank-based selection (25% chance): Based on fitness ranking.

        Returns:
            tuple: Two parent configurations for crossover.
        """

        def canonical_form(config, grid_size):
            """
            Compute the canonical form of a configuration by normalizing its position and rotation.

            Args:
                config (list[int]): Flattened 1D representation of the grid.
                grid_size (int): Size of the grid (NxN).

            Returns:
                tuple[int]: Canonical form of the configuration.
            """
            if config in self.canonical_forms_cache:
                return self.canonical_forms_cache[config]

            grid = np.array(config).reshape(grid_size, grid_size)
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

        def detect_recurrent_blocks(config, grid_size):
            """
            Detect recurring canonical blocks within the configuration, considering rotations.

            Args:
                config (list[int]): Flattened 1D representation of the grid.
                grid_size (int): Size of the grid (NxN).

            Returns:
                dict: Frequency of each canonical block in the configuration.
            """
            if config in self.block_frequencies_cache:
                return self.block_frequencies_cache[config]

            block_size = grid_size // 2  # Define the block size
            grid = np.array(config).reshape(grid_size, grid_size)
            block_frequency = {}

            for row in range(0, grid_size, block_size):
                for col in range(0, grid_size, block_size):
                    block = grid[row:row + block_size, col:col + block_size]
                    block_canonical = canonical_form(
                        block.flatten(), block_size)
                    if block_canonical not in block_frequency:
                        block_frequency[block_canonical] = 0
                    block_frequency[block_canonical] += 1

            self.block_frequencies_cache[config] = block_frequency
            return block_frequency

        def calculate_corrected_scores():
            """
            Calculate corrected scores by combining canonical form and cell frequency penalties.
            """
            total_cells = self.grid_size * self.grid_size
            frequency_vector = np.zeros(total_cells)
            canonical_frequency = {}
            block_frequencies = {}

            uniqueness_scores = []

            for config in self.population:
                frequency_vector += np.array(config)
                canonical = canonical_form(config, self.grid_size)
                if canonical not in canonical_frequency:
                    canonical_frequency[canonical] = 0
                canonical_frequency[canonical] += 1

                # Detect and update block frequencies
                block_frequency = detect_recurrent_blocks(
                    config, self.grid_size)
                for block, count in block_frequency.items():
                    if block not in block_frequencies:
                        block_frequencies[block] = 0
                    block_frequencies[block] += count

            corrected_scores = []

            for config in self.population:
                # **Use normalized_fitness_score instead of fitness_score**
                normalized_fitness = self.configuration_cache[config]['normalized_fitness_score']
                active_cells = [
                    i for i, cell in enumerate(config) if cell == 1]

                # Canonical form penalty
                canonical = canonical_form(config, self.grid_size)
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
                block_frequency = detect_recurrent_blocks(
                    config, self.grid_size)
                for block, count in block_frequency.items():
                    block_frequency_penalty *= block_frequencies.get(block, 1)

                # Combine penalties
                uniqueness_score = (
                    canonical_penalty * cell_frequency_penalty * block_frequency_penalty) ** 2
                uniqueness_scores.append(uniqueness_score)

            # Update min/max uniqueness scores globally
            self.min_uniqueness_score = min(uniqueness_scores)
            self.max_uniqueness_score = max(uniqueness_scores)

            for config, uniqueness_score in zip(self.population, uniqueness_scores):
                # Normalize uniqueness score
                normalized_uniqueness = (uniqueness_score - self.min_uniqueness_score) / \
                                        (self.max_uniqueness_score - self.min_uniqueness_score) \
                    if self.max_uniqueness_score != self.min_uniqueness_score else 1.0

                # **Use normalized_fitness in corrected_score**
                corrected_score = (
                    normalized_fitness if normalized_fitness is not None else 0) / max(1, normalized_uniqueness)
                corrected_scores.append((config, corrected_score))

            return corrected_scores

        # Use corrected scores every 10th generation
        if generation % 10 == 0:
            corrected_scores = calculate_corrected_scores()
        else:
            # **Use normalized_fitness_score instead of fitness_score**
            corrected_scores = [(config, self.configuration_cache[config]
                                 ['normalized_fitness_score']) for config in self.population]

        def normalized_probability_selection():
            configs, scores = zip(*corrected_scores)
            total_score = sum(scores)
            if total_score == 0:
                logging.info(
                    f"""Total score is 0, selecting random parents.""")
                return random.choices(configs, k=2)

            probabilities = [score / total_score for score in scores]
            return random.choices(configs, weights=probabilities, k=2)

        def tournament_selection():
            tournament_size = min(3, self.population_size // 4)
            candidates1 = random.sample(corrected_scores, k=tournament_size)
            candidates2 = random.sample(corrected_scores, k=tournament_size)
            parent1 = max(candidates1, key=lambda x: x[1])[0]
            parent2 = max(candidates2, key=lambda x: x[1])[0]
            return parent1, parent2

        def rank_based_selection():
            sorted_scores = sorted(
                corrected_scores, key=lambda x: x[1], reverse=True)
            configs, scores = zip(*sorted_scores)
            ranks = range(1, len(configs) + 1)
            total_rank = sum(ranks)
            probabilities = [rank / total_rank for rank in ranks]
            return random.choices(configs, weights=probabilities, k=2)

        selection_methods = [normalized_probability_selection,
                             tournament_selection, rank_based_selection]
        selected_method = random.choices(selection_methods, weights=[
                                         0.5, 0.25, 0.25], k=1)[0]

        return selected_method()

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
        def mutate_basic(configuration):
            new_configuration = list(configuration)
            for i in range(len(configuration)):
                if random.uniform(0, 1) < min(0.5, self.mutation_rate * 5):
                    new_configuration[i] = 0 if configuration[i] else 1
            return tuple(new_configuration)

        def mutate_harsh(configuration):
            new_configuration = list(configuration)
            cluster_size = random.randint(1, len(new_configuration))
            start = random.randint(0, len(new_configuration) - 1)
            value = random.randint(0, 1)
            for j in range(cluster_size):
                idx = (start + j) % len(new_configuration)
                new_configuration[idx] = value
            return tuple(new_configuration)

        def mutate_clusters(configuration, mutation_rate=0.1, cluster_size=3):
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
            return tuple(mutated)

        mutation_methods = [mutate_basic, mutate_clusters, mutate_harsh]
        mutate_func = random.choices(mutation_methods, [0.4, 0.4, 0.2], k=1)[0]
        return mutate_func(configuration)

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
        def crossover_basic(parent1, parent2):
            N = self.grid_size
            total_cells = N * N
            child = []
            for i in range(total_cells):
                if i % 2 == 0:
                    child.append(parent1[i])
                else:
                    child.append(parent2[i])
            return tuple(child)

        def crossover_simple(parent1, parent2):
            N = self.grid_size
            total_cells = N * N

            if len(parent1) != total_cells or len(parent2) != total_cells:
                logging.error(f"""Parent configurations must be {total_cells}, but got sizes: {
                              len(parent1)} and {len(parent2)}""")
                raise ValueError(f"""Parent configurations must be {
                                 total_cells}, but got sizes: {len(parent1)} and {len(parent2)}""")

            blocks_parent1 = [parent1[i * N: (i + 1) * N] for i in range(N)]
            blocks_parent2 = [parent2[i * N: (i + 1) * N] for i in range(N)]

            child_blocks = []
            for i in range(N):
                if i % 2 == 0:
                    child_blocks.extend(blocks_parent2[i])
                else:
                    child_blocks.extend(blocks_parent1[i])

            if len(child_blocks) != total_cells:
                logging.debug(f"""Child size mismatch, expected {
                              total_cells}, got {len(child_blocks)}""")
                child_blocks = child_blocks + \
                    [0] * (total_cells - len(child_blocks))

            return tuple(child_blocks)

        def crossover_complex(parent1, parent2):
            N = self.grid_size
            total_cells = N*N
            reminder = N % 2

            if len(parent1) != total_cells or len(parent2) != total_cells:
                logging.info(f"""Parent configurations must be {total_cells}, but got sizes: {
                             len(parent1)} and {len(parent2)}""")
                raise ValueError(f"""Parent configurations must be {
                                 total_cells}, but got sizes: {len(parent1)} and {len(parent2)}""")

            block_size = N
            blocks_parent1 = [
                parent1[i*block_size:(i+1)*block_size] for i in range(N)]
            blocks_parent2 = [
                parent2[i*block_size:(i+1)*block_size] for i in range(N)]

            block_alive_counts_parent1 = [
                sum(block) for block in blocks_parent1]
            block_alive_counts_parent2 = [
                sum(block) for block in blocks_parent2]
            max_alive_cells_parent1 = sum(block_alive_counts_parent1)
            max_alive_cells_parent2 = sum(block_alive_counts_parent2)

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
                    selected_parent = random.choices(
                        [1, 2], weights=[0.5, 0.5], k=1)[0]
                    if selected_parent == 1:
                        child_blocks.extend(blocks_parent1[i])
                    else:
                        child_blocks.extend(blocks_parent2[i])

            if len(child_blocks) != total_cells:
                logging.info(f"""Child size mismatch, expected {
                             total_cells}, got {len(child_blocks)}""")
                child_blocks = child_blocks + \
                    [0]*(total_cells - len(child_blocks))
            return tuple(child_blocks)

        crossover_methods = [crossover_basic,
                             crossover_simple, crossover_complex]
        selected_crossover_method = random.choices(
            crossover_methods, [0.3, 0.3, 0.4], k=1)[0]
        return selected_crossover_method(parent1, parent2)

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
        max_cluster_size = total_cells // 4
        min_cluster_size = min(3, self.grid_size)
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
            for _ in range(cluster_size):
                offset_row = random.randint(-1, 1)
                offset_col = random.randint(-1, 1)
                row = (center_row + offset_row) % self.grid_size
                col = (center_col + offset_col) % self.grid_size
                index = row * self.grid_size + col
                configuration[index] = 1
            population_pool.append(tuple(configuration))

        # Generate Scattered Configurations
        for _ in range(scatter_type_amount):
            configuration = [0] * total_cells
            scattered_cells = random.randint(
                min_scattered_cells, max_scattered_cells)
            scattered_indices = random.sample(
                range(total_cells), scattered_cells)
            for index in scattered_indices:
                configuration[index] = 1
            population_pool.append(tuple(configuration))

        # Generate Simple Patterns Configuration
        for _ in range(basic_patterns_type_amount):
            configuration = [0] * total_cells
            pattern_cells = random.randint(
                min_pattern_cells, max_pattern_cells)
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

        return population_pool

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
        uniform_amount = self.population_size // 3
        rem_amount = self.population_size % 3
        population = self.enrich_population_with_variety(
            clusters_type_amount=uniform_amount+rem_amount,
            scatter_type_amount=uniform_amount,
            basic_patterns_type_amount=uniform_amount
        )

        self.initial_population = set(population[:])
        self.population = set(population)
        self.compute_generation(generation=0)

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
            # Not enough history for 10 generations, use basic improvement ratio
            improvement_ratio = self.generations_cache[generation-1]['avg_fitness'] / max(
                1, self.generations_cache[generation]['avg_fitness'])
            self.mutation_rate = max(self.mutation_rate_lower_limit, min(
                self.initial_mutation_rate, improvement_ratio * self.mutation_rate))
        else:
            # Calculate improvement over the last 10 generations
            avg_fitness_last_10 = [
                self.generations_cache[g]['avg_fitness'] for g in range(generation-10, generation)
            ]
            improvement_ratio = avg_fitness_last_10[-1] / \
                max(1, avg_fitness_last_10[0])

            if improvement_ratio < 1.01:
                # Plateau detected, increase mutation rate
                self.mutation_rate = min(
                    self.initial_mutation_rate, self.mutation_rate * 1.2)
            else:
                # Fitness improving, decrease mutation rate
                self.mutation_rate = max(
                    self.mutation_rate_lower_limit, self.mutation_rate * 0.9)

    def check_for_stagnation(self, last_generation):
        """
        Monitor evolution progress and detect stagnation patterns.

        Analyzes recent generations to identify:
        - Complete stagnation (identical fitness scores)
        - Partial stagnation (low fitness diversity)
        - Adjusts mutation rate to escape local optima

        Args:
            last_generation (int): Current generation number
        """
        if last_generation < 10:
            return

        avg_fitness = [
            self.generations_cache[g]['avg_fitness']
            for g in range(last_generation - 10, last_generation)
        ]

        unique_fitness_scores = len(set(avg_fitness))
        total_generations = len(avg_fitness)
        stagnation_score = total_generations/unique_fitness_scores

        if stagnation_score > 5:
            self.mutation_rate = min(
                self.initial_mutation_rate, self.mutation_rate * 1.5)

        elif stagnation_score > 2:
            self.mutation_rate = min(
                self.initial_mutation_rate, self.mutation_rate * 1.2)

    def compute_generation(self, generation):
        """
        Evaluate current generation and record population statistics.

        Processes current generation by:
        - Calculating fitness metrics for all configurations
        - Updating generation cache with statistics
        - Tracking mutation rate history

        Args:
            generation (int): Current generation number
        """
        print(f"Computing Generation {generation+1} started.")
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
            initial_living_cells_count.append(
                self.configuration_cache[configuration]['initial_living_cells_count'])

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
            list: Results from get_experiment_results() containing best configurations
        """
        self.initialize()
        for generation in range(1, self.generations):
            self.populate(generation=generation)
            self.compute_generation(generation=generation)
            self.adjust_mutation_rate(generation)
            self.check_for_stagnation(generation)

        return self.get_experiment_results()

    def calc_statistics(self, generation, scores, lifespans, alive_growth_rates, stableness, max_alive_cells_count, initial_living_cells_count):
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

    def get_experiment_results(self):
        # Final selection of best configurations
        fitness_scores = [(config, self.configuration_cache[config]['fitness_score'])
                          for config in self.population]

        fitness_scores_initial_population = [(config, self.configuration_cache[config]['fitness_score'])
                                             for config in self.initial_population]

        logging.info(f""" initial population size : {
                     len(fitness_scores_initial_population)}""")
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

            logging.info(f"""Top Configuration:
                Configuration: {config}
                Fitness Score: {self.configuration_cache[config]['fitness_score']}
                'normalized_fitness_score': self.configuration_cache[config]['normalized_fitness_score'],
                Lifespan: {self.configuration_cache[config]['lifespan']}
                Total Alive Cells: {self.configuration_cache[config]['max_alive_cells_count']}
                Alive Growth: {self.configuration_cache[config]['alive_growth']}
                Initial Configuration Living Cells Count: {self.configuration_cache[config]['initial_living_cells_count']}""")
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

            logging.info(f"""Initial Configuration:
                Configuration: {config}
                Fitness Score: {self.configuration_cache[config]['fitness_score']}
                'normalized_fitness_score': self.configuration_cache[config]['normalized_fitness_score'],
                Lifespan: {self.configuration_cache[config]['lifespan']}
                Total Alive Cells: {self.configuration_cache[config]['max_alive_cells_count']}
                Alive Growth: {self.configuration_cache[config]['alive_growth']}
                Initial Configuration Living Cells Count: {self.configuration_cache[config]['initial_living_cells_count']}""")

            results.append(params_dict)

        initial_configurations_start_index = len(top_configs)
        return results, initial_configurations_start_index
