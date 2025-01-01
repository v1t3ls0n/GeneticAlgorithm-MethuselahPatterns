import logging
from GameOfLife import GameOfLife
import random
import math
import numpy as np
import collections


class GeneticAlgorithm:
    """
    A genetic algorithm implementation for optimizing Conway's Game of Life configurations.
    Utilizes evolutionary principles to evolve configurations that maximize desired properties 
    such as lifespan, cell growth, and pattern stability.

    The algorithm operates on a population of Game of Life configurations, iteratively applying
    genetic operations like selection, crossover, and mutation to evolve configurations over
    multiple generations. Fitness scores are calculated based on defined weights for various metrics,
    guiding the evolutionary process towards optimal solutions.

    Attributes:
        grid_size (int): Dimensions of the NxN grid for Game of Life configurations.
        population_size (int): Number of configurations in each generation.
        generations (int): Total number of generations to simulate.
        initial_mutation_rate (float): Starting mutation probability.
        mutation_rate_lower_limit (float): Minimum allowed mutation rate to prevent excessive mutations.
        alive_cells_weight (float): Weight assigned to the maximum number of living cells in fitness calculation.
        lifespan_weight (float): Weight assigned to the configuration's lifespan in fitness calculation.
        alive_growth_weight (float): Weight assigned to the cell growth ratio in fitness calculation.
        stableness_weight (float): Weight assigned to the pattern stability in fitness calculation.
        initial_living_cells_count_penalty_weight (float): Penalty weight for large initial configurations to promote efficiency.
        predefined_configurations (Optional[List[tuple[int]]]): Pre-made Game of Life patterns to include in the initial population.
        population (set[tuple[int]]): Current generation's configurations represented as tuples of integers.
        configuration_cache (defaultdict): Cache storing evaluated configurations and their associated metrics.
        generations_cache (defaultdict): Cache storing statistics for each generation.
        mutation_rate_history (list[float]): History of mutation rate changes over generations.
        canonical_forms_cache (defaultdict[tuple, tuple[int]]): Cache for canonical forms of configurations to identify equivalences.
        block_frequencies_cache (defaultdict[tuple, dict]): Cache for block frequencies within configurations to detect recurring patterns.
        initial_population (set[tuple[int]]): Initial set of configurations before evolution begins.
        min_fitness (float): Minimum fitness score observed across all evaluations.
        max_fitness (float): Maximum fitness score observed across all evaluations.
        min_uniqueness_score (float): Minimum uniqueness score observed across all evaluations.
        max_uniqueness_score (float): Maximum uniqueness score observed across all evaluations.
    """

    def __init__(self, grid_size, population_size, generations, initial_mutation_rate, mutation_rate_lower_limit,
                 alive_cells_weight, lifespan_weight, alive_growth_weight, stableness_weight,
                 initial_living_cells_count_penalty_weight, predefined_configurations=None):
        """
        Initialize the Genetic Algorithm with specified configuration parameters.

        Sets up the initial state of the algorithm, including population parameters, mutation rates,
        fitness weights, and optional predefined configurations.

        Args:
            grid_size (int): Size of the NxN grid for Game of Life configurations.
            population_size (int): Number of configurations per generation.
            generations (int): Total number of generations to evolve.
            initial_mutation_rate (float): Starting probability of mutation per gene (cell).
            mutation_rate_lower_limit (float): Minimum mutation rate to maintain diversity.
            alive_cells_weight (float): Weight for maximizing the number of living cells in fitness calculation.
            lifespan_weight (float): Weight for maximizing the lifespan (number of unique states).
            alive_growth_weight (float): Weight for maximizing the cell growth ratio.
            stableness_weight (float): Weight for maximizing pattern stability.
            initial_living_cells_count_penalty_weight (float): Weight for penalizing large initial configurations to promote efficiency.
            predefined_configurations (Optional[List[tuple[int]]]): Predefined Game of Life patterns to include in the initial population.

        Raises:
            ValueError: If any of the provided weights are negative.
        """
        print("Initializing GeneticAlgorithm.")
        # Validate weights to ensure they are non-negative
        weights = [
            alive_cells_weight, lifespan_weight, alive_growth_weight,
            stableness_weight, initial_living_cells_count_penalty_weight
        ]
        if any(w < 0 for w in weights):
            raise ValueError("All fitness weights must be non-negative.")

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
            lambda: collections.defaultdict())
        self.generations_cache = collections.defaultdict(
            lambda: collections.defaultdict())
        self.canonical_forms_cache = collections.defaultdict(
            tuple)  # Cache for canonical forms
        self.block_frequencies_cache = collections.defaultdict(
            dict)  # Cache for block frequencies
        self.population = set()
        self.initial_population = set()
        self.mutation_rate_history = []
        self.min_fitness = float('inf')  # Minimum fitness score observed
        self.max_fitness = float('-inf')  # Maximum fitness score observed
        self.min_uniqueness_score = float('inf')  # Minimum uniqueness score observed
        self.max_uniqueness_score = float('-inf')  # Maximum uniqueness score observed
        self.predefined_configurations = predefined_configurations

    def calc_fitness(self, lifespan, max_alive_cells_count, alive_growth, stableness, initial_living_cells_count):
        """
        Calculate the weighted fitness score by combining multiple optimization objectives.

        The fitness score is a composite value that balances:
            - Configuration longevity through lifespan
            - Peak population through the maximum number of alive cells
            - Growth dynamics through the alive growth ratio
            - Pattern stability through the stableness score
            - Efficiency through the initial size penalty

        Args:
            lifespan (int): Number of unique states the configuration persists before stopping.
            max_alive_cells_count (int): Maximum number of living cells observed in any generation.
            alive_growth (float): Ratio indicating growth dynamics.
            stableness (float): Measure of pattern stability over generations.
            initial_living_cells_count (int): Starting number of living cells in the configuration.

        Returns:
            float: The combined fitness score, adjusted by configuration parameters.

        Notes:
            - Higher fitness scores indicate more desirable configurations.
            - The large configuration penalty discourages overly complex initial patterns.
        """
        lifespan_score = lifespan * self.lifespan_weight
        alive_cells_score = max_alive_cells_count * self.alive_cells_weight
        growth_score = alive_growth * self.alive_growth_weight
        stableness_score = stableness * self.stableness_weight
        large_configuration_penalty = (
            1 / max(1, initial_living_cells_count * self.initial_living_cells_count_penalty_weight))
        combined_fitness = (lifespan_score + alive_cells_score + growth_score + stableness_score) * large_configuration_penalty
        return combined_fitness

    def evaluate(self, configuration):
        """
        Evaluate a configuration by simulating its evolution and calculating its fitness.

        Simulates the configuration through Conway's Game of Life, computes various metrics, and
        calculates a fitness score based on predefined weights. The results are cached to avoid
        redundant evaluations in future generations.

        Args:
            configuration (tuple[int]): Flattened 1D representation of the Game of Life grid, where
                                         1 represents a living cell and 0 represents a dead cell.

        Returns:
            dict: A dictionary containing evaluation results for the configuration, including:
                - fitness_score (float): Overall fitness value.
                - normalized_fitness_score (float): Fitness score normalized between 0 and 1.
                - history (tuple[tuple[int]]): Complete evolution history as a tuple of grid states.
                - lifespan (int): Number of unique states before termination.
                - alive_growth (float): Ratio indicating growth dynamics.
                - max_alive_cells_count (int): Peak number of living cells observed.
                - is_static (bool): Indicates if the pattern became static.
                - is_periodic (bool): Indicates if the pattern entered a periodic loop.
                - stableness (float): Measure of pattern stability.
                - initial_living_cells_count (int): Starting population size.

        Raises:
            ValueError: If the configuration size does not match the expected grid size.

        Notes:
            - Normalized fitness scores facilitate consistent selection and comparison across generations.
            - The caching mechanism optimizes performance by preventing re-evaluation of identical configurations.
        """
        configuration_tuple = tuple(configuration)

        # Check if the configuration is already cached
        if configuration_tuple in self.configuration_cache:
            # Recalculate normalized fitness if min/max fitness changed
            cached_fitness = self.configuration_cache[configuration_tuple]['fitness_score']
            self.configuration_cache[configuration_tuple]['normalized_fitness_score'] = (
                (cached_fitness - self.min_fitness) / (self.max_fitness - self.min_fitness)
                if self.max_fitness != self.min_fitness else 1.0
            )
            return self.configuration_cache[configuration_tuple]

        expected_size = self.grid_size * self.grid_size
        if len(configuration_tuple) != expected_size:
            raise ValueError(f"""Configuration size must be {expected_size}, but got {len(configuration_tuple)}""")

        def max_difference_with_distance(lst):
            """
            Calculate the maximum difference in alive cell counts over distance.

            This metric captures the growth dynamics by measuring how the number of alive cells
            changes over time, factoring in the generation distance.

            Args:
                lst (list[int]): List of alive cell counts across generations.

            Returns:
                float: The normalized maximum difference.
            """
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

        # Initialize and run the Game of Life simulation
        game = GameOfLife(self.grid_size, configuration_tuple)
        game.run()

        # Extract metrics from the simulation
        max_alive_cells_count = max(game.alive_history) if game.alive_history else 0
        initial_living_cells_count = sum(configuration_tuple)
        alive_growth = max_difference_with_distance(game.alive_history)
        stableness = game.stable_count / game.max_stable_generations if game.max_stable_generations > 0 else 0

        # Calculate fitness score
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

        # Cache the evaluation results
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

    def calculate_corrected_scores(self):
        """
        Calculate corrected scores by combining canonical form and cell frequency penalties.

        This method adjusts fitness scores by penalizing configurations that are overly common
        or exhibit highly recurring patterns, promoting diversity within the population.

        Returns:
            list[tuple[tuple[int], float]]: List of configurations paired with their corrected scores.
        """
        total_cells = self.grid_size * self.grid_size
        frequency_vector = np.zeros(total_cells)
        canonical_frequency = {}
        block_frequencies = {}

        uniqueness_scores = []

        # Aggregate frequency data across the population
        for config in self.population:
            frequency_vector += np.array(config)
            canonical = self.canonical_form(config, self.grid_size)
            canonical_frequency[canonical] = canonical_frequency.get(canonical, 0) + 1

            # Detect and update block frequencies
            block_frequency = self.detect_recurrent_blocks(config, self.grid_size)
            for block, count in block_frequency.items():
                block_frequencies[block] = block_frequencies.get(block, 0) + count

        corrected_scores = []

        for config in self.population:
            # Retrieve normalized fitness score
            normalized_fitness = self.configuration_cache[config].get('normalized_fitness_score', 0)
            active_cells = [i for i, cell in enumerate(config) if cell == 1]

            # Canonical form penalty: configurations sharing the same canonical form are penalized
            canonical = self.canonical_form(config, self.grid_size)
            canonical_penalty = canonical_frequency.get(canonical, 1)

            # Cell frequency penalty: configurations with highly common active cells are penalized
            if not active_cells:
                cell_frequency_penalty = 1  # Avoid division by zero for empty configurations
            else:
                total_frequency = sum(frequency_vector[i] for i in active_cells)
                cell_frequency_penalty = (total_frequency / len(active_cells)) ** 3

            # Block recurrence penalty: configurations with recurring blocks are penalized
            block_frequency_penalty = 1
            block_frequency = self.detect_recurrent_blocks(config, self.grid_size)
            for block, count in block_frequency.items():
                block_frequency_penalty *= block_frequencies.get(block, 1)

            # Combine penalties to form the uniqueness score
            uniqueness_score = (
                canonical_penalty * cell_frequency_penalty  * block_frequency_penalty 
            )
            uniqueness_scores.append(uniqueness_score)

            # Calculate corrected score by adjusting normalized fitness with uniqueness
            corrected_score = normalized_fitness / max(1, uniqueness_score)
            corrected_scores.append((config, corrected_score))

        # Update global min/max uniqueness scores
        if uniqueness_scores:
            self.min_uniqueness_score = min(self.min_uniqueness_score, min(uniqueness_scores))
            self.max_uniqueness_score = max(self.max_uniqueness_score, max(uniqueness_scores))
        else:
            self.min_uniqueness_score = 0
            self.max_uniqueness_score = 1

        return corrected_scores

    def populate(self, generation):
        """
        Generate the next generation of configurations using evolutionary strategies.

        The method employs different strategies based on the generation number:
            - **Every 10th Generation**: Introduces fresh configurations using predefined pattern types:
                - Clustered cells
                - Scattered cells
                - Basic geometric patterns
            - **Other Generations**: Applies genetic operations including selection, crossover, and mutation to evolve the population.

        Args:
            generation (int): Current generation number.

        Raises:
            ValueError: If sorting or selection processes fail due to inconsistent data.

        Notes:
            - The use of normalized fitness scores ensures consistent and unbiased selection.
            - The population is maintained at a fixed size by selecting top-performing configurations.
        """
        new_population = set()
        if generation % 10 != 0:
            # **Regular Generations**: Perform genetic operations
            amount = self.population_size // 4
            for _ in range(amount):
                try:
                    parent1, parent2 = self.select_parents(generation=generation)
                except ValueError as e:
                    logging.error(f"Error selecting parents: {e}")
                    continue
                child = self.crossover(parent1, parent2)
                if random.uniform(0, 1) < self.mutation_rate:
                    child = self.mutate(child)
                new_population.add(child)
        else:
            # **Every 10th Generation**: Introduce fresh configurations
            uniform_amount = self.population_size // 3
            rem_amount = self.population_size % 3
            new_population = set(self.enrich_population_with_variety(
                clusters_type_amount=uniform_amount + rem_amount,
                scatter_type_amount=uniform_amount,
                basic_patterns_type_amount=uniform_amount
            ))

        # Combine new and existing populations
        combined = list(new_population) + list(self.population)

        # Evaluate combined configurations and sort based on normalized fitness scores
        combined_evaluated = [(config, self.evaluate(config)['normalized_fitness_score'])
                              for config in combined]
        combined_evaluated.sort(key=lambda x: x[1], reverse=True)

        # Select the top configurations to form the new population
        self.population = set(
            [config for config, _ in combined_evaluated[:self.population_size]]
        )

        # If population is still below the desired size, fill in with remaining configurations
        i = 0
        while len(self.population) < self.population_size and i < len(combined_evaluated) - self.population_size:
            self.population.add(combined_evaluated[self.population_size + i][0])
            i += 1

        # Ensure population size consistency
        if len(self.population) != self.population_size:
            logging.warning(f"""Population size mismatch after population step. Expected {self.population_size}, got {len(self.population)}""")

    def select_parents(self, generation):
        """
        Select two parent configurations using a combination of selection methods.

        The selection process integrates canonical forms and block frequency analysis to adjust fitness
        scores dynamically, penalizing configurations that are overly common or exhibit highly recurring patterns.

        Selection Methods:
            1. **Normalized Probability Selection (50% chance)**:
               - Selects parents based on fitness scores proportional to their normalized fitness.
            2. **Tournament Selection (25% chance)**:
               - Selects parents by competing subsets of configurations.
            3. **Rank-Based Selection (25% chance)**:
               - Selects parents based on their ranking within the population.

        Args:
            generation (int): Current generation number.

        Returns:
            tuple[tuple[int], tuple[int]]: Two parent configurations selected for crossover.

        Raises:
            ValueError: If the population size is insufficient for tournament selection.
        """

        # Check if it's every 10th generation
        if generation % 10 == 0:
            corrected_scores = self.calculate_corrected_scores()
            logging.debug(f"Generation {generation}: Calculated corrected_scores for 10th generation.")
        else:
            corrected_scores = [(config, self.configuration_cache[config]['normalized_fitness_score']) for config in self.population]
            logging.debug(f"Generation {generation}: Using normalized_fitness_score for corrected_scores.")

        # Define inner functions after corrected_scores is assigned
        def normalized_probability_selection():
            """
            Select parents based on normalized probability proportional to their corrected scores.

            Returns:
                list[tuple[int]]: Two parent configurations selected based on weighted probabilities.
            """
            if not corrected_scores:
                logging.warning("No corrected_scores available for selection.")
                return random.choices(list(self.population), k=2)
            
            configs, scores = zip(*corrected_scores) if corrected_scores else ([], [])
            total_score = sum(scores)
            if total_score == 0:
                logging.info("Total score is 0, selecting random parents.")
                return random.choices(configs, k=2)
            probabilities = [score / total_score for score in scores]
            selected = random.choices(configs, weights=probabilities, k=2)
            logging.debug(f"Selected parents via normalized_probability_selection: {selected}")
            return selected

        def tournament_selection():
            """
            Select parents through tournament selection by competing subsets.

            Returns:
                tuple[tuple[int], tuple[int]]: Two parent configurations selected from tournaments.

            Raises:
                ValueError: If the tournament size exceeds the current population size.
            """
            tournament_size = min(3, self.population_size // 4) or 2  # Ensure at least 2
            if len(corrected_scores) < tournament_size * 2:
                raise ValueError("Not enough configurations for tournament selection.")

            candidates1 = random.sample(corrected_scores, k=tournament_size)
            candidates2 = random.sample(corrected_scores, k=tournament_size)
            parent1 = max(candidates1, key=lambda x: x[1])[0]
            parent2 = max(candidates2, key=lambda x: x[1])[0]
            logging.debug(f"Selected parents via tournament_selection: {parent1}, {parent2}")
            return parent1, parent2

        def rank_based_selection():
            """
            Select parents based on their rank within the population.

            Higher-ranked configurations have a higher probability of being selected.

            Returns:
                tuple[tuple[int], tuple[int]]: Two parent configurations selected based on rank.

            Raises:
                ValueError: If the population is empty.
            """
            if not corrected_scores:
                raise ValueError("No configurations available for rank-based selection.")

            sorted_scores = sorted(corrected_scores, key=lambda x: x[1], reverse=True)
            configs, scores = zip(*sorted_scores)
            ranks = range(1, len(configs) + 1)
            total_rank = sum(ranks)
            probabilities = [rank / total_rank for rank in ranks]
            selected = random.choices(configs, weights=probabilities, k=2)
            logging.debug(f"Selected parents via rank_based_selection: {selected}")
            return selected

        # Define the selection methods as a list
        selection_methods = [normalized_probability_selection, tournament_selection, rank_based_selection]

        # Define the selection weights
        selection_weights = [0.5, 0.25, 0.25]

        # Select the method based on weights
        selected_method = random.choices(selection_methods, weights=selection_weights, k=1)[0]

        # Execute the selected selection method
        parents = selected_method()

        return parents

    def mutate(self, configuration):
        """
        Apply a mutation strategy to a configuration to introduce genetic diversity.

        Three mutation strategies are available, each with a specified probability:
            1. **Basic Mutation (40% chance)**:
               - Randomly flips individual cells based on mutation rate.
               - Ensures uniform distribution of changes.
            2. **Cluster Mutation (40% chance)**:
               - Flips cells within random 3x3 neighborhoods.
               - Creates localized pattern alterations.
            3. **Harsh Mutation (20% chance)**:
               - Flips large contiguous blocks of cells.
               - Enables significant pattern transformations.

        Args:
            configuration (tuple[int]): Configuration to mutate, represented as a tuple of integers.

        Returns:
            tuple[int]: Mutated configuration as a tuple of integers.
        """
        def mutate_basic(configuration):
            """
            Perform basic mutation by flipping individual cells based on mutation probability.

            Args:
                configuration (tuple[int]): Configuration to mutate.

            Returns:
                tuple[int]: Mutated configuration.
            """
            new_configuration = list(configuration)
            for i in range(len(configuration)):
                if random.uniform(0, 1) < min(0.5, self.mutation_rate * 5):
                    new_configuration[i] = 0 if configuration[i] else 1
            return tuple(new_configuration)

        def mutate_harsh(configuration):
            """
            Perform harsh mutation by flipping large contiguous blocks of cells.

            Args:
                configuration (tuple[int]): Configuration to mutate.

            Returns:
                tuple[int]: Mutated configuration.
            """
            new_configuration = list(configuration)
            cluster_size = random.randint(1, max(1, len(new_configuration) // 10))  # Limit cluster size
            start = random.randint(0, len(new_configuration) - 1)
            value = random.randint(0, 1)
            for j in range(cluster_size):
                idx = (start + j) % len(new_configuration)
                new_configuration[idx] = value
            return tuple(new_configuration)

        def mutate_clusters(configuration, mutation_rate=0.1, cluster_size=3):
            """
            Perform cluster mutation by flipping cells within random neighborhoods.

            Args:
                configuration (tuple[int]): Configuration to mutate.
                mutation_rate (float): Probability of mutating each cluster.
                cluster_size (int): Number of clusters to mutate.

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
            return tuple(mutated)

        # Define mutation methods and their respective probabilities
        mutation_methods = [mutate_basic, mutate_clusters, mutate_harsh]
        mutation_probabilities = [0.4, 0.4, 0.2]
        mutate_func = random.choices(mutation_methods, weights=mutation_probabilities, k=1)[0]
        mutated_configuration = mutate_func(configuration)
        logging.debug(f"Mutated configuration from {configuration} to {mutated_configuration}")
        return mutated_configuration

    def crossover(self, parent1, parent2):
        """
        Create child configurations by combining genetic material from two parents using crossover strategies.

        Three crossover methods are available, each with a specified probability:
            1. **Basic Crossover (30% chance)**:
               - Alternates cells from each parent.
               - Simple but effective mixing strategy.
            2. **Simple Crossover (30% chance)**:
               - Alternates entire rows from each parent.
               - Preserves horizontal patterns.
            3. **Complex Crossover (40% chance)**:
               - Selects blocks based on living cell density.
               - Intelligently combines high-fitness regions.

        Args:
            parent1 (tuple[int]): First parent configuration.
            parent2 (tuple[int]): Second parent configuration.

        Returns:
            tuple[int]: Child configuration created through crossover.

        Raises:
            ValueError: If parent configurations do not match the expected grid size.
        """
        def crossover_basic(parent1, parent2):
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
            return tuple(child)

        def crossover_simple(parent1, parent2):
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
                logging.error(f"""Parent configurations must be {total_cells}, but got sizes: {len(parent1)} and {len(parent2)}""")
                raise ValueError(f"""Parent configurations must be {total_cells}, but got sizes: {len(parent1)} and {len(parent2)}""")

            blocks_parent1 = [parent1[i * N: (i + 1) * N] for i in range(N)]
            blocks_parent2 = [parent2[i * N: (i + 1) * N] for i in range(N)]

            child_blocks = []
            for i in range(N):
                if i % 2 == 0:
                    child_blocks.extend(blocks_parent2[i])
                else:
                    child_blocks.extend(blocks_parent1[i])

            if len(child_blocks) != total_cells:
                logging.debug(f"""Child size mismatch, expected {total_cells}, got {len(child_blocks)}""")
                child_blocks += [0] * (total_cells - len(child_blocks))

            child_configuration = tuple(child_blocks)
            logging.debug(f"Crossover simple created child configuration: {child_configuration}")
            return child_configuration

        def crossover_complex(parent1, parent2):
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
            reminder = N % 2

            if len(parent1) != total_cells or len(parent2) != total_cells:
                logging.info(f"""Parent configurations must be {total_cells}, but got sizes: {len(parent1)} and {len(parent2)}""")
                raise ValueError(f"""Parent configurations must be {total_cells}, but got sizes: {len(parent1)} and {len(parent2)}""")

            block_size = N // 2  # Define the block size to split the grid
            blocks_parent1 = [
                parent1[i * block_size:(i + 1) * block_size] for i in range(2)]
            blocks_parent2 = [
                parent2[i * block_size:(i + 1) * block_size] for i in range(2)]

            block_alive_counts_parent1 = [sum(block) for block in blocks_parent1]
            block_alive_counts_parent2 = [sum(block) for block in blocks_parent2]
            max_alive_cells_parent1 = sum(block_alive_counts_parent1)
            max_alive_cells_parent2 = sum(block_alive_counts_parent2)

            if max_alive_cells_parent1 > 0:
                probabilities_parent1 = [
                    (alive_count / max_alive_cells_parent1) if alive_count > 0 else (1 / total_cells)
                    for alive_count in block_alive_counts_parent1
                ]
            else:
                probabilities_parent1 = [1 / total_cells] * 2

            if max_alive_cells_parent2 > 0:
                probabilities_parent2 = [
                    (alive_count / max_alive_cells_parent2) if alive_count > 0 else (1 / total_cells)
                    for alive_count in block_alive_counts_parent2
                ]
            else:
                probabilities_parent2 = [1 / total_cells] * 2

            selected_blocks_parent1 = random.choices(
                range(2), weights=probabilities_parent1, k=(2 // 2) + reminder)
            remaining_blocks_parent2 = [i for i in range(2) if i not in selected_blocks_parent1]
            selected_blocks_parent2 = random.choices(
                remaining_blocks_parent2,
                weights=[probabilities_parent2[i] for i in remaining_blocks_parent2],
                k=2 // 2
            )

            child_blocks = []
            for i in range(2):
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
                logging.info(f"""Child size mismatch, expected {total_cells}, got {len(child_blocks)}""")
                child_blocks += [0] * (total_cells - len(child_blocks))

            child_configuration = tuple(child_blocks)
            logging.debug(f"Crossover complex created child configuration: {child_configuration}")
            return child_configuration

        # Define crossover methods and their respective probabilities
        crossover_methods = [crossover_basic, crossover_simple, crossover_complex]
        crossover_probabilities = [0.3, 0.3, 0.4]
        crossover_func = random.choices(crossover_methods, weights=crossover_probabilities, k=1)[0]
        child_configuration = crossover_func(parent1, parent2)
        logging.debug(f"Crossover created child configuration: {child_configuration}")
        return child_configuration

    def enrich_population_with_variety(self, clusters_type_amount, scatter_type_amount, basic_patterns_type_amount):
        """
        Generate a diverse set of configurations using distinct pattern types.

        This method enhances the initial population by introducing variety through three distinct
        pattern generation strategies:
            1. **Clustered Configurations**: Groups of adjacent living cells.
            2. **Scattered Configurations**: Randomly distributed living cells.
            3. **Basic Pattern Configurations**: Simple geometric arrangements with variations.

        Args:
            clusters_type_amount (int): Number of cluster-based configurations to generate.
            scatter_type_amount (int): Number of scattered configurations to generate.
            basic_patterns_type_amount (int): Number of basic pattern configurations to generate.

        Returns:
            list[tuple[int]]: A list of diverse initial configurations represented as tuples of integers.
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
            scattered_cells = random.randint(min_scattered_cells, max_scattered_cells)
            scattered_indices = random.sample(range(total_cells), scattered_cells)
            for index in scattered_indices:
                configuration[index] = 1
            population_pool.append(tuple(configuration))

        # Generate Basic Patterns Configurations
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
                additional_cells = random.sample(
                    [i for i in range(total_cells) if configuration[i] == 0],
                    pattern_cells - current_live_cells
                )
                for index in additional_cells:
                    configuration[index] = 1

            population_pool.append(tuple(configuration))

        logging.debug(f"Enriched population with variety: {len(population_pool)} configurations added.")
        return population_pool

    def initialize(self):
        """
        Initialize the population with diverse configurations and evaluate the initial generation.

        The population is created by evenly distributing configurations among clustered,
        scattered, and basic pattern types. Predefined configurations can be included if provided.
        After generation, the initial population is evaluated and cached.

        Notes:
            - Ensures a balanced starting point for the evolutionary process.
            - Caching the initial population avoids redundant evaluations in subsequent generations.
        """
        uniform_amount = self.population_size // 3
        rem_amount = self.population_size % 3
        population = self.enrich_population_with_variety(
            clusters_type_amount=uniform_amount + rem_amount,
            scatter_type_amount=uniform_amount,
            basic_patterns_type_amount=uniform_amount
        )

        self.initial_population = set(population)
        self.population = set(population)
        self.compute_generation(generation=0)
        logging.info(f"Initialized population with {len(self.population)} configurations.")

    def adjust_mutation_rate(self, generation):
        """
        Dynamically adjust the mutation rate based on population fitness trends.

        The mutation rate is modified to balance exploration and exploitation:
            - **Increase Mutation Rate**: When fitness plateaus or stagnates to encourage exploration.
            - **Decrease Mutation Rate**: When fitness shows consistent improvement to fine-tune solutions.
            - **Maintain Bounds**: Ensures mutation rate stays within specified limits.

        Args:
            generation (int): Current generation number.

        Notes:
            - Utilizes historical fitness data to inform mutation rate adjustments.
            - Prevents mutation rate from becoming too low or too high, maintaining evolutionary diversity.
        """
        if generation < 1:
            logging.debug("Not enough history to adjust mutation rate.")
            return

        if generation < 10:
            # Not enough history for 10 generations, use basic improvement ratio
            previous_avg_fitness = self.generations_cache[generation - 1].get('avg_fitness', 1)
            current_avg_fitness = self.generations_cache[generation].get('avg_fitness', 1)
            improvement_ratio = previous_avg_fitness / max(1, current_avg_fitness)
            self.mutation_rate = max(
                self.mutation_rate_lower_limit,
                min(self.initial_mutation_rate, improvement_ratio * self.mutation_rate)
            )
            logging.debug(f"""Generation {generation}: Adjusted mutation rate to {self.mutation_rate:.4f} based on improvement ratio {improvement_ratio:.4f}.""")
        else:
            # Calculate improvement over the last 10 generations
            avg_fitness_last_10 = [
                self.generations_cache[g].get('avg_fitness', 1)
                for g in range(generation - 10, generation)
            ]
            improvement_ratio = avg_fitness_last_10[-1] / max(1, avg_fitness_last_10[0])

            if improvement_ratio < 1.01:
                # Plateau detected, increase mutation rate
                self.mutation_rate = min(
                    self.initial_mutation_rate, self.mutation_rate * 1.2
                )
                logging.debug(f"""Generation {generation}: Plateau detected with improvement ratio {improvement_ratio:.4f}. Increased mutation rate to {self.mutation_rate:.4f}.""")
            else:
                # Fitness improving, decrease mutation rate
                self.mutation_rate = max(
                    self.mutation_rate_lower_limit, self.mutation_rate * 0.9
                )
                logging.debug(f"""Generation {generation}: Fitness improving with improvement ratio {improvement_ratio:.4f}. Decreased mutation rate to {self.mutation_rate:.4f}.""")

    def check_for_stagnation(self, last_generation):
        """
        Monitor evolution progress and detect stagnation patterns to adjust mutation rate accordingly.

        Analyzes the average fitness over recent generations to identify:
            - **Complete Stagnation**: Identical fitness scores over multiple generations.
            - **Partial Stagnation**: Low diversity in fitness scores.

        Adjusts the mutation rate to escape local optima:
            - **High Stagnation**: Significantly increase mutation rate.
            - **Moderate Stagnation**: Slightly increase mutation rate.

        Args:
            last_generation (int): Current generation number.

        Notes:
            - Relies on the last 10 generations for assessing stagnation.
            - Prevents premature convergence by injecting diversity when needed.
        """
        if last_generation < 10:
            logging.debug("Not enough generations to assess stagnation.")
            return

        avg_fitness = [
            self.generations_cache[g].get('avg_fitness', 0)
            for g in range(last_generation - 10, last_generation)
        ]

        unique_fitness_scores = len(set(avg_fitness))
        total_generations = len(avg_fitness)
        stagnation_score = total_generations / unique_fitness_scores if unique_fitness_scores > 0 else float('inf')

        if stagnation_score > 5:
            # Significant stagnation detected, increase mutation rate more aggressively
            self.mutation_rate = min(
                self.initial_mutation_rate, self.mutation_rate * 1.5
            )
            logging.info(f"""Generation {last_generation}: High stagnation detected (score: {stagnation_score:.2f}). Increased mutation rate to {self.mutation_rate:.4f}.""")
        elif stagnation_score > 2:
            # Moderate stagnation detected, increase mutation rate slightly
            self.mutation_rate = min(
                self.initial_mutation_rate, self.mutation_rate * 1.2
            )
            logging.info(f"""Generation {last_generation}: Moderate stagnation detected (score: {stagnation_score:.2f}). Increased mutation rate to {self.mutation_rate:.4f}.""")
        else:
            logging.debug(f"""Generation {last_generation}: No significant stagnation detected (score: {stagnation_score:.2f}). Mutation rate remains at {self.mutation_rate:.4f}.""")

    def compute_generation(self, generation):
        """
        Evaluate the current generation's population and record statistical metrics.

        Processes the current generation by:
            - Calculating fitness metrics for all configurations.
            - Updating the generation cache with aggregated statistics.
            - Tracking the mutation rate history.

        Args:
            generation (int): Current generation number.

        Notes:
            - Outputs a progress message indicating the start of generation evaluation.
            - Aggregates various metrics to facilitate analysis and visualization.
        """
        print(f"Computing Generation {generation + 1} started.")
        scores = []
        lifespans = []
        alive_growth_rates = []
        max_alive_cells_count = []
        stableness = []
        initial_living_cells_count = []

        for configuration in self.population:
            self.evaluate(configuration)
            config_metrics = self.configuration_cache[configuration]
            scores.append(config_metrics['fitness_score'])
            lifespans.append(config_metrics['lifespan'])
            alive_growth_rates.append(config_metrics['alive_growth'])
            max_alive_cells_count.append(config_metrics['max_alive_cells_count'])
            stableness.append(config_metrics['stableness'])
            initial_living_cells_count.append(config_metrics['initial_living_cells_count'])

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
        logging.info(f"Generation {generation + 1} evaluated.")

    def run(self):
        """
        Execute the complete evolutionary process over the specified number of generations.

        Steps:
            1. Initialize the population with diverse configurations.
            2. Iterate through each generation:
                - Generate new configurations via population step.
                - Evaluate and record generation statistics.
                - Adjust mutation rate based on fitness trends.
                - Check for stagnation and respond accordingly.
            3. Retrieve and return the experiment results upon completion.

        Returns:
            tuple[list[dict], int]: A tuple containing:
                - A list of dictionaries with details of the best configurations.
                - The starting index for initial configurations.

        Notes:
            - The evolution process is logged for monitoring and debugging purposes.
            - Ensures that the population remains diverse and converges towards optimal configurations.
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

        Aggregates various metrics to compute statistical measures such as mean and standard deviation
        for each fitness-related attribute. These statistics facilitate trend analysis and visualization.

        Args:
            generation (int): Current generation number.
            scores (list[float]): Fitness scores for all configurations.
            lifespans (list[int]): Lifespan values for all configurations.
            alive_growth_rates (list[float]): Growth rates for all configurations.
            stableness (list[float]): Stability measures for all configurations.
            max_alive_cells_count (list[int]): Peak number of alive cells for all configurations.
            initial_living_cells_count (list[int]): Initial living cell counts for all configurations.

        Notes:
            - Stores both average and standard deviation for each metric.
            - Enables comprehensive statistical analysis across generations.
        """
        scores = np.array(scores)
        lifespans = np.array(lifespans)
        alive_growth_rates = np.array(alive_growth_rates)
        stableness = np.array(stableness)
        max_alive_cells_count = np.array(max_alive_cells_count)

        self.generations_cache[generation]['avg_fitness'] = np.mean(scores) if len(scores) > 0 else 0
        self.generations_cache[generation]['avg_lifespan'] = np.mean(lifespans) if len(lifespans) > 0 else 0
        self.generations_cache[generation]['avg_alive_growth_rate'] = np.mean(alive_growth_rates) if len(alive_growth_rates) > 0 else 0
        self.generations_cache[generation]['avg_max_alive_cells_count'] = np.mean(max_alive_cells_count) if len(max_alive_cells_count) > 0 else 0
        self.generations_cache[generation]['avg_stableness'] = np.mean(stableness) if len(stableness) > 0 else 0
        self.generations_cache[generation]['avg_initial_living_cells_count'] = np.mean(initial_living_cells_count) if len(initial_living_cells_count) > 0 else 0

        self.generations_cache[generation]['std_fitness'] = np.std(scores) if len(scores) > 0 else 0
        self.generations_cache[generation]['std_lifespan'] = np.std(lifespans) if len(lifespans) > 0 else 0
        self.generations_cache[generation]['std_alive_growth_rate'] = np.std(alive_growth_rates) if len(alive_growth_rates) > 0 else 0
        self.generations_cache[generation]['std_max_alive_cells_count'] = np.std(max_alive_cells_count) if len(max_alive_cells_count) > 0 else 0
        self.generations_cache[generation]['std_initial_living_cells_count'] = np.std(initial_living_cells_count) if len(initial_living_cells_count) > 0 else 0

        logging.debug(f"Generation {generation + 1} statistics calculated.")

    def get_experiment_results(self):
        """
        Retrieve the final results of the evolutionary experiment.

        Compiles the best-performing configurations from the final population and the initial population.
        Each configuration is accompanied by its associated metrics for comprehensive analysis.

        Returns:
            tuple[list[dict], int]: A tuple containing:
                - A list of dictionaries, each representing a top configuration with detailed metrics.
                - An integer indicating the starting index for initial configurations.

        Notes:
            - Sorts configurations based on their raw fitness scores in descending order.
            - Logs detailed information about each top and initial configuration for debugging and review.
        """
        # Final selection of best configurations based on raw fitness scores
        fitness_scores = [(config, self.configuration_cache[config]['fitness_score'])
                          for config in self.population]

        # Gather fitness scores from the initial population
        fitness_scores_initial_population = [(config, self.configuration_cache[config]['fitness_score'])
                                             for config in self.initial_population]

        logging.info(f"""Initial population size: {len(fitness_scores_initial_population)}""")
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        fitness_scores_initial_population.sort(
            key=lambda x: x[1], reverse=True)

        # Select top configurations
        top_configs = fitness_scores[:self.population_size]

        results = []

        # Process top configurations
        for config, _ in top_configs:
            config_metrics = self.configuration_cache[config]
            params_dict = {
                'fitness_score': config_metrics['fitness_score'],
                'normalized_fitness_score': config_metrics.get('normalized_fitness_score', 0),
                'lifespan': config_metrics['lifespan'],
                'max_alive_cells_count': config_metrics['max_alive_cells_count'],
                'alive_growth': config_metrics['alive_growth'],
                'stableness': config_metrics['stableness'],
                'initial_living_cells_count': config_metrics['initial_living_cells_count'],
                'history': list(config_metrics['history']),
                'config': config,
                'is_first_generation': False
            }

            logging.info(f"""Top Configuration:
Configuration: {config}
Fitness Score: {config_metrics['fitness_score']}
Normalized Fitness Score: {config_metrics.get('normalized_fitness_score', 0)}
Lifespan: {config_metrics['lifespan']}
Total Alive Cells: {config_metrics['max_alive_cells_count']}
Alive Growth: {config_metrics['alive_growth']}
Initial Configuration Living Cells Count: {config_metrics['initial_living_cells_count']}""")
            results.append(params_dict)

        # Process initial population configurations
        for config, _ in fitness_scores_initial_population:
            config_metrics = self.configuration_cache[config]
            params_dict = {
                'fitness_score': config_metrics['fitness_score'],
                'normalized_fitness_score': config_metrics.get('normalized_fitness_score', 0),
                'lifespan': config_metrics['lifespan'],
                'max_alive_cells_count': config_metrics['max_alive_cells_count'],
                'alive_growth': config_metrics['alive_growth'],
                'stableness': config_metrics['stableness'],
                'initial_living_cells_count': config_metrics['initial_living_cells_count'],
                'history': list(config_metrics['history']),
                'config': config,
                'is_first_generation': True
            }

            logging.info(f"""Initial Configuration:
Configuration: {config}
Fitness Score: {config_metrics['fitness_score']}
Normalized Fitness Score: {config_metrics.get('normalized_fitness_score', 0)}
Lifespan: {config_metrics['lifespan']}
Total Alive Cells: {config_metrics['max_alive_cells_count']}
Alive Growth: {config_metrics['alive_growth']}
Initial Configuration Living Cells Count: {config_metrics['initial_living_cells_count']}""")

            results.append(params_dict)

        initial_configurations_start_index = len(top_configs)
        logging.info(f"Experiment results compiled with {len(results)} configurations.")
        return results, initial_configurations_start_index

    def canonical_form(self, config, grid_size):
        """
        Compute the canonical form of a configuration by normalizing its position and rotation.

        This ensures that configurations are compared based on their fundamental patterns,
        disregarding their position and orientation within the grid.

        Args:
            config (tuple[int]): Flattened 1D representation of the grid.
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
            translated_grid = np.roll(translated_grid, shift=-min_col, axis=1)

            # Generate all rotations and find the lexicographically smallest
            rotations = [np.rot90(translated_grid, k).flatten() for k in range(4)]
            canonical = tuple(min(rotations, key=lambda x: tuple(x)))

        self.canonical_forms_cache[config] = canonical
        return canonical

    def detect_recurrent_blocks(self, config, grid_size):
        """
        Detect recurring canonical blocks within the configuration, considering rotations.

        This method divides the grid into blocks and identifies frequently occurring patterns,
        enhancing diversity by penalizing recurring blocks.

        Args:
            config (tuple[int]): Flattened 1D representation of the grid.
            grid_size (int): Size of the grid (NxN).

        Returns:
            dict: Frequency of each canonical block within the configuration.
        """
        if config in self.block_frequencies_cache:
            return self.block_frequencies_cache[config]

        block_size = max(2, grid_size // 4)  # Define a reasonable block size
        grid = np.array(config).reshape(grid_size, grid_size)
        block_frequency = {}

        for row in range(0, grid_size, block_size):
            for col in range(0, grid_size, block_size):
                block = grid[row:row + block_size, col:col + block_size]
                # Handle cases where the block size does not perfectly divide the grid
                if block.size == 0:
                    continue
                block_canonical = self.canonical_form(block.flatten(), block_size)
                if block_canonical not in block_frequency:
                    block_frequency[block_canonical] = 0
                block_frequency[block_canonical] += 1

        self.block_frequencies_cache[config] = block_frequency
        return block_frequency
