
# --- START OF GameOfLife.py ---

"""
GameOfLife.py
-------------

Implements Conway's Game of Life logic. Given an initial configuration (a 1D list
representing a 2D grid), this class simulates each generation until the system
reaches a static or periodic state, or until a maximum number of iterations is reached.

Classes:
    GameOfLife
"""

import random
import logging

class GameOfLife:
    """
    The GameOfLife class simulates Conway's Game of Life for a given initial state.
    It tracks the evolution of the grid through multiple generations and determines
    whether the pattern becomes static (unchanged) or periodic (repeats a previously
    seen state).

    Attributes:
        grid_size (int): The dimension N of the NxN grid.
        grid (list[int]): A flat list of length N*N, containing 0s and 1s (dead or alive cells).
        initial_state (tuple[int]): A tuple storing the initial configuration (immutable).
        history (list[tuple[int]]): A record of all the grid states encountered so far.
        game_iteration_limit (int): A hard limit on the total number of generations to simulate.
        stable_count (int): Counts how many consecutive generations remained static or periodic.
        max_stable_generations (int): Once stable_count reaches this limit, the simulation stops.
        lifespan (int): The total number of unique states the grid has passed through before stopping.
        is_static (int): A flag (0 or 1) indicating if the grid has become static.
        is_periodic (int): A flag (0 or 1) indicating if the grid has become periodic.
        max_alive_cells_count (int): The maximum number of living cells observed in any generation.
        alive_growth (float): The ratio between max and min living cells during the simulation.
        alive_history (list[int]): Number of living cells for each generation (for analysis).
        stableness (float): A ratio indicating how many times the grid was detected stable
                            compared to max_stable_generations.
    """

    def __init__(self, grid_size, initial_state=None):
        """
        Initialize the Game of Life simulation.

        Args:
            grid_size (int): The dimension N of the NxN grid.
            initial_state (Iterable[int], optional): A starting configuration.
                If None, a zero-initialized grid is created.
        """
        self.grid_size = grid_size
        self.grid = [0]*(grid_size*grid_size) if initial_state is None else list(initial_state)

        # Store the initial state of the grid
        self.initial_state = tuple(self.grid)
        self.history = [self.initial_state]

        self.game_iteration_limit = 15000
        self.stable_count = 0
        self.max_stable_generations = 10
        self.lifespan = 0
        self.is_static = 0
        self.is_periodic = 0

        self.max_alive_cells_count = 0
        self.alive_growth = 0
        self.alive_history = [sum(self.grid)]

    def step(self):
        """
        Perform a single step (one generation) in the Game of Life.
        Applies the classic rules:
            - A living cell (1) with 2 or 3 neighbors stays alive.
            - A dead cell (0) with exactly 3 neighbors becomes alive.
            - Otherwise, the cell dies (or remains dead).
        Checks if the new grid is identical to the current grid (static),
        or matches any previous state (periodic).
        """
        cur_grid = self.grid[:]
        new_grid = [0] * (self.grid_size * self.grid_size)

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                index = x * self.grid_size + y
                alive_neighbors = self.count_alive_neighbors(x, y)
                if cur_grid[index] == 1:
                    # Survives if it has 2 or 3 neighbors
                    if alive_neighbors in [2, 3]:
                        new_grid[index] = 1
                else:
                    # A dead cell becomes alive if it has exactly 3 neighbors
                    if alive_neighbors == 3:
                        new_grid[index] = 1

        newState = tuple(new_grid)
        curState = tuple(cur_grid)

        # Static check: No change from previous generation
        if newState == curState:
            self.is_static = 1
        # Periodic check: If newState appeared before (excluding the immediate last)
        elif newState in self.history[:-1]:
            self.is_periodic = 1
        else:
            self.grid = new_grid

    def run(self):
        """
        Run the simulation until a static or periodic state is reached,
        or until we exceed game_iteration_limit. Also tracks the number
        of living cells each generation, maximum living cells, and alive growth.
        Finally, computes a 'stableness' score based on stable_count and max_stable_generations.
        """
        limiter = self.game_iteration_limit

        while limiter and ((not self.is_static and not self.is_periodic) or self.stable_count < self.max_stable_generations):
            alive_cell_count = self.get_alive_cells_count()
            # If no cells alive, mark as static
            if not alive_cell_count:
                self.is_static = 1

            self.alive_history.append(alive_cell_count)
            self.history.append(tuple(self.grid[:]))

            if self.is_periodic or self.is_static:
                self.stable_count += 1

            self.lifespan += 1
            self.step()
            limiter -= 1

        self.lifespan = len(set(self.history))
        self.max_alive_cells_count = max(self.alive_history)
        self.alive_growth = max(self.alive_history) / max(1, min(self.alive_history)) if self.alive_history else 1
        self.stableness = self.stable_count / self.max_stable_generations

    def count_alive_neighbors(self, x, y):
        """
        Count how many neighbors of cell (x, y) are alive.

        Args:
            x (int): Row index.
            y (int): Column index.

        Returns:
            int: Number of living neighbors around (x, y).
        """
        alive = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                nx, ny = x + i, y + j
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    index = nx * self.grid_size + ny
                    alive += self.grid[index]
        return alive

    def get_alive_cells_count(self):
        """
        Returns the total number of living cells in the grid.
        """
        return sum(self.grid)

    def get_lifespan(self):
        """
        Return the total number of unique states the grid went through before stopping.
        """
        return self.lifespan

    def get_alive_history(self):
        """
        Return the list that tracks how many cells were alive at each generation.
        """
        return self.alive_history

    def reset(self):
        """
        Reset the grid to its initial state (useful for repeated experiments).
        """
        logging.debug("Resetting the grid to initial state.")
        self.grid = list(self.initial_state)
        self.history = [self.initial_state]
        self.is_static = False
        self.is_periodic = False
        self.lifespan = 0
        self.stable_count = 0
        self.alive_history = [sum(self.grid)]

# --- END OF GameOfLife.py ---


# --- START OF GeneticAlgorithm.py ---

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
        self.configuration_cache = collections.defaultdict(collections.defaultdict)
        self.generations_cache = collections.defaultdict(collections.defaultdict)
        self.predefined_configurations = predefined_configurations
        self.alive_cells_per_block = alive_cells_per_block
        self.alive_blocks = alive_blocks
        self.best_histories = []
        self.population = set()
        self.mutation_rate_history = [initial_mutation_rate]

    def calc_fitness(self, lifespan, max_alive_cells_count, alive_growth, stableness):
        """
        Combine multiple factors (lifespan, alive cell count, growth, and stableness)
        into one final fitness score based on their respective weights.
        """
        lifespan_score = lifespan * self.lifespan_weight
        alive_cells_score = max_alive_cells_count * self.alive_cells_weight
        growth_score = alive_growth * self.alive_growth_weight
        stableness_score = stableness * self.stableness_weight
        return lifespan_score + alive_cells_score + growth_score + stableness_score

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
            raise ValueError(f"Configuration size must be {expected_size}, but got {len(configuration_tuple)}")

        if configuration_tuple in self.configuration_cache:
            return self.configuration_cache[configuration_tuple]

        # Create and run a GameOfLife instance
        game = GameOfLife(self.grid_size, configuration_tuple)
        game.run()

        fitness_score = self.calc_fitness(
            lifespan=game.lifespan,
            max_alive_cells_count=game.max_alive_cells_count,
            alive_growth=game.alive_growth,
            stableness=game.stableness
        )

        self.configuration_cache[configuration_tuple] = {
            'fitness_score': fitness_score,
            'history': tuple(game.history),
            'lifespan': game.lifespan,
            'alive_growth': game.alive_growth,
            'max_alive_cells_count': game.max_alive_cells_count,
            'is_static': game.is_static,
            'is periodic': game.is_periodic,
            'stableness': game.stableness
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
        self.population = [config for config, _ in fitness_scores[:self.population_size]]
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
                if cell == 0 and random.uniform(0,1) < self.mutation_rate:
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
        probabilities = [(1+avg_fitness)*(1 + fitness_variance) for _score in fitness_scores]
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
            logging.info(f"Parent configurations must be {total_cells}, but got sizes: {len(parent1)} and {len(parent2)}")
            raise ValueError(f"Parent configurations must be {total_cells}, but got sizes: {len(parent1)} and {len(parent2)}")

        block_size = total_cells // N
        blocks_parent1 = [parent1[i*block_size:(i+1)*block_size] for i in range(N)]
        blocks_parent2 = [parent2[i*block_size:(i+1)*block_size] for i in range(N)]

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

        selected_blocks_parent1 = random.choices(range(N), weights=probabilities_parent1, k=(N//2)+reminder)
        remaining_blocks_parent2 = [i for i in range(N) if i not in selected_blocks_parent1]
        selected_blocks_parent2 = random.choices(
            remaining_blocks_parent2,
            weights=[probabilities_parent2[i] for i in remaining_blocks_parent2],
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
                selected_parent = random.choices([1, 2], weights=[0.5, 0.5], k=1)[0]
                if selected_parent == 1:
                    child_blocks.extend(blocks_parent1[i])
                else:
                    child_blocks.extend(blocks_parent2[i])

        # Fix length if needed
        if len(child_blocks) != total_cells:
            logging.info(f"Child size mismatch, expected {total_cells}, got {len(child_blocks)}")
            child_blocks = child_blocks + [0]*(total_cells - len(child_blocks))
        return tuple(child_blocks)

    def initialize(self):
        """
        Create the initial random population of configurations, evaluate them all,
        and record basic statistics for generation 0.
        """
        print(f"Generation 1 started.")
        self.population = [self.random_configuration() for _ in range(self.population_size)]

        generation = 0
        scores = []
        lifespans = []
        alive_growth_rates = []
        max_alive_cells_count = []
        stableness = []
        for configuration in self.population:
            self.evaluate(configuration)
            scores.append(self.configuration_cache[configuration]['fitness_score'])
            lifespans.append(self.configuration_cache[configuration]['lifespan'])
            alive_growth_rates.append(self.configuration_cache[configuration]['alive_growth'])
            max_alive_cells_count.append(self.configuration_cache[configuration]['max_alive_cells_count'])
            stableness.append(self.configuration_cache[configuration]['stableness'])

        self.calc_statistics(generation=generation,
                             scores=scores,
                             lifespans=lifespans,
                             alive_growth_rates=alive_growth_rates,
                             max_alive_cells_count=max_alive_cells_count,
                             stableness=stableness)

        self.generations_cache[generation]['avg_fitness'] = np.average(scores)
        self.generations_cache[generation]['avg_lifespan'] = np.average(lifespans)
        self.generations_cache[generation]['avg_alive_growth_rate'] = np.average(alive_growth_rates)
        self.generations_cache[generation]['avg_max_alive_cells_count'] = np.average(max_alive_cells_count)
        self.generations_cache[generation]['std_fitness'] = np.std(scores)
        self.generations_cache[generation]['std_lifespan'] = np.std(lifespans)
        self.generations_cache[generation]['std_alive_growth_rate'] = np.std(alive_growth_rates)
        self.generations_cache[generation]['std_max_alive_cells_count'] = np.std(max_alive_cells_count)

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

        num_parts = self.grid_size
        part_size = self.grid_size
        alive_blocks_indices = random.sample(range(self.grid_size), self.alive_blocks)

        for part_index in alive_blocks_indices:
            start_idx = part_index*part_size
            end_idx = start_idx + part_size
            chosen_cells = random.sample(range(start_idx, end_idx),
                                         min(self.alive_cells_per_block, end_idx-start_idx))
            for cell in chosen_cells:
                configuration[cell] = 1
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
        self.generations_cache[generation]['avg_alive_growth_rate'] = np.mean(alive_growth_rates)
        self.generations_cache[generation]['avg_max_alive_cells_count'] = np.mean(max_alive_cells_count)
        self.generations_cache[generation]['avg_stableness'] = np.mean(stableness)

        self.generations_cache[generation]['std_fitness'] = np.std(scores)
        self.generations_cache[generation]['std_lifespan'] = np.std(lifespans)
        self.generations_cache[generation]['std_alive_growth_rate'] = np.std(alive_growth_rates)
        self.generations_cache[generation]['std_max_alive_cells_count'] = np.std(max_alive_cells_count)

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
            print(f"Computing Generation {generation+1} started.")
            self.populate()

            scores = []
            lifespans = []
            alive_growth_rates = []
            max_alive_cells_count = []
            stableness = []

            for configuration in self.population:
                self.evaluate(configuration)
                scores.append(self.configuration_cache[configuration]['fitness_score'])
                lifespans.append(self.configuration_cache[configuration]['lifespan'])
                alive_growth_rates.append(self.configuration_cache[configuration]['alive_growth'])
                max_alive_cells_count.append(self.configuration_cache[configuration]['max_alive_cells_count'])
                stableness.append(self.configuration_cache[configuration]['stableness'])

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
            logging.info(f"  Fitness Score: {self.configuration_cache[config]['fitness_score']}")
            logging.info(f"  Lifespan: {self.configuration_cache[config]['lifespan']}")
            logging.info(f"  Total Alive Cells: {self.configuration_cache[config]['max_alive_cells_count']}")
            logging.info(f"  Alive Growth: {self.configuration_cache[config]['alive_growth']}")

        return best_configs

    def adjust_mutation_rate(self, generation):
        """
        Dynamically adjust the mutation rate if we detect improvement or stagnation.
        If there's no improvement in average fitness, attempt to increase the rate.
        Otherwise, if there's improvement, gently reduce it, but never below mutation_rate_lower_limit.
        """
        if generation > 10 and self.generations_cache[generation]['avg_fitness'] == self.generations_cache[generation - 1]['avg_fitness']:
            logging.info("Mutation rate increased due to stagnation.")
            self.mutation_rate = min(self.mutation_rate_lower_limit, self.mutation_rate * 1.2)
        elif self.generations_cache[generation]['avg_fitness'] > self.generations_cache[generation - 1]['avg_fitness']:
            self.mutation_rate = max(self.mutation_rate_lower_limit, self.mutation_rate * 0.9)

    def check_for_stagnation(self, last_generation):
        """
        Checks the last 10 generations for identical average fitness, indicating stagnation.
        If found, forcibly raises the mutation rate to try and escape local minima.

        Args:
            last_generation (int): Index of the current generation in the loop.
        """
        avg_fitness = [int(self.generations_cache[g]['avg_fitness']) for g in range(last_generation)]
        if len(avg_fitness) >= 10 and len(set(avg_fitness[-10:])) == 1:
            logging.warning("Stagnation detected in the last 10 generations!")
            self.mutation_rate = min(self.mutation_rate_lower_limit, self.mutation_rate)*1.5

# --- END OF GeneticAlgorithm.py ---


# --- START OF InteractiveSimulation.py ---

"""
InteractiveSimulation.py
------------------------

Provides an interactive visualization of:
1. The top evolved Game of Life configurations found by the GA (allowing the user to iterate through generations).
2. Graphical plots of the statistical metrics (fitness, lifespan, growth, alive cells, mutation rate) over GA generations.
"""

import logging
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

class InteractiveSimulation:
    """
    Displays the best 5 configurations found by the genetic algorithm in an interactive
    viewer, and also plots the evolution of various statistics across generations.

    Attributes:
        configurations (list[tuple[int]]): The top final configurations.
        histories (list[list[tuple[int]]]): The corresponding list of states (one list of states per config).
        grid_size (int): The dimension N of each NxN grid.
        generations_cache (dict): Contains average & std metrics across all generations.
        mutation_rate_history (list): Stores mutation rates across generations.
        current_config_index (int): Which of the best 5 configurations is currently displayed.
        current_generation (int): Which generation in that config's history is shown.
    """

    def __init__(self, configurations, histories, grid_size, generations_cache, mutation_rate_history):
        """
        Initialize figures for grid visualization and for the statistic plots.
        Set up keyboard callbacks for user interaction.

        Args:
            configurations (list): Top final configurations from the GA.
            histories (list): The state history for each of those top configurations.
            grid_size (int): NxN dimension for rendering the grid.
            generations_cache (dict): Aggregated metrics (fitness, lifespan, etc.).
            mutation_rate_history (list): Recorded mutation rates across generations.
        """
        print("Initializing Interactive Simulation and Metrics.")
        
        self.configurations = configurations
        self.histories = histories
        self.grid_size = grid_size
        self.generations_cache = generations_cache
        self.mutation_rate_history = mutation_rate_history
        self.current_config_index = 0
        self.current_generation = 0

        # Figure for the grid
        self.grid_fig, self.grid_ax = plt.subplots(figsize=(5, 5))
        self.grid_ax.set_title(f"Best Initial Configuration No {self.current_config_index + 1},  State No {self.current_generation}")
        self.grid_ax.set_xlabel("Use arrow keys:\n←/→ to move between states (days)\n↑/↓ to move between best initial configurations")

        self.update_grid()

        # Another figure for the metrics
        self.stats_fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 3, figure=self.stats_fig)

        # Subplots for average fitness, lifespan, growth rate, alive cells, mutation rate
        self.standardized_fitness_plot = self.stats_fig.add_subplot(gs[0, 0])
        self.standardized_fitness_plot.set_title('Standardized Fitness')
        self.standardized_fitness_plot.set_xlabel('Generation')
        self.standardized_fitness_plot.set_ylabel('Standardized Fitness')

        self.standardized_lifespan_plot = self.stats_fig.add_subplot(gs[0, 1])
        self.standardized_lifespan_plot.set_title('Standardized Lifespan')
        self.standardized_lifespan_plot.set_xlabel('Generation')
        self.standardized_lifespan_plot.set_ylabel('Standardized Lifespan')

        self.standardized_growth_rate_plot = self.stats_fig.add_subplot(gs[0, 2])
        self.standardized_growth_rate_plot.set_title('Standardized Growth Rate')
        self.standardized_growth_rate_plot.set_xlabel('Generation')
        self.standardized_growth_rate_plot.set_ylabel('Standardized Growth Rate')

        self.standardized_alive_cells_plot = self.stats_fig.add_subplot(gs[1, 0])
        self.standardized_alive_cells_plot.set_title('Standardized Alive Cells')
        self.standardized_alive_cells_plot.set_xlabel('Generation')
        self.standardized_alive_cells_plot.set_ylabel('Standardized Alive Cells')

        self.mutation_rate_plot = self.stats_fig.add_subplot(gs[1, 1:])
        self.mutation_rate_plot.set_title('Mutation Rate')
        self.mutation_rate_plot.set_xlabel('Generation')
        self.mutation_rate_plot.set_ylabel('Mutation Rate')

        # Keyboard and close events
        self.grid_fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.grid_fig.canvas.mpl_connect('close_event', self.on_close)
        self.stats_fig.canvas.mpl_connect('close_event', self.on_close)

        self.grid_fig.tight_layout()
        self.stats_fig.tight_layout()
        self.render_statistics()

    def on_close(self, event):
        """
        Called when the user closes the window. Closes all plots and exits the program.
        """
        logging.info("Window closed. Exiting program.")
        plt.close(self.grid_fig)
        plt.close(self.stats_fig)
        exit()

    def on_key(self, event):
        """
        Handles keyboard events:
            - UP: go to next configuration
            - DOWN: go to previous configuration
            - RIGHT: go to the next generation within the current configuration
            - LEFT: go to the previous generation
        """
        if event.key == 'up':
            self.next_configuration()
        elif event.key == 'down':
            self.previous_configuration()
        elif event.key == 'right':
            self.next_generation()
        elif event.key == 'left':
            self.previous_generation()

    def next_configuration(self):
        """
        Cycle forward among the top 5 best configurations.
        Resets the generation index to 0 upon changing configuration.
        """
        self.current_config_index = (self.current_config_index + 1) % len(self.configurations)
        self.current_generation = 0
        self.update_grid()

    def previous_configuration(self):
        """
        Cycle backward among the top 5 best configurations.
        Resets the generation index to 0 upon changing configuration.
        """
        self.current_config_index = (self.current_config_index - 1) % len(self.configurations)
        self.current_generation = 0
        self.update_grid()

    def next_generation(self):
        """
        Move to the next generation (if any) within the current configuration's evolution history.
        """
        if self.current_generation + 1 < len(self.histories[self.current_config_index]):
            self.current_generation += 1
            self.update_grid()

    def previous_generation(self):
        """
        Move to the previous generation (if any) within the current configuration's evolution history.
        """
        if self.current_generation > 0:
            self.current_generation -= 1
            self.update_grid()

    def update_grid(self):
        """
        Redraw the NxN grid for the current configuration index and generation index.
        """
        grid = [
            self.histories[self.current_config_index][self.current_generation][i*self.grid_size : (i+1)*self.grid_size]
            for i in range(self.grid_size)
        ]
        self.grid_ax.clear()
        self.grid_ax.imshow(grid, cmap="binary")
        self.grid_ax.set_title(f"Configuration {self.current_config_index + 1}, Generation {self.current_generation}")
        self.grid_ax.set_xlabel("Use arrow keys:\n←/→ to move between states (days)\n↑/↓ to move between best initial configurations")
        self.grid_fig.canvas.draw()

    def render_statistics(self):
        """
        Generate plots for average fitness, lifespan, growth rate, alive cells, and mutation rate
        across the generations as stored in generations_cache and mutation_rate_history.
        """
        generations = list(self.generations_cache.keys())

        # Standardized Fitness
        avg_fitness = [self.generations_cache[g]['avg_fitness'] for g in generations]
        std_fitness = [self.generations_cache[g]['std_fitness'] for g in generations]

        self.standardized_fitness_plot.clear()
        self.standardized_fitness_plot.plot(generations, avg_fitness, label='Standardized Fitness', color='blue')
        self.standardized_fitness_plot.fill_between(generations,
                                                    np.subtract(avg_fitness, std_fitness),
                                                    np.add(avg_fitness, std_fitness),
                                                    color='blue', alpha=0.2, label='Standard Deviation')
        self.standardized_fitness_plot.set_title("Standardized Fitness over Generations")
        self.standardized_fitness_plot.set_xlabel("Generation")
        self.standardized_fitness_plot.set_ylabel("Standardized Fitness")
        self.standardized_fitness_plot.legend()

        # Standardized Lifespan
        avg_lifespan = [self.generations_cache[g]['avg_lifespan'] for g in generations]
        std_lifespan = [self.generations_cache[g]['std_lifespan'] for g in generations]

        self.standardized_lifespan_plot.clear()
        self.standardized_lifespan_plot.plot(generations, avg_lifespan, label='Standardized Lifespan', color='green')
        self.standardized_lifespan_plot.fill_between(generations,
                                                     np.subtract(avg_lifespan, std_lifespan),
                                                     np.add(avg_lifespan, std_lifespan),
                                                     color='green', alpha=0.2, label='Standard Deviation')
        self.standardized_lifespan_plot.set_title("Standardized Lifespan over Generations")
        self.standardized_lifespan_plot.set_xlabel("Generation")
        self.standardized_lifespan_plot.set_ylabel("Standardized Lifespan")
        self.standardized_lifespan_plot.legend()

        # Standardized Growth Rate
        avg_alive_growth_rate = [self.generations_cache[g]['avg_alive_growth_rate'] for g in generations]
        std_alive_growth_rate = [self.generations_cache[g]['std_alive_growth_rate'] for g in generations]

        self.standardized_growth_rate_plot.clear()
        self.standardized_growth_rate_plot.plot(generations, avg_alive_growth_rate, label='Standardized Growth Rate', color='red')
        self.standardized_growth_rate_plot.fill_between(generations,
                                                        np.subtract(avg_alive_growth_rate, std_alive_growth_rate),
                                                        np.add(avg_alive_growth_rate, std_alive_growth_rate),
                                                        color='red', alpha=0.2, label='Standard Deviation')
        self.standardized_growth_rate_plot.set_title("Standardized Growth Rate over Generations")
        self.standardized_growth_rate_plot.set_xlabel("Generation")
        self.standardized_growth_rate_plot.set_ylabel("Standardized Growth Rate")
        self.standardized_growth_rate_plot.legend()

        # Standardized Alive Cells
        avg_max_alive_cells_count = [self.generations_cache[g]['avg_max_alive_cells_count'] for g in generations]
        std_max_alive_cells_count = [self.generations_cache[g]['std_max_alive_cells_count'] for g in generations]

        self.standardized_alive_cells_plot.clear()
        self.standardized_alive_cells_plot.plot(generations, avg_max_alive_cells_count, label='Standardized Alive Cells', color='purple')
        self.standardized_alive_cells_plot.fill_between(generations,
                                                        np.subtract(avg_max_alive_cells_count, std_max_alive_cells_count),
                                                        np.add(avg_max_alive_cells_count, std_max_alive_cells_count),
                                                        color='purple', alpha=0.2, label='Standard Deviation')
        self.standardized_alive_cells_plot.set_title("Standardized Alive Cells over Generations")
        self.standardized_alive_cells_plot.set_xlabel("Generation")
        self.standardized_alive_cells_plot.set_ylabel("Standardized Alive Cells")
        self.standardized_alive_cells_plot.legend()

        # Mutation Rate
        self.mutation_rate_plot.clear()
        self.mutation_rate_plot.plot(generations, self.mutation_rate_history, label='Mutation Rate', color='orange')
        self.mutation_rate_plot.set_title("Mutation Rate over Generations")
        self.mutation_rate_plot.set_xlabel("Generation")
        self.mutation_rate_plot.set_ylabel("Mutation Rate")
        self.mutation_rate_plot.legend()

        self.stats_fig.tight_layout()

    def run(self):
        """
        Opens the interactive matplotlib windows. The user can close them to stop the program.
        """
        logging.info("Running interactive simulation.")
        plt.show()

# --- END OF InteractiveSimulation.py ---


# --- START OF main.py ---

"""
main.py
-------

Defines the main entry point of the program. It sets up logging, runs the GeneticAlgorithm
with specified parameters, and then launches the InteractiveSimulation to visualize results.
"""

import logging
from InteractiveSimulation import InteractiveSimulation
from GeneticAlgorithm import GeneticAlgorithm

# Configure logging to append to a log file
logging.basicConfig(filename="simulation.log",
                    filemode='a',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def main(grid_size=10,
         population_size=50,
         generations=200,
         initial_mutation_rate=1.0,
         mutation_rate_lower_limit=0.2,
         alive_cells_weight=0.12,
         lifespan_weight=200.0,
         alive_growth_weight=0.1,
         stableness_weight=0.01,
         alive_cells_per_block=5,
         alive_blocks=1,
         predefined_configurations=None):
    """
    Main function that drives the process:
    1. Instantiates the GeneticAlgorithm with the given parameters.
    2. Runs the GA to completion, returning the top 5 best configurations.
    3. Passes those configurations to InteractiveSimulation for visualization and analysis.

    Args:
        grid_size (int): NxN dimension of the grid.
        population_size (int): Number of individuals in each GA generation.
        generations (int): Number of generations to run in the GA.
        initial_mutation_rate (float): Probability of mutating a cell at the start.
        mutation_rate_lower_limit (float): The minimum bound on mutation rate.
        alive_cells_weight (float): Fitness weight for maximum alive cells.
        lifespan_weight (float): Fitness weight for lifespan.
        alive_growth_weight (float): Fitness weight for growth ratio.
        stableness_weight (float): Fitness weight for how quickly a pattern stabilizes or not.
        alive_cells_per_block (int): In random init, how many living cells per block?
        alive_blocks (int): In random init, how many blocks receive living cells?
        predefined_configurations (None or iterable): If you want to pass known patterns.
    """
    logging.info(f"Starting run with parameters: "
                 f"grid_size={grid_size}, "
                 f"population_size={population_size}, "
                 f"generations={generations}, "
                 f"initial_mutation_rate={initial_mutation_rate}, "
                 f"alive_cells_weight={alive_cells_weight}, "
                 f"mutation_rate_lower_limit={mutation_rate_lower_limit}, "
                 f"lifespan_weight={lifespan_weight}, "
                 f"alive_growth_weight={alive_growth_weight}, "
                 f"stableness_weight={stableness_weight}, "
                 f"alive_cells_per_block={alive_cells_per_block}, "
                 f"alive_blocks={alive_blocks}, "
                 f"predefined_configurations={predefined_configurations}")

    algorithm = GeneticAlgorithm(grid_size,
                                 population_size,
                                 generations,
                                 initial_mutation_rate,
                                 mutation_rate_lower_limit,
                                 alive_cells_weight,
                                 lifespan_weight,
                                 alive_growth_weight,
                                 stableness_weight,
                                 alive_cells_per_block=alive_cells_per_block,
                                 alive_blocks=alive_blocks,
                                 predefined_configurations=predefined_configurations)

    best_configs = algorithm.run()

    # Launch interactive simulation with the best configurations
    simulation = InteractiveSimulation(best_configs,
                                       algorithm.best_histories,
                                       grid_size,
                                       generations_cache=algorithm.generations_cache,
                                       mutation_rate_history=algorithm.mutation_rate_history)
    simulation.run()


def get_user_param(prompt: str, default_value: str) -> str:
    """
    Prompt the user for a parameter with a default fallback.
    If the user just presses Enter, the default is returned.

    Args:
        prompt (str): The text shown to the user.
        default_value (str): The default string if the user does not provide input.

    Returns:
        str: The user-entered string or the default if empty.
    """
    user_input = input(f"{prompt} [{default_value}]: ").strip()
    return user_input if user_input else default_value


def run_main_interactively():
    """
    Interactive function that asks the user whether to use all default parameters or
    to input custom values for each parameter individually.
    """
    use_defaults = input("Use default values for ALL parameters? (y/N): ").lower()
    if use_defaults.startswith('y'):
        main()
    else:
        grid_size = int(get_user_param("Enter grid_size", "10"))
        population_size = int(get_user_param("Enter population_size", "50"))
        generations = int(get_user_param("Enter generations", "200"))
        initial_mutation_rate = float(get_user_param("Enter initial_mutation_rate", "1.0"))
        mutation_rate_lower_limit = float(get_user_param("Enter mutation_rate_lower_limit", "0.2"))
        alive_cells_weight = float(get_user_param("Enter alive_cells_weight", "0.12"))
        lifespan_weight = float(get_user_param("Enter lifespan_weight", "200.0"))
        alive_growth_weight = float(get_user_param("Enter alive_growth_weight", "0.1"))
        stableness_weight = float(get_user_param("Enter stableness_weight", "0.01"))
        alive_cells_per_block = int(get_user_param("Enter alive_cells_per_block", "5"))
        alive_blocks = int(get_user_param("Enter alive_blocks", "1"))

        main(grid_size=grid_size,
             population_size=population_size,
             generations=generations,
             initial_mutation_rate=initial_mutation_rate,
             mutation_rate_lower_limit=mutation_rate_lower_limit,
             alive_cells_weight=alive_cells_weight,
             lifespan_weight=lifespan_weight,
             alive_growth_weight=alive_growth_weight,
             stableness_weight=stableness_weight,
             alive_cells_per_block=alive_cells_per_block,
             alive_blocks=alive_blocks,
             predefined_configurations=None)


if __name__ == '__main__':
    run_main_interactively()

# --- END OF main.py ---

