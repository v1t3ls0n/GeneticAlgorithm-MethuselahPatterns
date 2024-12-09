# GeneticAlgorithm.py

import numpy as np
import random
import logging
from Configuration import Configuration


class GeneticAlgorithm:
    def __init__(self, config):
        """
        Initialize the Genetic Algorithm.
        :param config: Config instance for shared parameters.
        """
        self.config = config
        self.population = self.initialize_population()

    def initialize_population(self):
        """Create the initial population with random configurations."""
        population = []
        for i in range(self.config.population_size):
            initial_state = np.zeros((self.config.grid_size, self.config.grid_size), dtype=int)

            # Randomly place between 3 to 7 live cells for diversity
            live_cells = 0
            num_live_cells = random.randint(3, 7)
            while live_cells < num_live_cells:
                x, y = random.randint(0, self.config.grid_size - 1), random.randint(0, self.config.grid_size - 1)
                if initial_state[x, y] == 0:
                    initial_state[x, y] = 1
                    live_cells += 1

            config_name = f"Config_{i + 1}"
            config = Configuration(name=config_name, initial_state=initial_state, config=self.config)
            population.append(config)
            logging.debug(f"Initialized {config_name} with initial_state:\n{initial_state}")

        logging.debug(f"Population initialized with {len(population)} configurations.")
        return population

    def evolve(self):
        """
        Run the genetic algorithm across generations.
        :return: The best configuration after evolution.
        """
        for generation in range(self.config.generations):
            logging.info(f"Generation {generation + 1}: Running evolution step...")
            logging.debug(f"Population size: {len(self.population)}")

            # Evaluate fitness and analyze all configurations
            for config in self.population:
                if not config.type:
                    config.analyze()
                    logging.debug(f"Analyzed {config.name}: {config.summary()}")

            # Update metrics and save top 5 configurations
            self.update_metrics()
            self.save_top_5_configs()

            # Log generation stats
            if self.config.metrics["best_fitness"]:
                best_fitness = max(self.config.metrics["best_fitness"])
                avg_fitness = self.config.metrics["avg_fitness"][-1]
                fitness_variance = self.config.metrics["fitness_variance"][-1]
                logging.info(f"Generation {generation + 1}: Best Fitness = {best_fitness}, Avg Fitness = {avg_fitness}, Variance = {fitness_variance}")
            else:
                logging.warning("No fitness metrics available.")

            # Create the next generation
            self.population = self.next_generation(generation)
            logging.debug(f"New population size: {len(self.population)}")

        logging.info("Evolution completed.")
        return self.config.top_5_configs[0] if self.config.top_5_configs else None

    def next_generation(self, current_generation):
        """Generate the next population using selection, crossover, mutation, and elitism."""
        new_population = []

        # Elitism: retain the top 2 configurations
        elitism_count = 2
        elites = sorted(self.population, key=self.fitness, reverse=True)[:elitism_count]
        new_population.extend(elites)
        logging.debug(f"Elitism: Retaining {elitism_count} top configurations.")

        # Generate the rest of the population
        while len(new_population) < self.config.population_size:
            parent1, parent2 = self.selection()
            child = self.crossover(parent1, parent2)
            child = self.mutate(child, current_generation)
            new_population.append(child)
            logging.debug(f"Created child {child.name} from parents {parent1.name} & {parent2.name}")

        return new_population

    def selection(self):
        """Select two parents using tournament selection."""
        tournament_size = 5
        selected = random.sample(self.population, min(tournament_size, len(self.population)))
        parent1 = max(selected, key=self.fitness)
        parent2 = max(selected, key=self.fitness)
        return parent1, parent2

    def crossover(self, parent1, parent2):
        """Perform crossover to create a child configuration."""
        grid_size = self.config.grid_size
        crossover_point = random.randint(1, grid_size - 1)
        child_state_top = parent1.initial_state[:crossover_point, :]
        child_state_bottom = parent2.initial_state[crossover_point:, :]
        child_state = np.vstack((child_state_top, child_state_bottom))
        child_name = f"Child_{random.randint(1000,9999)}"
        return Configuration(name=child_name, initial_state=child_state, config=self.config)

    def mutate(self, config, generation):
        """Apply mutation to a configuration with an adaptive mutation rate."""
        # Example: Decrease mutation rate as generations progress
        adaptive_mutation_rate = self.config.mutation_rate * (1 - (generation / self.config.generations))
        if random.random() < adaptive_mutation_rate:
            x, y = random.randint(0, self.config.grid_size - 1), random.randint(0, self.config.grid_size - 1)
            config.initial_state[x, y] = 1 - config.initial_state[x, y]  # Flip the cell value
            logging.debug(f"Mutated {config.name} at position ({x}, {y}) with mutation rate {adaptive_mutation_rate:.4f}")
        return config

    def fitness(self, config):
        """
        Evaluate the fitness of a configuration based on stabilization time and final size.
        :param config: A Configuration instance.
        :return: The fitness score.
        """
        if config.stabilization_time is None or config.final_size is None:
            config.analyze()
            logging.debug(f"Fitness recalculated for {config.name}")

        stabilization_time = config.stabilization_time or 0
        final_size = config.final_size or 0

        # Apply a penalty if final_size is 0 to discourage extinction
        penalty = 1000 if final_size == 0 else 0

        # Adjust fitness function to balance stabilization_time and final_size
        fitness_score = self.config.alpha * stabilization_time + self.config.beta * final_size - penalty

        logging.debug(f"Fitness for {config.name}: {fitness_score} (Penalty: {penalty})")
        return fitness_score

    def update_metrics(self):
        """Track metrics like best fitness, average fitness, and variance across generations."""
        fitness_scores = [self.fitness(config) for config in self.population]
        self.config.update_metrics("best_fitness", max(fitness_scores))
        self.config.update_metrics("avg_fitness", np.mean(fitness_scores))
        self.config.update_metrics("fitness_variance", np.var(fitness_scores))

    def save_top_5_configs(self):
        """Save the top 5 configurations based on fitness."""
        sorted_population = sorted(self.population, key=self.fitness, reverse=True)
        self.config.top_5_configs = sorted_population[:5]  # Assign to Config's top_5_configs
        logging.debug("Top 5 Configurations:")
        for config in self.config.top_5_configs:
            logging.debug(config.summary())
