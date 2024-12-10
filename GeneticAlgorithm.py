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
        """Create a diverse initial population with a configurable maximum initial size."""
        population = []
        for i in range(self.config.population_size):
            initial_state = np.zeros((self.config.grid_size, self.config.grid_size), dtype=int)

            # Populate the grid with up to max_initial_size live cells
            live_cells_count = 0
            while live_cells_count < self.config.max_initial_size:
                x = random.randint(0, self.config.grid_size - 1)
                y = random.randint(0, self.config.grid_size - 1)
                if initial_state[x, y] == 0:  # Ensure only new live cells are counted
                    initial_state[x, y] = 1
                    live_cells_count += 1

            config_name = f"Config_{i + 1}"
            config = Configuration(name=config_name, initial_state=initial_state, config=self.config)
            population.append(config)
            logging.debug(f"Initialized {config_name} with {live_cells_count} live cells.")
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
                logging.info(f"Generation {generation + 1}: Best Fitness = {best_fitness}, "
                             f"Avg Fitness = {avg_fitness}, Variance = {fitness_variance}")
            else:
                logging.warning("No fitness metrics available.")

            # Create the next generation
            self.population = self.next_generation(generation)
            logging.debug(f"New population size: {len(self.population)}")

        logging.info("Evolution completed.")
        return self.config.top_5_configs[0] if self.config.top_5_configs else None

    def selection(self):
        """Select parents using tournament selection."""
        tournament_size = 5
        selected = random.sample(self.population, tournament_size)
        return max(selected, key=self.fitness), max(selected, key=self.fitness)

    def next_generation(self, generation):
        """Create the next population using selection, crossover, and mutation."""
        new_population = []

        # Elitism: Retain top 2 configurations
        elites = sorted(self.population, key=self.fitness, reverse=True)[:2]
        new_population.extend(elites)

        # Generate new configurations
        while len(new_population) < self.config.population_size:
            parent1, parent2 = self.selection()
            child = self.crossover(parent1, parent2)
            child = self.mutate(child, generation)
            new_population.append(child)

        return new_population

    def crossover(self, parent1, parent2):
        """Create a child configuration by combining two parents."""
        child_state = np.zeros_like(parent1.initial_state)
        for x in range(self.config.grid_size):
            for y in range(self.config.grid_size):
                # Inherit cells from one of the parents
                child_state[x, y] = random.choice([parent1.initial_state[x, y], parent2.initial_state[x, y]])

        child_name = f"Child_{random.randint(1000, 9999)}"
        child_config = Configuration(name=child_name, initial_state=child_state, config=self.config)
        logging.debug(f"Created child {child_name} from parents {parent1.name} & {parent2.name}")
        return child_config

    def mutate(self, config, generation):
        """Apply controlled mutation respecting the max_initial_size."""
        mutation_rate = max(0.01, self.config.mutation_rate * (1 - generation / self.config.generations))
        num_mutations = int(mutation_rate * self.config.grid_size**2)
        for _ in range(num_mutations):
            x, y = random.randint(0, self.config.grid_size - 1), random.randint(0, self.config.grid_size - 1)
            current_live_cells = np.sum(config.initial_state)
            if current_live_cells < self.config.max_initial_size or config.initial_state[x, y] == 1:
                # Flip cell state only if under max_initial_size or flipping to 0
                config.initial_state[x, y] = 1 - config.initial_state[x, y]
        return config

    def fitness(self, config):
        """Evaluate fitness with an emphasis on delayed stabilization and size."""
        if config.stabilization_time is None or config.final_size is None:
            config.analyze()

        lifetime = config.lifetime or 0
        final_size = config.final_size or 0
        penalty = 0

        # Penalize quick stabilization
        if lifetime < self.config.stability_threshold:
            penalty += 500

        # Reward larger final sizes and longer lifetimes
        fitness_score = self.config.alpha * lifetime + self.config.beta * final_size - penalty
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
