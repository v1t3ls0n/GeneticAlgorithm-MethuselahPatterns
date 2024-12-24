import numpy as np
import random
import matplotlib.pyplot as plt
import time

class GameOfLife:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        self.history = []  # לשמור את כל המצבים

    def set_initial_state(self, initial_state):
        self.grid = np.array(initial_state, dtype=int)
        self.history = [self.grid.copy()]  # שמירת המצב ההתחלתי

    def step(self):
        new_grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                alive_neighbors = self.count_alive_neighbors(x, y)
                if self.grid[x, y] == 1:
                    if alive_neighbors in [2, 3]:
                        new_grid[x, y] = 1
                else:
                    if alive_neighbors == 3:
                        new_grid[x, y] = 1
        self.grid = new_grid
        self.history.append(self.grid.copy())  # שמירת המצב הנוכחי

    def count_alive_neighbors(self, x, y):
        alive = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                nx, ny = (x + i) % self.grid_size, (y + j) % self.grid_size
                alive += self.grid[nx, ny]
        return alive

    def get_alive_cells_count(self):
        return np.sum(self.grid)

    def is_static(self):
        temp_grid = self.grid.copy()
        self.step()
        is_static = np.array_equal(self.grid, temp_grid)
        self.grid = temp_grid  # Revert to previous state
        return is_static

    def get_lifespan(self, max_generations):
        """בדיקת אורך החיים של קונפיגורציה עד להתייצבות או למוות"""
        for generation in range(max_generations):
            if self.is_static() or self.get_alive_cells_count() == 0:
                return generation
            self.step()
        return max_generations


class GeneticAlgorithm:
    def __init__(self, grid_size, population_size, generations, mutation_rate):
        self.grid_size = grid_size
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = [self.random_configuration() for _ in range(population_size)]

    def random_configuration(self):
        return np.random.choice([0, 1], size=(self.grid_size, self.grid_size), p=[0.9, 0.1])

    def fitness(self, configuration):
        game = GameOfLife(self.grid_size)
        game.set_initial_state(configuration)
        lifespan = game.get_lifespan(1000)
        total_alive_cells = sum(np.sum(state) for state in game.history)
        return lifespan + total_alive_cells / 100  # שילוב של אורך חיים ומורכבות

    def select_parents(self):
        fitness_scores = [self.fitness(config) for config in self.population]
        total_fitness = sum(fitness_scores)
        probabilities = [score / total_fitness for score in fitness_scores]
        parents = random.choices(self.population, weights=probabilities, k=2)
        return parents

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(0, self.grid_size - 1)
        child = np.vstack((parent1[:crossover_point, :], parent2[crossover_point:, :]))
        return child

    def mutate(self, configuration):
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if random.random() < self.mutation_rate:
                    configuration[x, y] = 1 - configuration[x, y]
        return configuration

    def run(self):
        fitness_scores = [(config, self.fitness(config)) for config in self.population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        return [config for config, _ in fitness_scores[:5]]


class Simulation:
    def __init__(self, configurations, grid_size, generations):
        self.configurations = configurations
        self.grid_size = grid_size
        self.generations = generations

    def simulate(self):
        for idx, config in enumerate(self.configurations):
            game = GameOfLife(self.grid_size)
            game.set_initial_state(config)

            plt.figure(figsize=(5, 5))
            for generation in range(self.generations):
                plt.clf()
                plt.imshow(game.grid, cmap="binary")
                plt.title(f"Configuration {idx + 1} - Generation {generation}")
                plt.axis("off")
                plt.pause(0.1)
                game.step()

            plt.show()


# דוגמה לשימוש
grid_size = 10
population_size = 20
generations = 50
mutation_rate = 0.01

algorithm = GeneticAlgorithm(grid_size, population_size, generations, mutation_rate)
best_configs = algorithm.run()

print("חמש התצורות הטובות ביותר שנמצאו:")
for idx, config in enumerate(best_configs):
    print(f"Configuration {idx + 1}:")
    print(config)

simulation = Simulation(best_configs, grid_size, 20)
simulation.simulate()
