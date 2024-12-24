import random
import logging
import matplotlib.pyplot as plt

# הגדרת לוגינג
logging.basicConfig(
                    filename="simulation.log",
                    filemode='w',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class InteractiveSimulation:
    def __init__(self, configurations, histories, grid_size):
        self.configurations = configurations
        self.histories = histories
        self.grid_size = grid_size
        self.current_config_index = 0
        self.current_generation = 0
        self.game = GameOfLife(
            grid_size, self.configurations[self.current_config_index])
        self.fig, self.ax = plt.subplots()

        grid = [
            self.game.grid[i * grid_size:(i + 1) * grid_size] for i in range(grid_size)]
        self.img = self.ax.imshow(grid, cmap="binary")
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.update_plot()

    def on_key(self, event):
        if event.key == 'right':
            self.next_configuration()
        elif event.key == 'left':
            self.previous_configuration()
        elif event.key == 'up':
            self.next_generation()
        elif event.key == 'down':
            self.previous_generation()

    def next_configuration(self):
        self.current_config_index = (
            self.current_config_index + 1) % len(self.configurations)
        self.current_generation = 0
        self.game = GameOfLife(
            self.grid_size, self.configurations[self.current_config_index])
        self.update_plot()

    def previous_configuration(self):
        self.current_config_index = (
            self.current_config_index - 1) % len(self.configurations)
        self.current_generation = 0
        self.game = GameOfLife(
            self.grid_size, self.configurations[self.current_config_index])
        self.update_plot()

    def next_generation(self):
        if self.current_generation + 1 < len(self.histories[self.current_config_index]):
            self.current_generation += 1
            self.game.grid = self.histories[self.current_config_index][self.current_generation]
            self.update_plot()

    def previous_generation(self):
        if self.current_generation > 0:
            self.current_generation -= 1
            self.game.grid = self.histories[self.current_config_index][self.current_generation]
            self.update_plot()

    def update_plot(self):
        grid = [self.game.grid[i *
                               self.grid_size:(i + 1) * self.grid_size] for i in range(self.grid_size)]
        self.img.set_data(grid)
        self.ax.set_title(f"""Configuration {
                          self.current_config_index + 1}, Generation {self.current_generation}""")
        self.fig.canvas.draw()

    def run(self):
        plt.show()


class GameOfLife:
    def __init__(self, grid_size, initial_state=None):
        self.grid_size = grid_size
        grid = [
            0] * (grid_size * grid_size) if initial_state is None else list(initial_state)
        self.grid = grid
        self.history = [tuple(grid)]  # לשמור את כל המצבים
        self.stable_count = 0  # מונה את מספר הדורות שבהם המצב נשאר יציב

    def step(self):
        new_grid = [0] * (self.grid_size * self.grid_size)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                index = x * self.grid_size + y
                alive_neighbors = self.count_alive_neighbors(x, y)
                if self.grid[index] == 1:
                    if alive_neighbors in [2, 3]:
                        new_grid[index] = 1
                else:
                    if alive_neighbors == 3:
                        new_grid[index] = 1
        self.grid = new_grid
        self.history.append(tuple(new_grid))  # שמירת המצב הנוכחי כ-tuple

    def count_alive_neighbors(self, x, y):
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
        alive_count = sum(self.grid)
        return alive_count

    def is_static_or_repeating(self):
        if self.get_alive_cells_count() == 0:
            return True
        current_state = tuple(self.grid)
        if current_state in self.history[:-1]:
            return True
        return False

    def get_lifespan(self, max_generations):
        for generation in range(max_generations):
            if self.is_static_or_repeating():
                return generation
            self.step()
        return max_generations


class GeneticAlgorithm:
    def __init__(self, grid_size, population_size, generations, mutation_rate, initial_alive_cells,
                 alive_cells_weight, max_lifespan, lifespan_weight, alive_growth_weight,
                 num_parts=4, cells_per_part=1, parts_with_cells=2, predefined_configurations=None, min_fitness_score=1):
        self.grid_size = grid_size
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.initial_alive_cells = initial_alive_cells
        self.alive_cells_weight = alive_cells_weight
        self.max_lifespan = max_lifespan
        self.lifespan_weight = lifespan_weight
        self.alive_growth_weight = alive_growth_weight
        self.fitness_cache = {}
        self.predefined_configurations = predefined_configurations
        self.num_parts = num_parts  # מספר החלקים שהגריד יתחלק אליהם
        self.cells_per_part = cells_per_part  # מספר התאים החיים שנרצה לשים בכל חלק
        # מספר החלקים שבהם נרצה לשים תאים חיים
        self.parts_with_cells = parts_with_cells
        self.min_fitness_score = min_fitness_score
        self.best_histories = []
        self.population = self.initialize_population()

    def fitness(self, configuration):
        configuration_tuple = tuple(configuration)

        if configuration_tuple not in self.fitness_cache:
            alive_cells_count = sum(configuration_tuple)
            if alive_cells_count != self.initial_alive_cells:
                self.fitness_cache[configuration_tuple] = self.min_fitness_score - 1
                return self.min_fitness_score - 1

        game = GameOfLife(self.grid_size, configuration_tuple)

        lifespan = game.get_lifespan(self.max_lifespan)

        alive_counts = [game.get_alive_cells_count()]
        for _ in range(lifespan):
            game.step()
            alive_counts.append(game.get_alive_cells_count())

        total_alive_cells = sum(alive_counts)
        alive_growth = max(alive_counts) - self.initial_alive_cells

        fitness_score = (lifespan * self.lifespan_weight +
                         total_alive_cells / self.alive_cells_weight +
                         alive_growth * self.alive_growth_weight)

        fitness_score = max(fitness_score, self.min_fitness_score)
        self.fitness_cache[configuration_tuple] = fitness_score
        return fitness_score

    def mutate(self, configuration):
        N = self.grid_size
        mid = N // 2

        # המרה מווקטור למטריצה דו-ממדית
        matrix = [configuration[i * N:(i + 1) * N] for i in range(N)]

        # חילוק למקטעים, התייחסות לשארית במקרה של גודל אי-זוגי
        top_left = [row[:mid] for row in matrix[:mid]]  # רבע שמאלי עליון
        top_right = [row[mid:] for row in matrix[:mid]]  # רבע ימין עליון
        bottom_left = [row[:mid] for row in matrix[mid:]]  # רבע שמאלי תחתון
        bottom_right = [row[mid:] for row in matrix[mid:]]  # רבע ימין תחתון

        # אם יש שארית, נוסיף את השורות החסרות
        if N % 2 != 0:
            extra_row = matrix[N-1]
            top_left.append(extra_row[:mid])
            top_right.append(extra_row[mid:])
            bottom_left.append(extra_row[:mid])
            bottom_right.append(extra_row[mid:])

        # אוסף את כל הרבעים ברשימה
        quarters = [top_left, top_right, bottom_left, bottom_right]

        # ביצוע permute רנדומלי של הרבעים
        random.shuffle(quarters)

        # שילוב מחדש של הרבעים לגריד
        new_matrix = []
        for i in range(mid):
            # חיבור של החלק העליון
            new_matrix.append(quarters[0][i] + quarters[1][i])
        for i in range(mid):
            # חיבור של החלק התחתון
            new_matrix.append(quarters[2][i] + quarters[3][i])

        # אם יש שארית, נוסיף אותה למטריצה החדשה
        if N % 2 != 0:
            new_matrix[-1] += extra_row

        # המרה חזרה לווקטור
        new_configuration = [cell for row in new_matrix for cell in row]

        return tuple(new_configuration)

    def select_parents(self):
        fitness_scores = [self.fitness(config) for config in self.population]
        total_fitness = sum(fitness_scores)
        if total_fitness == self.min_fitness_score:
            return random.choices(self.population, k=2)
        probabilities = [score / total_fitness for score in fitness_scores]
        parents = random.choices(self.population, weights=probabilities, k=2)
        return parents

    def crossover(self, parent1, parent2):
        # גודל הגריד
        N = self.grid_size
        mid = N // 2  # חישוב חצי מהגודל

        # חילוק הווקטור של ההורים לרבעים
        father_top_left = [parent1[i * N + j]
                           # הרבע השמאלי העליון
                           for i in range(mid) for j in range(mid)]
        father_bottom_right = [parent1[(i + mid) * N + (j + mid)]
                               # הרבע הימני התחתון
                               for i in range(mid) for j in range(mid)]

        # מהאמא (הרבע הימני העליון והרבע השמאלי התחתון)
        # הרבע הימני העליון
        mother_top_right = [parent2[i * N + (j + mid)]
                            for i in range(mid) for j in range(mid)]
        mother_bottom_left = [parent2[(i + mid) * N + j] for i in range(mid)
                              for j in range(mid)]  # הרבע השמאלי התחתון

        # שילוב הרבעים להילד:
        child_top_left = father_top_left
        child_top_right = mother_top_right
        child_bottom_left = mother_bottom_left
        child_bottom_right = father_bottom_right

        # יצירת הילד כווקטור אחד
        child = []
        # חיבור של החלק העליון
        child.extend(child_top_left)
        child.extend(child_top_right)
        # חיבור של החלק התחתון
        child.extend(child_bottom_left)
        child.extend(child_bottom_right)

        # החזרת הילד כווקטור חד-ממדי
        return tuple(child)

    def initialize_population(self):
        if self.predefined_configurations is not None:
            population = [self.expand_configuration(
                config) for config in self.predefined_configurations]
            while len(population) < self.population_size:
                population.append(self.random_configuration())
            return population
        return [self.random_configuration() for _ in range(self.population_size)]

    def expand_configuration(self, configuration):
        """ להרחיב את הקונפיגורציות כך שיתחילו בפינה השמאלית העליונה """
        rows, cols = len(configuration), len(configuration[0])
        expanded = [0] * (self.grid_size * self.grid_size)
        for i in range(rows):
            for j in range(cols):
                expanded[i * self.grid_size + j] = configuration[i][j]
        return tuple(expanded)

    def random_configuration(self):
        while True:
            configuration = [0] * (self.grid_size * self.grid_size)
            all_cells = list(range(self.grid_size * self.grid_size))

            # חישוב חלוקה ל- num_parts חלקים
            part_size = self.grid_size * self.grid_size // self.num_parts
            parts_with_cells_indices = random.sample(
                range(self.num_parts), self.parts_with_cells)

            # נבחר תאים לכל חלק שנבחר
            for part_index in parts_with_cells_indices:
                start_idx = part_index * part_size
                end_idx = start_idx + part_size
                chosen_cells = random.sample(
                    all_cells[start_idx:end_idx], self.cells_per_part)
                for cell in chosen_cells:
                    configuration[cell] = 1

            if sum(configuration) == self.initial_alive_cells:
                break

        return tuple(configuration)

    def run(self):
        all_fitness_scores = []
        for generation in range(self.generations):
            new_population = []
            for i in range(self.population_size):
                parent1, parent2 = self.select_parents()
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)

            self.population = new_population
            self.fitness_cache.clear()

            fitness_scores = [self.fitness(config) for config in self.population]
            all_fitness_scores.append(fitness_scores)

            # לוגינג לדור הנוכחי
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            std_fitness = (sum([(score - avg_fitness) ** 2 for score in fitness_scores]) / len(fitness_scores)) ** 0.5
            logging.info(f"""Generation {generation + 1}: Avg Fitness: {avg_fitness}, Std Dev: {std_fitness}""")

            # מציאת הקונפיגורציה הטובה ביותר בדור הנוכחי
            best_fitness_score = max(fitness_scores)
            best_config_index = fitness_scores.index(best_fitness_score)
            best_config = self.population[best_config_index]

            # יצירת אובייקט GameOfLife עם הקונפיגורציה הטובה ביותר
            game = GameOfLife(self.grid_size, best_config)
            lifespan = game.get_lifespan(self.max_lifespan)
            total_alive_cells = sum([game.get_alive_cells_count() for _ in range(lifespan)])
            alive_growth = max([game.get_alive_cells_count() for _ in range(lifespan)]) - self.initial_alive_cells

            # לוגינג של הקונפיגורציה הטובה ביותר בדור הנוכחי
            logging.info(f"""  Best Configuration in Generation {generation + 1}:""")
            logging.info(f"""    Fitness Score: {best_fitness_score}""")
            logging.info(f"""    Lifespan: {lifespan}""")
            logging.info(f"""    Total Alive Cells: {total_alive_cells}""")
            logging.info(f"""    Alive Growth: {alive_growth}""")

        fitness_scores = [(config, self.fitness(config)) for config in self.population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        best_configs = [config for config, _ in fitness_scores[:5]]

        self.best_histories = []  # לאתחל את ההיסטוריות עבור הקונפיגורציות הטובות ביותר
        for config in best_configs:
            game = GameOfLife(self.grid_size, config)
            lifespan = game.get_lifespan(self.max_lifespan)
            history = game.history  # שמירה של ההיסטוריה של כל הדורות
            self.best_histories.append(history)

            # לוגינג של 5 הקונפיגורציות הטובות ביותר
            total_alive_cells = sum([game.get_alive_cells_count() for _ in range(lifespan)])
            alive_growth = max([game.get_alive_cells_count() for _ in range(lifespan)]) - self.initial_alive_cells
            logging.info(f"""Top {config} Configuration:""")
            logging.info(f"""  Fitness Score: {self.fitness(config)}""")
            logging.info(f"""  Lifespan: {lifespan}""")
            logging.info(f"""  Total Alive Cells: {total_alive_cells}""")
            logging.info(f"""  Alive Growth: {alive_growth}""")

        return best_configs



def main(grid_size, population_size, generations, mutation_rate, initial_alive_cells,
         alive_cells_weight, max_lifespan, lifespan_weight, alive_growth_weight, predefined_configurations=None):
    # חישוב פרמטרים מתוך grid_size ו-initial_alive_cells
    # נניח 10x10 חלקים בתוך גריד בגודל grid_size
    num_parts = (grid_size // 10) ** 2
    # נחשב כמה תאים חיים יהיו בכל חלק
    cells_per_part = initial_alive_cells // num_parts
    # נוודא שמספר החלקים לא יעלה על מספר התאים החיים
    parts_with_cells = min(num_parts, (initial_alive_cells // cells_per_part))

    # יצירת מופע של אלגוריתם גנטי עם כל הפרמטרים
    algorithm = GeneticAlgorithm(
        grid_size, population_size, generations, mutation_rate, initial_alive_cells,
        alive_cells_weight, max_lifespan, lifespan_weight, alive_growth_weight,
        num_parts=num_parts, cells_per_part=cells_per_part, parts_with_cells=parts_with_cells,
        predefined_configurations=predefined_configurations
    )

    best_configs = algorithm.run()

    simulation = InteractiveSimulation(
        best_configs, algorithm.best_histories, grid_size)
    simulation.run()


# קריאה לפונקציה עם חישוב אוטומטי של פרמטרים
main(
    grid_size=20, population_size=20, generations=20, mutation_rate=0.02,
    initial_alive_cells=5, alive_cells_weight=50, max_lifespan=5000,
    lifespan_weight=10, alive_growth_weight=5
)
