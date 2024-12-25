import logging
from InteractiveSimulation import InteractiveSimulation
from GeneticAlgorithm import GeneticAlgorithm
# Set up logging to append to the same file for each run of the program
logging.basicConfig(filename="simulation.log",
                    filemode='w',  # Use 'a' to append to the file
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def main(grid_size, population_size, generations, mutation_rate, alive_cells_weight,
         lifespan_weight, alive_growth_weight, cells_per_part, parts_with_cells, predefined_configurations=None):
    logging.info(f"""Starting run with parameters: grid_size={grid_size}, population_size={population_size}, generations={generations}, mutation_rate={mutation_rate}, alive_cells_weight={
                 alive_cells_weight}, lifespan_weight={lifespan_weight}, alive_growth_weight={alive_growth_weight}, cells_per_part={cells_per_part}, parts_with_cells={parts_with_cells}""")

    algorithm = GeneticAlgorithm(
        grid_size, population_size, generations, mutation_rate,
        alive_cells_weight, lifespan_weight, alive_growth_weight,
        cells_per_part=cells_per_part, parts_with_cells=parts_with_cells,
        predefined_configurations=predefined_configurations
    )

    best_configs = algorithm.run()

    simulation = InteractiveSimulation(
        best_configs, algorithm.best_histories, grid_size)
    simulation.run()


# Example call to main function
main(grid_size=4, population_size=20, generations=50, mutation_rate=0.5,
     alive_cells_weight=1.3, lifespan_weight=2, alive_growth_weight=1.1,
     cells_per_part=5, parts_with_cells=1)
