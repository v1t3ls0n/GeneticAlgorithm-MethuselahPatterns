import logging
from InteractiveSimulation import InteractiveSimulation
from GeneticAlgorithm import GeneticAlgorithm
# Set up logging to append to the same file for each run of the program
logging.basicConfig(filename="simulation.log",
                    filemode='a',  # Use 'a' to append to the file
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def main(grid_size, population_size, generations, initial_mutation_rate, alive_cells_weight, mutation_rate_lower_limit,
         lifespan_weight, alive_growth_weight, stableness_weight, alive_cells_per_block, alive_blocks, predefined_configurations=None):
    logging.info(f"""Starting run with parameters:
                 grid_size={grid_size}, 
                 population_size={population_size}, 
                 generations={generations}, 
                 initial_mutation_rate={initial_mutation_rate}, 
                 alive_cells_weight={alive_cells_weight}, 
                 mutation_rate_lower_limit={mutation_rate_lower_limit},
                 lifespan_weight={lifespan_weight}, 
                 alive_growth_weight={alive_growth_weight}, 
                 stableness_weight={stableness_weight},
                 alive_cells_per_block={alive_cells_per_block}, 
                 alive_blocks={alive_blocks}, 
                 predefined_configurations={predefined_configurations}""")

    algorithm = GeneticAlgorithm(
        grid_size, population_size, generations, initial_mutation_rate, mutation_rate_lower_limit,
        alive_cells_weight, lifespan_weight, alive_growth_weight, stableness_weight,
        alive_cells_per_block=alive_cells_per_block, alive_blocks=alive_blocks,
        predefined_configurations=predefined_configurations
    )

    best_configs = algorithm.run()

    # Pass the mutation rate history to the InteractiveSimulation
    simulation = InteractiveSimulation(
        best_configs, algorithm.best_histories, grid_size, generations_cache=algorithm.generations_cache, 
        mutation_rate_history=algorithm.mutation_rate_history)
    simulation.run()

# Example call to main function
main(grid_size=5, population_size=50, generations=200, initial_mutation_rate=1,mutation_rate_lower_limit = 0.25,
     alive_cells_weight=0.12, lifespan_weight=200, alive_growth_weight=0.03, stableness_weight = 0.01,

     alive_cells_per_block=5, alive_blocks=1)
