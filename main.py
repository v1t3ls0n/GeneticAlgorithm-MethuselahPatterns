"""
main.py
-------

Defines the main entry point of the program. It sets up logging, runs the GeneticAlgorithm
with specified parameters, and then launches the InteractiveSimulation to visualize results.

Modules and Functions:
    - `main`: Drives the Genetic Algorithm process and launches the visualization.
    - `get_user_param`: Simplifies user interaction for parameter input with default values.
    - `run_main_interactively`: Allows users to either use default values or input custom parameters interactively.
    - `if __name__ == '__main__'`: Entry point to launch the program interactively.
"""

import logging
from InteractiveSimulation import InteractiveSimulation
from GeneticAlgorithm import GeneticAlgorithm

# Configure logging to append to a log file
logging.basicConfig(filename="simulation.log",
                    filemode='a',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def main(grid_size=5,
         population_size=100,
         generations=100,
         initial_mutation_rate=0.5,
         alive_cells_weight=0.12,
         mutation_rate_lower_limit=0.1,
         lifespan_weight=200.0,
         alive_growth_weight=0.3,
         stableness_weight=1.0,
         alive_cells_per_block=5,
         alive_blocks=1,
         initial_living_cells_count_weight=10,
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
        alive_cells_per_block (int): Number of living cells per block in random initialization.
        alive_blocks (int): Number of blocks containing living cells in random initialization.
        predefined_configurations (None or iterable): Optional, known patterns for initialization.
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
                 f"initial_living_cells_count_weight={initial_living_cells_count_weight}, "
                 f"predefined_configurations={predefined_configurations}")

    # Instantiate the GeneticAlgorithm
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
                                 initial_living_cells_count_weight=initial_living_cells_count_weight,
                                 predefined_configurations=predefined_configurations)

    # Run the algorithm and retrieve top configurations and their parameters
    best_configs, best_params = algorithm.run()
    run_params = {
        "grid_size": grid_size,
        "population_size": population_size,
        "generations": generations,
        "initial_mutation_rate": initial_mutation_rate,
        "mutation_rate_lower_limit": mutation_rate_lower_limit,
        "alive_cells_weight": alive_cells_weight,
        "lifespan_weight": lifespan_weight,
        "alive_growth_weight": alive_growth_weight,
        "stableness_weight": stableness_weight,
        "alive_cells_per_block": alive_cells_per_block,
        "alive_blocks": alive_blocks,
        "initial_living_cells_count_weight": initial_living_cells_count_weight,
        "predefined_configurations": predefined_configurations
    }

    # Launch interactive simulation with the best configurations
    simulation = InteractiveSimulation(configurations=best_configs,
                                       best_params=best_params,
                                       histories=algorithm.best_histories,
                                       grid_size=grid_size,
                                       generations_cache=algorithm.generations_cache,
                                       mutation_rate_history=algorithm.mutation_rate_history,
                                       run_params=run_params)
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

    Functionality:
        - If the user opts for default values, the `main` function is invoked directly.
        - Otherwise, the user is prompted for each parameter, with defaults available for convenience.
    """
    use_defaults = input("Use default values for ALL parameters? (y/N): ").lower()
    if use_defaults.startswith('y') or use_defaults == "":
        main()
    else:
        grid_size = int(get_user_param("Enter grid_size", "5"))
        population_size = int(get_user_param("Enter population_size", "100"))
        generations = int(get_user_param("Enter generations", "100"))
        initial_mutation_rate = float(get_user_param("Enter initial_mutation_rate", "0.5"))
        mutation_rate_lower_limit = float(get_user_param("Enter mutation_rate_lower_limit", "0.1"))
        alive_cells_weight = float(get_user_param("Enter alive_cells_weight", "0.12"))
        lifespan_weight = float(get_user_param("Enter lifespan_weight", "100.0"))
        alive_growth_weight = float(get_user_param("Enter alive_growth_weight", "0.3"))
        stableness_weight = float(get_user_param("Enter stableness_weight", "1.0"))
        initial_living_cells_count_weight = float(get_user_param("Enter initial_living_cells_count_weight", "10.0"))
        alive_cells_per_block = int(get_user_param("Enter alive_cells_per_block", "5"))
        alive_blocks = int(get_user_param("Enter alive_blocks", "1"))

        # Run main with custom parameters
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
             initial_living_cells_count_weight=initial_living_cells_count_weight,
             predefined_configurations=None)


if __name__ == '__main__':
    run_main_interactively()
