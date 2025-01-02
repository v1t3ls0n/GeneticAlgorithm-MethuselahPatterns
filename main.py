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
from GeneticAlgorithm import GeneticAlgorithm
from InteractiveSimulation import InteractiveSimulation

# Configure logging to append to a log file with triple-quoted messages
logging.basicConfig(
    filename="simulation.log",
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class Configuration:
    def __init__(
        self,
        grid_size=40,
        population_size=32,
        generations=100,
        mutation_rate_upper_limit=0.3,
        mutation_rate_lower_limit=0.1,
        alive_cells_weight=0.12,
        lifespan_weight=100.0,
        alive_growth_weight=2,
        initial_living_cells_count_penalty_weight=100,
        predefined_configurations=None,
        boundary_type="wrap"
    ):
        self.grid_size = grid_size
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate_upper_limit = mutation_rate_upper_limit
        self.mutation_rate_lower_limit = mutation_rate_lower_limit
        self.alive_cells_weight = alive_cells_weight
        self.lifespan_weight = lifespan_weight
        self.alive_growth_weight = alive_growth_weight
        self.initial_living_cells_count_penalty_weight = initial_living_cells_count_penalty_weight
        self.predefined_configurations = predefined_configurations
        self.boundary_type = boundary_type

    def as_dict(self):
        """Convert configuration attributes to a dictionary."""
        return vars(self)


default_config = Configuration()
# Fetch default values as a dictionary
default_params = default_config.as_dict()


def main(grid_size,
         population_size,
         generations,
         mutation_rate_upper_limit,
         alive_cells_weight,
         mutation_rate_lower_limit,
         lifespan_weight,
         alive_growth_weight,
         initial_living_cells_count_penalty_weight,
         predefined_configurations,
         boundary_type):
    """
    Main function that drives the process:
    1. Instantiates the GeneticAlgorithm with the given parameters.
    2. Runs the GA to completion, returning the top configurations.
    3. Passes those configurations to InteractiveSimulation for visualization and analysis.

    Args:
        grid_size (int): NxN dimension of the grid.
        population_size (int): Number of individuals in each GA generation.
        generations (int): Number of generations to run in the GA.
        mutation_rate_upper_limit (float): Probability of mutating a cell at the start.
        mutation_rate_lower_limit (float): The minimum bound on mutation rate.
        alive_cells_weight (float): Fitness weight for maximum alive cells.
        lifespan_weight (float): Fitness weight for lifespan.
        alive_growth_weight (float): Fitness weight for growth ratio.
        initial_living_cells_count_penalty_weight (float): Weight for penalizing large initial patterns.
        predefined_configurations (None or iterable): Optional, known patterns for initialization.
        boundary_type (str): Boundary type for the Game of Life grid ("wrap" or "fixed").
    """
    logging.info(f"""Starting run with parameters:
    grid_size={grid_size},
    population_size={population_size},
    generations={generations},
    mutation_rate_upper_limit={mutation_rate_upper_limit},
    alive_cells_weight={alive_cells_weight},
    mutation_rate_lower_limit={mutation_rate_lower_limit},
    lifespan_weight={lifespan_weight},
    alive_growth_weight={alive_growth_weight},
    initial_living_cells_count_penalty_weight={initial_living_cells_count_penalty_weight},
    predefined_configurations={predefined_configurations},
    boundary_type={boundary_type}""")

    # Validate boundary type
    if boundary_type not in {"wrap", "fill"}:
        raise ValueError(f"Invalid boundary_type: {boundary_type}. Must be 'wrap' or 'fill'.")

    # Instantiate the GeneticAlgorithm
    algorithm = GeneticAlgorithm(
        grid_size=grid_size,
        population_size=population_size,
        generations=generations,
        mutation_rate_upper_limit=mutation_rate_upper_limit,
        mutation_rate_lower_limit=mutation_rate_lower_limit,
        alive_cells_weight=alive_cells_weight,
        lifespan_weight=lifespan_weight,
        alive_growth_weight=alive_growth_weight,
        initial_living_cells_count_penalty_weight=initial_living_cells_count_penalty_weight,
        predefined_configurations=predefined_configurations,
        boundary_type=boundary_type
    )

    # Run the algorithm and retrieve top configurations and their parameters
    results, initial_configurations_start_index = algorithm.run()

    # Extract only the 'config' from the results for visualization
    run_params = {
        "grid_size": grid_size,
        "population_size": population_size,
        "generations": generations,
        "mutation_rate_upper_limit": mutation_rate_upper_limit,
        "mutation_rate_lower_limit": mutation_rate_lower_limit,
        "alive_cells_weight": alive_cells_weight,
        "lifespan_weight": lifespan_weight,
        "alive_growth_weight": alive_growth_weight,
        "initial_living_cells_count_penalty_weight": initial_living_cells_count_penalty_weight,
        "predefined_configurations": predefined_configurations,
        "boundary_type": boundary_type
    }

    # Launch interactive simulation with the best configurations
    simulation = InteractiveSimulation(
        configurations=results,
        initial_configurations_start_index=initial_configurations_start_index,
        grid_size=grid_size,
        generations_statistics=algorithm.generations_statistics,
        mutation_rate_history=algorithm.mutation_rate_history,
        diversity_history=algorithm.diversity_history,
        run_params=run_params
    )
    simulation.run()

def get_user_param_with_explanation(prompt: str, default_value: str, explanation: str) -> str:
    """
    Prompt the user for a parameter with a default fallback and display an explanation.

    Args:
        prompt (str): The text shown to the user.
        default_value (str): The default string if the user does not provide input.
        explanation (str): Explanation of the parameter and its options.

    Returns:
        str: The user-entered string or the default if empty.
    """
    print(explanation)  # Display the explanation before prompting
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
    # Create a default configuration instance
    default_config = Configuration()
    # Fetch default values as a dictionary
    default_params = default_config.as_dict()

    # Ask if the user wants to use all defaults
    use_defaults = input(
        "Use default values for ALL parameters? (y/N): ").strip().lower()
    if use_defaults.startswith('y') or use_defaults == "":
        main(**default_params)
        return

    else:
        # Interactively get parameters, falling back on the defaults dynamically
        updated_params = {}
        for key, value in default_params.items():
            if key == "boundary_type":
                explanation = (
                    "The boundary_type parameter determines how the edges of the grid are treated:\n"
                    " - 'wrap': The grid edges wrap around, so the top connects to the bottom and the left to the right.\n"
                    " - 'fill': The grid edges are static and treated as always dead cells."
                )
                user_input = get_user_param_with_explanation(
                    f"Enter boundary type ('wrap' or 'fill')", value, explanation)
                if user_input not in {"wrap", "fill"}:
                    print("Invalid boundary type. Defaulting to 'wrap'.")
                    updated_params[key] = "wrap"
                else:
                    updated_params[key] = user_input
            else:
                user_input = input(f"{key} [{value}]: ").strip()
                if user_input == "":
                    updated_params[key] = value  # Use the default
                else:
                    try:
                        # Attempt to cast the user input to the type of the default value
                        updated_params[key] = type(value)(user_input)
                    except ValueError:
                        logging.warning(f"Invalid input for {key}. Using default value: {value}")
                        updated_params[key] = value

        # Pass the updated parameters to main
        main(**updated_params)


if __name__ == '__main__':
    run_main_interactively()
