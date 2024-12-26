import logging
from InteractiveSimulation import InteractiveSimulation
from GeneticAlgorithm import GeneticAlgorithm
# Set up logging to append to the same file for each run of the program
logging.basicConfig(filename="simulation.log",
                    filemode='a',  # Use 'a' to append to the file
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


def get_user_param(prompt: str, default_value: str) -> str:
    """
    מציג למשתמש prompt, ומחזיר את הטקסט שהמשתמש הקליד.
    אם המשתמש לוחץ Enter בלי להקליד כלום - מחזירים את default_value.
    """
    user_input = input(f"{prompt} [{default_value}]: ").strip()
    return user_input if user_input else default_value


def run_main_interactively():
    """
    פונקציית מעטפת שמבקשת מהמשתמש להזין ערכים או לבחור ברירת מחדל לכולם.
    """
    use_defaults = input("Use default values for ALL parameters? (y/N): ").lower()
    if use_defaults.startswith('y'):
        # מפעילים את main עם ערכי ברירת המחדל
        main()
    else:
        # מאפשרים להזין באופן אינטראקטיבי לכל פרמטר
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
    # מפעילים את הפונקציה שתקרא ל-main עם הפרמטרים שהמשתמש בוחר
    run_main_interactively()