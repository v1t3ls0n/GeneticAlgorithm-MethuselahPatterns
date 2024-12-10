# main.py

import logging
from Config import Config
from GeneticAlgorithm import GeneticAlgorithm
from GUI import GeneticAlgorithmGUI
from Configuration import Configuration
import numpy as np
logging.basicConfig(level=logging.DEBUG,
                        filename="debug.log",
                        filemode='w',  # Overwrite log file on each run
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%H:%M:%S')

def test_analyze():
    initial_state = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 0, 0],
    ])
    config = Config.get_instance()
    config.grid_size = 5
    config.max_generations = 1000
    config.stability_threshold = 5

    config_obj = Configuration(name="TestConfig", initial_state=initial_state, config=config)
    config_obj.analyze()

    print(f"Type: {config_obj.type}")
    print(f"Lifetime: {config_obj.lifetime}")
    print(f"Final Size: {config_obj.final_size}")
    print(f"Size Difference: {config_obj.size_difference}")



def main():


    # Get the singleton Config instance and reset to ensure clean state
    config = Config.get_instance()
    config.reset()

    # Explicitly set configuration parameters
    # try:
    #     config.grid_size = 20  # Set to a manageable size (e.g., 20)
    #     config.max_generations = 100
    #     config.stability_threshold = 5
    #     config.population_size = 50
    #     config.generations = 100  # Number of GA generations
    #     config.mutation_rate = 0.1
    #     config.alpha = 0.5
    #     config.beta = 1.5
    #     config.simulation_delay = 0.1
    #     config.metrics = {
    #             "best_fitness": [],
    #             "avg_fitness": [],
    #             "fitness_variance": []
    #         }
    #     config.top_5_configs = []
    #     config._initialized = True
    #     # Additional parameters can be set here as needed
    # except ValueError as ve:
    #     logging.error(f"Configuration error: {ve}")
    #     print(f"Configuration error: {ve}")
    #     return

    # Initialize the Genetic Algorithm
    ga = GeneticAlgorithm(config)

    # Evolve the population
    best_config = ga.evolve()

    # Log the best configuration
    if best_config:
        logging.info("\nBest Configuration Details:")
        for key, value in best_config.summary().items():
            logging.info(f"{key}: {value}")
    else:
        logging.info("No configurations found.")

    # Launch the GUI
    try:
        gui = GeneticAlgorithmGUI(ga)
        gui.run()
    except ValueError as e:
        logging.error(f"Failed to launch GUI: {e}")
        print(f"Failed to launch GUI: {e}")

if __name__ == "__main__":
    main()
    # test_analyze()
