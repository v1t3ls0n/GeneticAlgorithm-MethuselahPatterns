# main.py

import logging
from Config import Config
from GeneticAlgorithm import GeneticAlgorithm
from GUI import GeneticAlgorithmGUI


def main():
    # Configure logging
    logging.basicConfig(level=logging.DEBUG,
                        filename = "app.log",
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%H:%M:%S')

    # Get the singleton Config instance
    config = Config.get_instance()

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


if __name__ == "__main__":
    main()
