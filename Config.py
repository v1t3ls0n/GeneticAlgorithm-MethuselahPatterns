# Config.py

import logging

class Config:
    _instance = None  # Singleton instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):  # Ensure attributes are initialized only once
            # Simulation parameters
            self._grid_size = 20  # Default grid size (20x20)
            self._max_generations = 200
            self._stability_threshold = 5  # Reduced for quicker stabilization detection

            # Genetic algorithm parameters
            self._population_size = 20
            self._generations = 50  # Number of GA generations
            self._mutation_rate = 0.1
            self._alpha = 0.5  # Weight for stabilization time
            self._beta = 1.5   # Weight for final size

            # Visualization parameters
            self._simulation_delay = 0.1  # Delay between generations in seconds

            # Metrics
            self._metrics = {
                "best_fitness": [],
                "avg_fitness": [],
                "fitness_variance": []
            }

            # Top results
            self._top_5_configs = []

            self._initialized = True

    # Getters and setters for parameters
    @property
    def grid_size(self):
        return self._grid_size

    @grid_size.setter
    def grid_size(self, value):
        if isinstance(value, int) and value > 0:
            self._grid_size = value
            logging.debug(f"Set grid_size to {value}")
        else:
            logging.error("Invalid grid_size value. It must be a positive integer.")
            raise ValueError("grid_size must be a positive integer.")

    @property
    def max_generations(self):
        return self._max_generations

    @max_generations.setter
    def max_generations(self, value):
        if isinstance(value, int) and value > 0:
            self._max_generations = value
            logging.debug(f"Set max_generations to {value}")
        else:
            logging.error("Invalid max_generations value. It must be a positive integer.")
            raise ValueError("max_generations must be a positive integer.")

    @property
    def stability_threshold(self):
        return self._stability_threshold

    @stability_threshold.setter
    def stability_threshold(self, value):
        if isinstance(value, int) and value > 0:
            self._stability_threshold = value
            logging.debug(f"Set stability_threshold to {value}")
        else:
            logging.error("Invalid stability_threshold value. It must be a positive integer.")
            raise ValueError("stability_threshold must be a positive integer.")

    @property
    def population_size(self):
        return self._population_size

    @population_size.setter
    def population_size(self, value):
        if isinstance(value, int) and value > 0:
            self._population_size = value
            logging.debug(f"Set population_size to {value}")
        else:
            logging.error("Invalid population_size value. It must be a positive integer.")
            raise ValueError("population_size must be a positive integer.")

    @property
    def generations(self):
        return self._generations

    @generations.setter
    def generations(self, value):
        if isinstance(value, int) and value > 0:
            self._generations = value
            logging.debug(f"Set generations to {value}")
        else:
            logging.error("Invalid generations value. It must be a positive integer.")
            raise ValueError("generations must be a positive integer.")

    @property
    def mutation_rate(self):
        return self._mutation_rate

    @mutation_rate.setter
    def mutation_rate(self, value):
        if isinstance(value, float) and 0 <= value <= 1:
            self._mutation_rate = value
            logging.debug(f"Set mutation_rate to {value}")
        else:
            logging.error("Invalid mutation_rate value. It must be a float between 0 and 1.")
            raise ValueError("mutation_rate must be a float between 0 and 1.")

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if isinstance(value, (int, float)) and value >= 0:
            self._alpha = value
            logging.debug(f"Set alpha to {value}")
        else:
            logging.error("Invalid alpha value. It must be a non-negative number.")
            raise ValueError("alpha must be a non-negative number.")

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        if isinstance(value, (int, float)) and value >= 0:
            self._beta = value
            logging.debug(f"Set beta to {value}")
        else:
            logging.error("Invalid beta value. It must be a non-negative number.")
            raise ValueError("beta must be a non-negative number.")

    @property
    def simulation_delay(self):
        return self._simulation_delay

    @simulation_delay.setter
    def simulation_delay(self, value):
        if isinstance(value, (int, float)) and value >= 0:
            self._simulation_delay = value
            logging.debug(f"Set simulation_delay to {value}")
        else:
            logging.error("Invalid simulation_delay value. It must be a non-negative number.")
            raise ValueError("simulation_delay must be a non-negative number.")

    # Metrics
    @property
    def metrics(self):
        return self._metrics

    def update_metrics(self, key, value):
        """Update a specific metric."""
        if key in self._metrics:
            self._metrics[key].append(value)
            logging.debug(f"Updated metric '{key}' with value {value}")
        else:
            logging.error(f"Invalid metric key: {key}")
            raise KeyError(f"Metric '{key}' does not exist.")

    # Top results
    @property
    def top_5_configs(self):
        return self._top_5_configs

    @top_5_configs.setter
    def top_5_configs(self, value):
        if isinstance(value, list):
            self._top_5_configs = value[:5]  # Ensure only top 5 are stored
            logging.debug("Updated top_5_configs")
        else:
            logging.error("top_5_configs must be a list.")
            raise ValueError("top_5_configs must be a list.")

    @classmethod
    def get_instance(cls):
        """Provide a class method to get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def reset(self):
        """Reset configuration to default values."""
        logging.debug("Resetting configuration to default values.")
        self._grid_size = 20
        self._max_generations = 200
        self._stability_threshold = 5
        self._population_size = 20
        self._generations = 50
        self._mutation_rate = 0.1
        self._alpha = 0.5
        self._beta = 1.5
        self._simulation_delay = 0.1
        self._metrics = {
            "best_fitness": [],
            "avg_fitness": [],
            "fitness_variance": []
        }
        self._top_5_configs = []
