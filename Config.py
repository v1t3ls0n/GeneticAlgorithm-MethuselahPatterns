# Config.py

class Config:
    _instance = None  # Singleton instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):  # Ensure attributes are initialized only once
            # Simulation parameters
            self._grid_size = 20
            self._max_generations = 200
            self._stability_threshold = 5

            # Genetic algorithm parameters
            self._population_size = 20
            self._generations = 50
            self._mutation_rate = 0.1
            self._alpha = 0.5  # Weight for stabilization time
            self._beta = 1.5   # Weight for final size

            # Visualization parameters
            self._simulation_delay = 0.1

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
        self._grid_size = value

    @property
    def max_generations(self):
        return self._max_generations

    @max_generations.setter
    def max_generations(self, value):
        self._max_generations = value

    @property
    def stability_threshold(self):
        return self._stability_threshold

    @stability_threshold.setter
    def stability_threshold(self, value):
        self._stability_threshold = value

    @property
    def population_size(self):
        return self._population_size

    @population_size.setter
    def population_size(self, value):
        self._population_size = value

    @property
    def generations(self):
        return self._generations

    @generations.setter
    def generations(self, value):
        self._generations = value

    @property
    def mutation_rate(self):
        return self._mutation_rate

    @mutation_rate.setter
    def mutation_rate(self, value):
        self._mutation_rate = value

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = value

    @property
    def simulation_delay(self):
        return self._simulation_delay

    @simulation_delay.setter
    def simulation_delay(self, value):
        self._simulation_delay = value

    # Metrics
    @property
    def metrics(self):
        return self._metrics

    def update_metrics(self, key, value):
        """Update a specific metric."""
        if key in self._metrics:
            self._metrics[key].append(value)

    # Top results
    @property
    def top_5_configs(self):
        return self._top_5_configs

    @top_5_configs.setter
    def top_5_configs(self, value):
        self._top_5_configs = value

    @classmethod
    def get_instance(cls):
        """Provide a class method to get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
