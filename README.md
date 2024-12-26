# Genetic Algorithm for Finding Methuselahs in Conway's Game of Life

This project implements a genetic algorithm designed to evolve configurations in Conway's Game of Life to optimize for long lifespans, maximum alive cells, and growth ratios. The primary goal is to discover "Methuselahs" — patterns that live for many generations before stabilizing or repeating.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Main Function Parameters](#main-function-parameters)
- [How to Use](#how-to-use)
- [File Structure](#file-structure)

---

## Overview

The system consists of:
1. A genetic algorithm that evolves configurations based on fitness scores.
2. A simulation of Conway's Game of Life to evaluate configurations.
3. An interactive visualization tool for exploring results.

The genetic algorithm evaluates configurations for:
- **Lifespan**: The number of unique generations before stabilization or repetition.
- **Maximum Alive Cells**: The maximum number of living cells in any generation.
- **Growth Ratio**: The ratio between the maximum and minimum number of living cells.
- **Stableness**: The frequency of reaching static or periodic behavior.

---

## Features

- **Dynamic Mutation Rate**: Adjusted based on the fitness trends.
- **Interactive Visualization**: Navigate through generations and configurations.
- **Fitness Optimization**: Multi-objective scoring system to balance lifespan, growth, and stability.

---

## Main Function Parameters

The `main` function is the entry point for running the genetic algorithm. Below is a detailed explanation of its parameters:

### Parameters

1. **`grid_size`**
   - **Type**: `int`
   - **Default**: `20`
   - **Description**: The dimension of the grid (NxN) used for the Game of Life simulation.

2. **`population_size`**
   - **Type**: `int`
   - **Default**: `20`
   - **Description**: The number of configurations in the population for each generation.

3. **`generations`**
   - **Type**: `int`
   - **Default**: `100`
   - **Description**: The number of generations to simulate in the genetic algorithm.

4. **`initial_mutation_rate`**
   - **Type**: `float`
   - **Default**: `1.0`
   - **Description**: The initial probability of mutating a cell in a configuration.

5. **`mutation_rate_lower_limit`**
   - **Type**: `float`
   - **Default**: `0.2`
   - **Description**: The minimum mutation rate to avoid over-stabilization of the population.

6. **`alive_cells_weight`**
   - **Type**: `float`
   - **Default**: `0.12`
   - **Description**: The weight assigned to the maximum number of alive cells in the fitness score.

7. **`lifespan_weight`**
   - **Type**: `float`
   - **Default**: `200.0`
   - **Description**: The weight assigned to the lifespan of a configuration in the fitness score.

8. **`alive_growth_weight`**
   - **Type**: `float`
   - **Default**: `0.1`
   - **Description**: The weight assigned to the growth ratio (max/min alive cells) in the fitness score.

9. **`stableness_weight`**
   - **Type**: `float`
   - **Default**: `0.01`
   - **Description**: The weight assigned to how often a configuration reaches stability.

10. **`alive_cells_per_block`**
    - **Type**: `int`
    - **Default**: `5`
    - **Description**: The maximum number of alive cells in a block during random initialization.

11. **`alive_blocks`**
    - **Type**: `int`
    - **Default**: `3`
    - **Description**: The number of blocks containing alive cells in the initial random configuration.

12. **`initial_living_cells_count_weight`**
    - **Type**: `float`
    - **Default**: `0.7`
    - **Description**: The weight for penalizing large initial configurations in the fitness score.

13. **`predefined_configurations`**
    - **Type**: `None or Iterable`
    - **Default**: `None`
    - **Description**: Allows injecting known patterns as initial configurations.


---

## Features

- **Genetic Algorithm**:
  - Custom fitness evaluation combining lifespan, alive cell counts, and stability.
  - Parent selection, mutation, and crossover for population evolution.
  - Dynamic mutation rate adjustment to handle stagnation.
- **Simulation**:
  - Full implementation of Conway's Game of Life with grid dynamics.
  - Detection of static and periodic states.
- **Interactive Visualization**:
  - Dual-window setup for grid and metrics visualization.
  - Navigation through configurations and generations using keyboard controls.
- **Logging and Analysis**:
  - Detailed logs for population statistics and algorithm progress.
  - Metrics include lifespan, growth ratio, stability, and mutation rate history.

---

## Installation

### Prerequisites
- Python 3.8+
- Required libraries: `numpy`, `matplotlib`, `PyQt5`

### Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Interactive Mode
Run the program interactively:
```bash
python main.py
```

### Command-Line Parameters
Customize the simulation by passing arguments to the `main` function in `main.py`. Example:
```python
main(grid_size=30, population_size=50, generations=150, initial_mutation_rate=0.5)
```

### Key Parameters
- `grid_size`: Dimension of the NxN grid.
- `population_size`: Number of individuals per generation.
- `generations`: Number of generations to simulate.
- `initial_mutation_rate`: Starting probability of mutation.
- `lifespan_weight`: Fitness weight for lifespan.
- `alive_cells_weight`: Fitness weight for alive cells.
- `stableness_weight`: Fitness weight for stability.

---

## Visualization

### Grid Window
Displays the NxN grid of the current configuration and generation. Navigate through configurations and generations using:
- **Arrow Keys**:
  - UP/DOWN: Switch configurations.
  - LEFT/RIGHT: Switch generations.

### Metrics Window
Displays population statistics over generations, including:
- Fitness scores
- Lifespan
- Growth rates
- Mutation rate history

---

## Logging and Results

All simulation details are logged to `simulation.log`. Key metrics include:
- Fitness scores for all configurations.
- Mutation rate adjustments.
- Detection of stagnation.
- Top-performing configurations with their metrics.

### Sample Log Output
```
2024-12-26 18:46:40,905 - INFO - Starting run with parameters: grid_size=20, population_size=20, generations=200, ...
2024-12-26 18:47:36,031 - INFO - Mutation rate increased due to stagnation.
2024-12-26 18:48:15,799 - INFO - Top Configuration:
2024-12-26 18:48:15,799 - INFO -   Configuration: (1, 0, 0, 0, 1, ...)
2024-12-26 18:48:15,799 - INFO - Fitness Score: 2296.899
2024-12-26 18:48:15,799 - INFO - Lifespan: 333
```

---

## Folder Structure

```plaintext
📂 Project Root
├── GameOfLife.py             # Simulation logic for Conway's Game of Life.
├── GeneticAlgorithm.py       # Genetic Algorithm for Methuselah discovery.
├── InteractiveSimulation.py  # Interactive visualization using PyQt5 and Matplotlib.
├── main.py                   # Entry point for running the program.
├📂  scripts                   # Utility scripts for maintenance and testing.
├── simulation.log            # Log file for simulation results and metrics.
├── requirements.txt          # Python dependencies.
├── .gitignore                # Git ignore rules.
```

---

## Acknowledgments

This project was inspired by the beauty of Conway's Game of Life and the computational challenge of discovering Methuselahs. Special thanks to the Open University of Israel for providing foundational AI concepts.

---

## Future Enhancements

- Implement parallel processing for faster fitness evaluation.
- Extend visualization to include 3D metrics for better insights.
- Add a web-based interface for broader accessibility.

---

Feel free to suggest improvements or raise issues in the repository!
```

This `README.md` is formatted for GitHub and includes all relevant information about your project, making it easy to understand and contribute to. Let me know if you'd like further refinements!