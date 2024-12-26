# Methuselah Pattern Finder for Conway's Game of Life

A genetic algorithm implementation designed to discover Methuselah patterns in Conway's Game of Life. Methuselahs are rare, small initial patterns that evolve for many generations before stabilizing, making them particularly interesting configurations to study.

## Overview

This project uses genetic algorithms to evolve and discover potential Methuselah patterns in Conway's Game of Life. It optimizes for:
- Long lifespans (many generations before stabilization)
- High maximum cell counts during evolution
- Significant growth rates from initial state
- Small initial configurations that produce complex behavior

## Features

- **Genetic Algorithm Implementation**
  - Custom fitness function balancing multiple objectives
  - Dynamic mutation rate adjustment to prevent stagnation
  - Block-based crossover strategy
  - Configurable population size and generation count

- **Game of Life Simulator**
  - Efficient implementation using 1D array mapped to 2D grid
  - Detection of static and periodic states
  - Comprehensive statistics tracking

- **Interactive Visualization**
  - Dual-window interface showing:
    - Grid evolution of best patterns
    - Statistical plots (fitness, lifespan, growth rates)
    - Generation-by-generation playback
  - Navigation controls for exploring discovered patterns

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- PyQt5

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gameoflife-methuselah-finder.git

# Install required packages
pip install numpy matplotlib PyQt5
```

## Usage

### Basic Run
```python
python main.py
```
This will start the program with default parameters optimized for Methuselah pattern discovery.

### Custom Parameters
You can customize various parameters when running the program:

```python
from main import main

main(
    grid_size=20,                           # Size of the Game of Life grid
    population_size=20,                     # Number of patterns per generation
    generations=100,                        # How many generations to evolve
    initial_mutation_rate=1.0,              # Starting mutation probability
    alive_cells_weight=0.12,               # Fitness weight for max alive cells
    lifespan_weight=200.0,                 # Fitness weight for pattern longevity
    alive_growth_weight=0.1,               # Fitness weight for growth ratio
    stableness_weight=0.01,                # Fitness weight for stability
    alive_cells_per_block=5,               # Max alive cells per block in initialization
    alive_blocks=3,                        # Number of blocks with alive cells
    initial_living_cells_count_weight=0.7   # Weight to favor smaller initial patterns
)
```

## Key Components

### GameOfLife.py
Implements the core Game of Life simulation logic, including:
- Grid state management
- Neighbor counting
- State transition rules
- Pattern evolution tracking

### GeneticAlgorithm.py
Handles the evolution of patterns through:
- Selection of parent patterns
- Crossover operations
- Mutation
- Fitness evaluation
- Population management

### InteractiveSimulation.py
Provides visualization and analysis tools:
- Pattern evolution display
- Statistical tracking
- Interactive navigation
- Performance metrics

## Visualization Interface

The program provides two windows:
1. **Grid Window**: Shows the current pattern's evolution
   - Use arrow keys to navigate patterns and generations
   - Displays pattern statistics

2. **Stats Window**: Shows evolution metrics
   - Fitness trends
   - Lifespan statistics
   - Growth rates
   - Mutation rate adaptation

## Contributing

Contributions are welcome! Some areas for potential improvement:

- Additional fitness metrics for Methuselah detection
- Alternative crossover strategies
- Performance optimizations
- Pattern classification tools
- Enhanced visualization features

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on Conway's Game of Life
- Inspired by research into Methuselah patterns
- Built with Python's scientific computing stack

## Author

[Guy Vitelson]