# Methuselah Pattern Finder for Conway's Game of Life

A genetic algorithm implementation designed to discover Methuselah patterns in Conway's Game of Life. Methuselahs are rare, small initial patterns that evolve for many generations before stabilizing, making them particularly interesting configurations to study.

## 🎮 Overview

This project uses genetic algorithms to evolve and discover potential Methuselah patterns in Conway's Game of Life. It optimizes for:
- Long lifespans (many generations before stabilization)
- High maximum cell counts during evolution
- Significant growth rates from initial state
- Small initial configurations that produce complex behavior

## 🚀 Features

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

## 📋 Requirements

- Python 3.7+
- NumPy
- Matplotlib
- PyQt5

## 💻 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gameoflife-methuselah-finder.git

# Install required packages
pip install numpy matplotlib PyQt5
```
## 🎯 Usage

Run the main script with default parameters:
```bash
python main.py
```

When prompted, you can either:
- Use default parameters by entering 'y'
- Customize parameters by entering 'n' and following the prompts

### Customizable Parameters:

- `grid_size`: Dimension of the NxN grid (default: 20)
- `population_size`: Number of configurations per generation (default: 20)
- `generations`: Number of evolutionary generations (default: 100)
- `initial_mutation_rate`: Starting mutation probability (default: 1.0)
- `mutation_rate_lower_limit`: Minimum mutation rate (default: 0.2)
- `alive_cells_weight`: Fitness weight for maximum population (default: 0.12)
- `lifespan_weight`: Fitness weight for pattern longevity (default: 200.0)
- `alive_growth_weight`: Fitness weight for population growth (default: 0.1)
- `stableness_weight`: Fitness weight for pattern stability (default: 0.01)
- `alive_cells_per_block`: Initial cells per active block (default: 5)
- `alive_blocks`: Number of active blocks in initialization (default: 3)
- `initial_living_cells_count_weight`: Penalty for large initial patterns (default: 0.7)

## 🖥️ Interactive Visualization

The visualization interface consists of two windows:

### Grid Window
- Displays the current Game of Life pattern
- Navigation:
  - ↑/↓: Switch between best configurations
  - ←/→: Step through generations
  - Shows current configuration stats in the window title

### Stats Window
- Multiple plots showing:
  - Fitness evolution
  - Pattern lifespan trends
  - Population growth rates
  - Alive cell counts
  - Mutation rate adaptation
- Run parameters display
- Focus control buttons for window management


## 🧬 Project Structure

- `GameOfLife.py`: Core Game of Life simulation
- `GeneticAlgorithm.py`: Evolution and optimization logic
- `InteractiveSimulation.py`: Visualization interface
- `main.py`: Program entry point and parameter handling


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


## 📊 Understanding the Metrics

- **Lifespan**: Number of unique states before stabilization
- **Max Alive Cells**: Peak population achieved
- **Growth Rate**: Ratio between maximum and minimum populations
- **Stableness**: How quickly patterns reach stable or periodic states
- **Initial Size**: Number of alive cells in the starting configuration

## 🎛️ Fine-tuning Tips

1. **For Longer-lived Patterns**:
   - Increase `lifespan_weight`
   - Decrease `stableness_weight`

2. **For More Dynamic Patterns**:
   - Increase `alive_growth_weight`
   - Decrease `initial_living_cells_count_weight`

3. **For Stable Oscillators**:
   - Increase `stableness_weight`
   - Balance with moderate `lifespan_weight`

4. **For Dense Patterns**:
   - Increase `alive_cells_weight`
   - Increase `alive_cells_per_block`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Bug fixes
- Feature enhancements
- Documentation improvements
- Performance optimizations

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- [Guy Vitelson] - Initial work

## 🙏 Acknowledgments

- Based on Conway's Game of Life cellular automaton
- Inspired by genetic algorithm applications in pattern discovery
- Thanks to the PyQt and Matplotlib communities for visualization tools

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@software{game_of_life_genetic,
  author = {Guy Vitelson},
  title = {Game of Life Genetic Algorithm},
  year = {2024},
  url = {https://github.com/v1t3ls0n/Game-of-Life-Genetic-Algorithm}
}
```
