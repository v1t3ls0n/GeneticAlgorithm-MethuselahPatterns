# Game of Life Genetic Algorithm

A Python implementation that uses genetic algorithms to evolve interesting patterns in Conway's Game of Life. The project combines cellular automata with evolutionary computation to discover configurations that exhibit desired behaviors like longevity, growth, and stability.

## 🎮 Overview

This project consists of three main components:

1. **Game of Life Implementation**: A class that simulates Conway's Game of Life rules and tracks various metrics about the evolution of patterns.
2. **Genetic Algorithm**: An evolutionary system that breeds and mutates Game of Life configurations to optimize for specific characteristics.
3. **Interactive Visualization**: A two-window GUI system built with Matplotlib and PyQt5 for visualizing both the evolved patterns and the optimization process.

## 🚀 Features

- Genetic algorithm optimization for:
  - Pattern lifespan (number of unique states)
  - Maximum cell population
  - Growth ratio (max/min population)
  - Pattern stability
  - Compact initial configurations
- Dynamic mutation rate adjustment
- Stagnation detection and recovery
- Comprehensive fitness metrics tracking
- Interactive visualization with:
  - Real-time pattern evolution display
  - Statistical analysis plots
  - Navigation between top configurations
  - Generation-by-generation playback

## 📋 Requirements

```
python >= 3.7
numpy
matplotlib
PyQt5
logging
```

## 💻 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/game-of-life-genetic
cd game-of-life-genetic
```

2. Install required packages:
```bash
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

- [Your Name] - Initial work

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