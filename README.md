# Methuselah Pattern Finder for Conway's Game of Life

A genetic algorithm implementation designed to discover Methuselah patterns in Conway's Game of Life. Methuselahs are rare, small initial patterns that evolve for many generations before stabilizing, making them particularly interesting configurations to study.

## 🎮 Overview

This project uses genetic algorithms to evolve and discover potential Methuselah patterns in Conway's Game of Life. It optimizes for:
- Long lifespans (many generations before stabilization)
- High maximum cell counts during evolution
- Significant growth rates from the initial state
- Small initial configurations that produce complex behavior

## 🚀 Features

### **Genetic Algorithm**
- **Fitness Function**:
  - Optimized for:
    - Long lifespan
    - Maximum alive cell count
    - Significant growth ratio
    - Penalties for large initial configurations
- **Dynamic Mutation Rate Adjustment**:
  - Adapts mutation rate based on stagnation and fitness trends:
    - Increases mutation when fitness plateaus
    - Decreases mutation when fitness improves
- **Selection Strategies**:
  - Normalized Probability (Roulette Wheel)
  - Tournament Selection
  - Rank-Based Selection
- **Crossover Methods**:
  - Basic crossover: Alternates cells between parents
  - Simple crossover: Alternates rows between parents
  - Complex crossover: Uses block-based selection based on fitness
- **Mutation Methods**:
  - Basic mutation: Random flips with probability
  - Cluster mutation: Flips cells in small neighborhoods
  - Harsh mutation: Flips large contiguous blocks of cells
- **Diversity Tracking**:
  - Uses Hamming distance to measure population diversity
  - Prevents convergence to duplicate patterns by enforcing canonical uniqueness
- **Dynamic Penalty System**:
  - Penalizes configurations based on:
    - Canonical form frequency
    - Block frequency
    - Cell frequency

### **Game of Life Simulation**
- **Core Simulation**:
  - Efficient implementation using a 1D array mapped to a 2D grid with NumPy
  - Supports Conway's Game of Life rules for state transitions
  - Configurable boundary types:
    - `wrap`: Periodic boundaries (cells wrap around edges)
    - `fixed`: Fixed boundaries (edges remain static and dead)
- **Pattern Analysis**:
  - Detects static and periodic behavior
  - Tracks lifespan, population dynamics, and stability
- **Performance Metrics**:
  - Lifespan
  - Maximum alive cells
  - Growth rates
  - Stability of patterns

### **Interactive Visualization**
- Dual-window interface showing:
  - **Grid Window**: Displays the NxN Game of Life grid and its evolution
  - **Stats Window**: Shows various metrics, including fitness, lifespan, growth rate, and mutation rate adaptation
  - **Run Parameters Window**: Displays the parameters used in the genetic algorithm run
- Generation-by-generation playback and navigation controls:
  - Switch between discovered configurations
  - Step through generations
  - Zoom into specific metrics

## 📋 Requirements

- Python 3.7+
- NumPy
- Matplotlib
- PyQt5
- SciPy (for convolution in Game of Life simulation)

## 💻 Installation

```bash
# Clone the repository
git clone https://github.com/v1t3ls0n/Methuselah-Pattern-Finder-for-Conway-s-Game-of-Life

# Install required packages
pip install numpy matplotlib pyqt5 scipy
```

## 🎯 Usage

Run the main script with default parameters:

```bash
python main.py
```

When prompted, you can either:
- Use default parameters by entering 'y'
- Customize parameters by entering 'n' and following the prompts

### Customizable Parameters

- `grid_size`: Dimension of the NxN grid (default: 20)
- `population_size`: Number of configurations per generation (default: 20)
- `generations`: Number of evolutionary generations (default: 100)
- `initial_mutation_rate`: Starting mutation probability (default: 1.0)
- `mutation_rate_lower_limit`: Minimum mutation rate (default: 0.2)
- `alive_cells_weight`: Fitness weight for maximum population (default: 0.12)
- `lifespan_weight`: Fitness weight for pattern longevity (default: 200.0)
- `alive_growth_weight`: Fitness weight for population growth (default: 0.1)
- `initial_living_cells_count_penalty_weight`: Penalty for large initial patterns (default: 0.7)
- `boundary_type`: Type of boundary to use for the Game of Life grid (`wrap` or `fixed`, default: `wrap`)

## 🔄 Interactive Visualization

The visualization interface consists of three windows:

### 1. Grid Window
- Displays the current Game of Life pattern
- Navigation:
  - ↑/↓: Switch between best configurations
  - ←/→: Step through generations
  - Shows current configuration stats in the window title

### 2. Stats Window
- Multiple plots showing:
  - Fitness evolution
  - Pattern lifespan trends
  - Population growth rates
  - Alive cell counts
  - Mutation rate adaptation
  - Diversity metrics (average Hamming distance between configurations)
- Run parameters display
- Focus control buttons for window management

### 3. Run Parameters Window
- Displays the parameters used in the GA run for reference.

## 🔢 Key Metrics

- **Lifespan**: Number of unique states before stabilization
- **Max Alive Cells**: Peak population achieved
- **Growth Rate**: Ratio between maximum and minimum populations
- **Stableness**: Whether the pattern stabilizes or becomes periodic
- **Initial Size**: Number of alive cells in the starting configuration
- **Diversity**: Measures the average Hamming distance between configurations over generations

## 📊 Fine-tuning Tips

1. **For Longer-lived Patterns**:
   - Increase `lifespan_weight`
   - Decrease `stableness_weight`

2. **For More Dynamic Patterns**:
   - Increase `alive_growth_weight`
   - Decrease `initial_living_cells_count_penalty_weight`

3. **For Stable Oscillators**:
   - Increase `stableness_weight`
   - Balance with moderate `lifespan_weight`

4. **For Dense Patterns**:
   - Increase `alive_cells_weight`
   - Increase `alive_cells_per_block`

## 🤓 Advanced Features

### Canonical Uniqueness
Ensures all configurations are unique by normalizing patterns:
- Removes redundant rotations and translations
- Tracks canonical forms to prevent duplication

### Diversity Tracking
- Measures Hamming distance between configurations
- Ensures population retains genetic variety

### Corrected Scores
- Adjusts fitness based on frequency of canonical forms and recurrent blocks
- Encourages exploration of less common patterns

## 🙏 Acknowledgments

- Based on Conway's Game of Life cellular automaton
- Inspired by genetic algorithm applications in pattern discovery
- Thanks to the PyQt and Matplotlib communities for visualization tools

## 📈 Citation

If you use this code in your research, please cite:

```bibtex
@software{game_of_life_genetic,
  author = {Guy Vitelson},
  title = {Methuselah Pattern Finder for Conway's Game of Life},
  year = 2024,
  url = {https://github.com/v1t3ls0n/Methuselah-Pattern-Finder-for-Conway-s-Game-of-Life}
}
```

