# Neural Circuit Extraction and Modular Composition

This repository implements a framework for training transformer models on algorithmic tasks, extracting interpretable neural circuits, and composing them in a modular way to solve more complex problems.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/simplest_arc_agi.git
cd simplest_arc_agi
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `src/data_generation/`: Modules for generating algorithmic task datasets
- `src/models/`: Transformer architecture and related model components
- `src/training/`: Trainer classes and utilities for model training
- `src/database/`: Circuit database for storing and retrieving neural circuits
- `src/utils/`: Utility functions shared across modules
- `main.py`: Main script demonstrating the complete pipeline

## Quick Start

Run a basic training experiment on modular addition:

```bash
python main.py --task add_mod_11 --modulus 11 --max_epochs 5000
```

This will:
1. Generate a dataset for addition modulo 11
2. Create and train a transformer model
3. Extract a simple circuit representation
4. Store the circuit in the database

## Configuration Options

The main script accepts several configuration options:

- `--task`: Task name (default: "add_mod_11")
- `--modulus`: Modulus for modular addition (default: 11)
- `--train_ratio`: Fraction of examples for training (default: 0.5)
- `--hidden_size`: Hidden size of transformer (default: 128)
- `--num_layers`: Number of transformer layers (default: 2)
- `--num_heads`: Number of attention heads (default: 4)
- `--max_epochs`: Maximum number of epochs (default: 10000)
- `--batch_size`: Batch size (default: 64)
- `--learning_rate`: Learning rate (default: 1e-3)
- `--weight_decay`: Weight decay (default: 0.01)

## Next Steps

After training models and extracting circuits:

1. **Explore the database**: Query circuits based on task, performance, or features.
2. **Compose circuits**: Use extracted circuits as building blocks for solving more complex tasks.
3. **Implement advanced extraction**: Add more sophisticated circuit extraction techniques.
4. **Expand task library**: Add new algorithmic tasks beyond modular addition.

## License

[MIT License](LICENSE) 