# Neural Circuit Extraction and Modular Composition Framework

Welcome to the documentation for the Neural Circuit Extraction and Modular Composition Framework, an open-source project focused on extracting interpretable neural circuits from transformer models and composing them to create more transparent, modular AI systems.

## Project Vision

We aim to advance AI interpretability and modularity by:

1. Training specialized transformer models on carefully constructed tasks
2. Extracting neural circuits that implement specific cognitive functions
3. Creating a database of reusable, documented circuit components
4. Developing a composition framework for building new capabilities from existing circuits
5. Evaluating system performance on the Abstraction and Reasoning Corpus (ARC)

Our approach is inspired by François Chollet's work on measuring intelligence through skill-acquisition efficiency rather than task-specific performance.

## Documentation Structure

This documentation is organized into the following sections:

### Overview and Concepts
- [Project Overview](overview.md) - High-level introduction to the project vision, goals, and approach

### Core Components
- [Data Generation](components/data_generation.md) - Task-specific dataset generation for circuit extraction
- [Transformer Architecture](components/transformer_architecture.md) - Modified transformer models optimized for circuit extraction
- [Training Pipeline](components/training_pipeline.md) - Concurrent training framework with hyperparameter optimization
- [Scalability Infrastructure](components/scalability.md) - Scaling infrastructure for models, datasets, and inference
- [Circuit Database](components/circuit_database.md) - Storage and retrieval system for extracted circuits
- [Explanation & Interpretability](components/explanation.md) - Tools for analyzing and interpreting circuit behavior
- [Modular Composition](components/modular_composition.md) - Framework for composing circuits into new capabilities

### Strategic Planning
- [Implementation Roadmap](roadmap.md) - Phased development plan with milestones
- [ARC Evaluation Protocol](arc_evaluation.md) - Testing methodology aligned with ARC principles
- [Future Directions](future_directions.md) - Research and development roadmap

### API Reference
- [Data Generation API](api/data_generation.md) - API for task generation
- [Models API](api/models.md) - API for transformer models
- [Training API](api/training.md) - API for training infrastructure
- [Database API](api/database.md) - API for circuit database

### Tutorials
- [Getting Started](tutorials/getting_started.md) - First steps with the framework
- [Training a Model](tutorials/training.md) - Guide to training specialized models
- [Extracting Circuits](tutorials/extraction.md) - How to extract neural circuits
- [Composing Circuits](tutorials/composition.md) - Guide to circuit composition

## Key Features

- **Interpretable by Design**: Our transformer architecture is optimized for circuit extraction
- **Concurrent Training**: Efficient parallelized training across multiple models
- **Circuit Database**: Centralized storage of extracted circuits with detailed metadata
- **Compositional Framework**: Tools for combining circuits to create new capabilities
- **ARC Alignment**: Evaluation methodology aligned with the Abstraction and Reasoning Corpus
- **Scalability**: Infrastructure for handling larger models and datasets

## Getting Started

To get started with the Neural Circuit Extraction Framework:

1. Check out the [Getting Started](tutorials/getting_started.md) tutorial
2. Explore the [Project Overview](overview.md) for a conceptual introduction
3. Review the [Implementation Roadmap](roadmap.md) to understand the development plan

## Contributing

We welcome contributions to the Neural Circuit Extraction Framework! See our [GitHub repository](https://github.com/yourusername/simplest_arc_agi) for contribution guidelines.

## License

This project is licensed under the MIT License. See the LICENSE file in the GitHub repository for details.

```bash
# Clone the repository
git clone https://github.com/yourusername/simplest_arc_agi.git
cd simplest_arc_agi

# Install dependencies
pip install -r requirements.txt

# Run a simple training experiment
python main.py --task add_mod_11 --modulus 11 --max_epochs 5000
```
