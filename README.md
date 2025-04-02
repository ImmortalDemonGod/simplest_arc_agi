# Neural Circuit Extraction Framework

A framework for extracting, analyzing, and composing neural circuits from transformer models, aimed at advancing interpretability, modularity, and reusability in AI systems.

## Project Overview

The Neural Circuit Extraction and Modular Composition Framework provides tools and methodologies for:

1. **Training specialized transformer models** on carefully constructed tasks
2. **Extracting neural circuits** that implement specific cognitive functions
3. **Analyzing and interpreting** the behavior of these circuits
4. **Composing circuits** to create new capabilities
5. **Evaluating performance** on the Abstraction and Reasoning Corpus (ARC)

Our approach is inspired by François Chollet's work on measuring intelligence through skill-acquisition efficiency and aims to create more transparent, modular AI systems.

## Documentation

This project includes comprehensive documentation built with MkDocs. The documentation covers:

- Project overview and motivation
- Detailed component descriptions
- Implementation roadmap
- ARC evaluation protocol
- Future directions

### Viewing the Documentation

To view the documentation locally:

1. Make sure you have MkDocs installed:
   ```bash
   pip install mkdocs mkdocs-material
   ```

2. Navigate to the project directory and run:
   ```bash
   mkdocs serve
   ```

3. Open your browser to http://127.0.0.1:8000

### Documentation Structure

- **Overview** - High-level project introduction
- **Components** - Detailed information about each system component:
  - Data Generation
  - Transformer Architecture
  - Training Pipeline
  - Scalability Infrastructure
  - Circuit Database
  - Explanation & Interpretability
  - Modular Composition
- **Implementation Roadmap** - Phased development plan
- **ARC Evaluation Protocol** - Testing methodology
- **Future Directions** - Research and development plans

## Installation

To install the framework:

```bash
# Clone the repository
git clone https://github.com/yourusername/simplest_arc_agi.git
cd simplest_arc_agi

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from src.data_generation import TaskGenerator
from src.models import TransformerModel
from src.training import Trainer
from src.extraction import CircuitExtractor
from src.database import CircuitDatabase
from src.composition import CircuitComposer

# Generate data for a simple task
task_gen = TaskGenerator()
train_data, test_data = task_gen.generate_task("pattern_completion", n_examples=1000)

# Train a transformer model
model = TransformerModel(config={
    "d_model": 128, 
    "n_layers": 4, 
    "n_heads": 4
})
trainer = Trainer(model)
trainer.train(train_data, epochs=50)

# Extract circuits
extractor = CircuitExtractor(model)
circuits = extractor.extract_circuits(train_data)

# Save circuits to database
db = CircuitDatabase("./circuit_db")
for circuit_name, circuit in circuits.items():
    db.add_circuit(circuit_name, circuit)

# Compose circuits for a new task
composer = CircuitComposer(db)
composed_circuit = composer.compose({
    "components": ["pattern_recognizer", "shape_transformer"],
    "connections": [{"from": "pattern_recognizer.output", "to": "shape_transformer.input"}]
})

# Test the composed circuit
results = composed_circuit.apply(test_data)
```

## Project Structure

```
simplest_arc_agi/
├── docs/                 # Documentation
├── src/                  # Source code
│   ├── data_generation/  # Task generation components
│   ├── models/           # Transformer architecture
│   ├── training/         # Training infrastructure
│   ├── extraction/       # Circuit extraction tools
│   ├── database/         # Circuit database
│   ├── explanation/      # Interpretability tools
│   ├── composition/      # Circuit composition framework
│   └── evaluation/       # ARC evaluation system
├── tests/                # Test suite
├── examples/             # Example notebooks and scripts
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Contributing

We welcome contributions to the Neural Circuit Extraction Framework! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this framework in your research, please cite:

```
@software{neural_circuit_extraction_framework,
  author = {ARC AGI Team},
  title = {Neural Circuit Extraction and Modular Composition Framework},
  year = {2023},
  url = {https://github.com/yourusername/simplest_arc_agi}
}
```

## Acknowledgments

- François Chollet for the original ARC challenge and insights on intelligence measurement
- The mechanistic interpretability research community for pioneering circuit analysis approaches 