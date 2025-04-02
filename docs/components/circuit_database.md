# Circuit Database and Analysis Framework

The Circuit Database and Analysis Framework is the core of our system, enabling the extraction, cataloging, and composition of neural circuits from trained models.

## Circuit Extraction Pipeline

We implement automated methods to extract interpretable circuits from transformer models:

### Extraction Techniques

Our system employs two primary extraction methods:

#### Cross-Layer Transcoders (CLTs) via Dictionary Learning

- **Sparse Autoencoders**: Train autoencoders (transcoders) to map activations into an interpretable, sparse feature space
- **Parameter Tuning**: Explore variations in sparsity penalties, dictionary sizes, and architectures
- **Layer Mapping**: Map activations between (or within) layers to isolate computation paths

#### Attribution Graph Construction

- **Causal Tracing**: Trace influence from inputs to outputs through identified CLT features
- **Attribution Methods**: Use techniques like integrated gradients, activation patching, and path patching
- **Graph Pruning**: Heavily prune paths to focus on significant, interpretable connections

These techniques allow us to identify the specific subnetworks responsible for algorithmic computations within the larger model.

## Centralized Circuit Database

We design and implement a structured database for storing and retrieving neural circuits:

### Database Schema

Our schema includes the following key fields:

- **`circuit_id`**: Unique identifier for the circuit instance
- **`task_name`**: The specific algorithmic task (e.g., `add_mod_97`, `grid_rotate_90`)
- **`task_description`**: Human-readable description and parameters
- **`model_architecture`**: Base model details (layers, width), optimizations used, pruning level, parameter count
- **`training_details`**: Key hyperparameters, dataset split, training steps to generalization
- **`circuit_structure`**: Representation of the identified circuit components (nodes, connections, weights)
- **`interpretation`**: Human-readable label and semantic description of the circuit's function
- **`activation_examples`**: Representative input examples that strongly activate the circuit
- **`performance_metrics`**: Fidelity/accuracy of the isolated circuit
- **`interface_definition`**: Description of the circuit's input/output interface (crucial for composition)
- **`metadata`**: Pointers to source code, relevant papers, validation results, date extracted

### Circuit Tagging System

We implement a tagging system to categorize circuits by:

- **Prior Alignment**: `Core_Object`, `Core_Geometry`, `Core_Number`, `Core_Topology`, etc.
- **Complexity**: `Primitive`, `Compound`, `Highly_Specialized`
- **Generality**: Quantitative scores based on cross-task validation

## Visualization Tools

We develop interactive visualization tools for exploring and understanding circuits:

- **Attribution Graph Visualization**: Tools to explore paths and node details
- **Feature Activation Explorer**: View activation patterns across different inputs
- **Circuit Comparison Views**: Side-by-side comparison of different circuits
- **Database Query Interface**: User-friendly interface for searching and filtering circuits

## Modular Composition Framework

The modular composition framework treats circuits as reusable functional modules:

### Standardized Interfaces

To enable composition, we define standards for circuit communication:

- **Common Representation Spaces**: Attempt to enforce shared representation during CLT training
- **Adapter Networks**: Use learned transformations or small adapters between composed modules
- **Normalization Layers**: Apply normalization (e.g., LayerNorm) at circuit boundaries
- **Interface Documentation**: Clearly document requirements in the database

### Composition Engine

We develop mechanisms to functionally chain circuits:

#### Static Composition
Define fixed graphs of circuit connections for specific complex tasks:
```python
output = circuit_B(circuit_A(input))
```

#### Dynamic Composition
Implement a meta-controller (potentially LLM-based) to select and route between circuits based on input or intermediate state.

### "Code-like" Representation

We explore representing composed circuits in a human-readable format:

```python
# Example of a code-like representation for (a*b + c) / 2

def compute_expression(a, b, c):
    # Circuit IDs are retrieved from DB
    multiply_circuit = get_circuit_by_id('mul_mod_p_v1')
    add_circuit = get_circuit_by_id('add_mod_p_v2')
    halve_circuit = get_circuit_by_id('halve_int_v1')

    # Execute composition pipeline
    intermediate1 = multiply_circuit(a, b)
    intermediate2 = add_circuit(intermediate1, c)
    result = halve_circuit(intermediate2)
    return result
```

### LLM Assistance for Composition

We investigate using modern LLMs (e.g., GPT-4, Claude 3) to assist with:

- **Suggesting Compositions**: Given a task description and circuit database, propose compositions
- **Verifying Correctness**: Analyze proposed composition plans for logical soundness
- **Translating Descriptions**: Convert natural language algorithm descriptions to code-like circuit compositions
- **Generating Adapters**: Create "glue code" for interface compatibility

## Implementation

Our current implementation provides a simple version of the circuit database:

```python
class CircuitDatabase:
    def __init__(self, db_path="circuits.db"):
        """Initialize database connection"""
        self.db_path = db_path
        self._initialize_db()
    
    def _initialize_db(self):
        """Create database schema if it doesn't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create circuits table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS circuits (
            circuit_id TEXT PRIMARY KEY,
            task_name TEXT NOT NULL,
            task_description TEXT,
            model_architecture TEXT NOT NULL,
            training_details TEXT,
            circuit_structure TEXT,
            interpretation TEXT,
            interface_definition TEXT,
            metadata TEXT,
            creation_date TEXT,
            fidelity REAL,
            extraction_method TEXT
        )
        ''')
        
        # Create additional tables for tags, activation examples, etc.
        # ...
        
        conn.commit()
        conn.close()
    
    def add_circuit(self, circuit_id, task_name, model_architecture, circuit_structure, 
                   interface_definition, task_description=None, training_details=None, 
                   interpretation=None, metadata=None, fidelity=None, 
                   extraction_method=None, tags=None):
        """Add a new circuit to the database"""
        # Implementation for adding circuits
        # ...
    
    def query_circuits(self, task_name=None, tags=None, min_fidelity=None, 
                      extraction_method=None, keyword=None):
        """Query circuits based on various criteria"""
        # Implementation for querying circuits
        # ...
```

## Future Directions

Future work on the circuit database and analysis framework includes:

1. **Advanced Extraction Techniques**: Implement more sophisticated circuit extraction methods
2. **Interface Standardization**: Develop rigorous standards for circuit interfaces
3. **Automated Discovery**: Tools to automatically identify and extract circuits from trained models
4. **Composition Verification**: Methods to verify the correctness of composed circuits
5. **LLM Integration**: Deeper integration with LLMs for automated composition and analysis 