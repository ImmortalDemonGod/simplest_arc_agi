# Data Generation Module

The Data Generation Module is responsible for automatically creating structured datasets for a wide range of algorithmic tasks. These datasets serve as the foundation for training our transformer models and extracting algorithmic neural circuits.

## Key Features

- **Diverse Algorithmic Task Generation**: Create datasets spanning from simple operations to complex algorithmic challenges
- **Parameterized Generation**: Control dataset characteristics like complexity, size, and representation
- **Novel Task Synthesis**: Automatically create new tasks by combining existing primitives

## Algorithmic Task Library

The module maintains and expands a library of primitive functions implementing various algorithmic operations:

### Arithmetic Operations
- `add`, `subtract`, `multiply`, `divide` (integer, modular, tuple-based)
- Example: Modular addition `(a + b) % n`

### Logical Operations
- `even`, `flip`, `both`, `either`, `positive`, `equality`
- Example: Boolean operations like `a AND b`

### Set/Container Operations
- `combine`, `intersection`, `difference`, `dedupe`, `order`, `size`, `filter`
- Example: Set operations like union and intersection

### Grid/Patch Manipulations
- Based on `arc_types` primitives
- `rot90`, `hmirror`, `crop`, `objects`, `fill`, `upscale`
- Example: Rotating or reflecting shapes in a grid

### Sequence Operations
- `dedupe`, `order`, simple string edits
- Example: Removing duplicates from a sequence

## Structured Dataset Generator

The Structured Dataset Generator creates formatted datasets ready for training:

- **Input/Output Format**: Generates sequences like "Input: a OP b | Output: c" or more complex grid structures
- **Binary Operation Tables**: Focus on tasks like `a ◦ b = c` (modular arithmetic, group operations) for grokking studies
- **Parameterization**:
  - Task complexity (modulus size, grid dimensions)
  - Input/output representations (base 10 vs. symbolic tokens, grid serialization)
  - Dataset splits (e.g., 50% train for grokking studies)
  - Optional noise or outlier injection

## Novel Task Synthesis Engine

The Novel Task Synthesis Engine creates new, potentially more complex tasks:

- **Composition**: Generate new tasks by composing primitive functions (e.g., `f(x,y,z) = add(multiply(x,y), z)`)
- **Validation Checks**: Ensure generated tasks are:
  - Well-posed and solvable within defined constraints
  - Non-trivial (not solvable by simple heuristics)
  - Suitable for probing specific reasoning capabilities

## Implementation

Our current implementation focuses on binary operations, particularly modular arithmetic. Below is an example of modular addition data generation:

```python
def generate_modular_addition_data(modulus: int, train_ratio: float = 0.5):
    """
    Generate a dataset for modular addition: (a + b) % modulus
    
    Args:
        modulus: The modulus for the addition operation
        train_ratio: Fraction of examples to use for training
    
    Returns:
        Dictionary containing input and target tensors for train and test sets
    """
    # Generate all possible a,b pairs
    all_pairs = [(a, b) for a in range(modulus) for b in range(modulus)]
    all_inputs = torch.tensor(all_pairs, dtype=torch.long)
    
    # Calculate targets: (a + b) % modulus
    all_targets = (all_inputs[:, 0] + all_inputs[:, 1]) % modulus
    
    # Shuffle and split into train and test
    indices = torch.randperm(len(all_pairs))
    all_inputs = all_inputs[indices]
    all_targets = all_targets[indices]
    
    train_size = int(len(all_pairs) * train_ratio)
    
    return {
        "train_inputs": all_inputs[:train_size],
        "train_targets": all_targets[:train_size],
        "test_inputs": all_inputs[train_size:],
        "test_targets": all_targets[train_size:]
    }
```

## Future Directions

Future work on the Data Generation Module includes:

1. **Expanded Primitive Library**: Implement more diverse algorithmic primitives
2. **Multi-Step Tasks**: Generate tasks requiring multi-step reasoning
3. **Task Difficulty Metrics**: Develop methods to quantify and control task complexity
4. **Curriculum Generation**: Create progressive task sequences of increasing difficulty
5. **Adversarial Examples**: Generate challenging edge cases to test model robustness 