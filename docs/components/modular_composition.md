# Modular Circuit Composition

This document details our framework for composing extracted neural circuits into novel, functionally complex systems.

## Core Principles

Our modular composition framework is built on these foundational principles:

1. **Compositionality**: Neural circuits can be meaningfully combined to create more complex functionality
2. **Interface Standardization**: Well-defined interfaces enable reliable circuit connection
3. **Functional Preservation**: Composed circuits maintain core functionality from their original contexts
4. **Emergent Behavior**: Novel capabilities can emerge from composition of simpler circuits

## Circuit Interface Definition

Each circuit in our system has standardized interfaces:

```python
class CircuitInterface:
    def __init__(self, name, input_dims, output_dims, activation_constraints=None):
        self.name = name
        self.input_dims = input_dims  # Dict mapping input names to shapes
        self.output_dims = output_dims  # Dict mapping output names to shapes
        self.activation_constraints = activation_constraints or {}  # Optional constraints
        
    def is_compatible_with(self, other_interface, connection_point):
        """Check if this interface is compatible with another at the given connection point."""
        if connection_point not in self.output_dims or connection_point not in other_interface.input_dims:
            return False
            
        # Check dimension compatibility
        my_output_shape = self.output_dims[connection_point]
        other_input_shape = other_interface.input_dims[connection_point]
        
        return my_output_shape == other_input_shape
```

Key interface attributes:
- **Named Inputs/Outputs**: Semantic labels for connection points
- **Dimensional Specifications**: Shape requirements for compatible connections
- **Activation Statistics**: Expected distributions and ranges
- **Functional Contract**: Behavioral guarantees provided by the circuit

## Composition Methods

Our framework supports several composition patterns:

### 1. Sequential Composition

Connecting circuits in a feed-forward sequence:

```python
def compose_sequential(circuit_a, circuit_b, connection_mapping):
    """Connect circuit_a outputs to circuit_b inputs according to the mapping."""
    if not validate_sequential_compatibility(circuit_a, circuit_b, connection_mapping):
        raise IncompatibleCircuitsError("Circuits cannot be sequentially composed")
        
    composed_circuit = ComposedCircuit()
    
    # Add both circuits to the composition
    composed_circuit.add_subcircuit("A", circuit_a)
    composed_circuit.add_subcircuit("B", circuit_b)
    
    # Create the connections
    for output_name, input_name in connection_mapping.items():
        composed_circuit.connect(
            ("A", output_name),
            ("B", input_name)
        )
        
    return composed_circuit
```

### 2. Parallel Composition

Running multiple circuits side-by-side:

```python
def compose_parallel(circuits, shared_inputs=None, merged_outputs=None):
    """Compose circuits to run in parallel, optionally sharing inputs and merging outputs."""
    composed_circuit = ComposedCircuit()
    
    # Add all subcircuits
    for i, circuit in enumerate(circuits):
        composed_circuit.add_subcircuit(f"Sub{i}", circuit)
    
    # Connect shared inputs if specified
    if shared_inputs:
        for input_name, target_circuits in shared_inputs.items():
            for target_idx, target_input in target_circuits:
                composed_circuit.share_input(
                    input_name,
                    (f"Sub{target_idx}", target_input)
                )
    
    # Configure merged outputs if specified
    if merged_outputs:
        for output_name, source_outputs in merged_outputs.items():
            composed_circuit.merge_outputs(
                output_name,
                [(f"Sub{idx}", output) for idx, output in source_outputs]
            )
    
    return composed_circuit
```

### 3. Recurrent Composition

Creating feedback loops between circuits:

```python
def compose_recurrent(circuit, feedback_connections, max_iterations=10):
    """Create a recurrent composition where outputs feed back into inputs."""
    if not validate_recurrent_compatibility(circuit, feedback_connections):
        raise IncompatibleCircuitsError("Circuit cannot be recurrently composed")
        
    composed_circuit = RecurrentComposedCircuit(max_iterations)
    
    # Add the circuit to be recurrently connected
    composed_circuit.add_subcircuit("Main", circuit)
    
    # Create the feedback connections
    for output_name, input_name in feedback_connections.items():
        composed_circuit.add_feedback(
            ("Main", output_name),
            ("Main", input_name)
        )
        
    return composed_circuit
```

### 4. Hierarchical Composition

Organizing circuits into hierarchical structures:

```python
def compose_hierarchical(subcircuits, connections, interface_mapping):
    """Create a hierarchical composition of subcircuits."""
    hierarchical_circuit = HierarchicalCircuit()
    
    # Add all subcircuits
    for name, circuit in subcircuits.items():
        hierarchical_circuit.add_subcircuit(name, circuit)
    
    # Create internal connections
    for (source_name, source_output), (target_name, target_input) in connections:
        hierarchical_circuit.connect(
            (source_name, source_output),
            (target_name, target_input)
        )
    
    # Define the external interface
    for external_name, (circuit_name, internal_name) in interface_mapping.items():
        hierarchical_circuit.map_interface(external_name, (circuit_name, internal_name))
    
    return hierarchical_circuit
```

## Adaptation Mechanisms

Our system includes several techniques to adapt circuits for composition:

### 1. Activation Adapters

Small learned networks that transform activations between incompatible circuits:

```python
class ActivationAdapter:
    def __init__(self, source_dims, target_dims, adapter_type="mlp"):
        self.source_dims = source_dims
        self.target_dims = target_dims
        self.adapter_type = adapter_type
        self.adapter_network = self._create_adapter_network()
        
    def _create_adapter_network(self):
        """Create the appropriate adapter network based on type and dimensions."""
        if self.adapter_type == "mlp":
            return MLP(
                input_size=np.prod(self.source_dims),
                hidden_sizes=[100, 100],
                output_size=np.prod(self.target_dims),
                activation=nn.ReLU()
            )
        elif self.adapter_type == "linear":
            return nn.Linear(np.prod(self.source_dims), np.prod(self.target_dims))
        # Other adapter types...
        
    def transform(self, source_activation):
        """Transform activations from source to target dimensions."""
        # Flatten, transform, reshape
        flattened = source_activation.reshape(-1)
        transformed = self.adapter_network(flattened)
        return transformed.reshape(self.target_dims)
        
    def train(self, source_examples, target_examples, optimizer, loss_fn, epochs=100):
        """Train the adapter to map between source and target examples."""
        # Training implementation
```

### 2. Distribution Matching

Ensuring activation distributions are compatible between circuits:

```python
def match_distributions(source_circuit, target_circuit, connection_point, calibration_dataset):
    """Create a distribution matching function for connecting circuits."""
    # Collect activation statistics
    source_activations = collect_activations(source_circuit, calibration_dataset, connection_point)
    target_activations = collect_activations(target_circuit, calibration_dataset, connection_point)
    
    # Compute source statistics
    source_mean = np.mean(source_activations, axis=0)
    source_std = np.std(source_activations, axis=0)
    
    # Compute target statistics
    target_mean = np.mean(target_activations, axis=0)
    target_std = np.std(target_activations, axis=0)
    
    # Create transformation function
    def transform_activation(activation):
        # Normalize to z-scores using source statistics
        normalized = (activation - source_mean) / (source_std + 1e-8)
        # Scale to target distribution
        return normalized * target_std + target_mean
        
    return transform_activation
```

### 3. Prompting-Based Adapters

Using LLMs to generate adapter code:

```python
def generate_adapter_with_llm(source_circuit, target_circuit, connection_point, examples, llm):
    """Use an LLM to generate adapter code based on examples."""
    # Create prompt with context and examples
    prompt = f"""
    I need to create an adapter function that converts the output from 
    "{source_circuit.name}" at "{connection_point}" to be compatible with 
    the input of "{target_circuit.name}" at the same connection point.
    
    Source circuit output: {source_circuit.interface.output_dims[connection_point]}
    Target circuit input: {target_circuit.interface.input_dims[connection_point]}
    
    Here are some example pairs of activations (source → target):
    """
    
    for source_act, target_act in examples:
        prompt += f"\nSource: {source_act}\nTarget: {target_act}\n"
    
    prompt += "\nWrite a Python function that performs this conversion:"
    
    # Get code from LLM
    adapter_code = llm.generate_code(prompt)
    
    # Compile and return the function
    adapter_function = compile_function_from_code(adapter_code)
    return adapter_function
```

## LLM-Assisted Composition

Our framework leverages LLM capabilities to aid circuit composition:

### 1. Interface Inference

```python
def infer_circuit_interface(circuit_code, circuit_name, llm):
    """Use LLM to infer a circuit's interface from its code."""
    prompt = f"""
    Analyze the following neural circuit code and identify its interface.
    Determine the input/output tensor dimensions and semantics.
    
    Circuit name: {circuit_name}
    
    Code:
    {circuit_code}
    
    Return a JSON object with the following structure:
    {{
        "inputs": [
            {{"name": "input_name", "dimensions": [dim1, dim2, ...], "semantics": "description"}}
        ],
        "outputs": [
            {{"name": "output_name", "dimensions": [dim1, dim2, ...], "semantics": "description"}}
        ]
    }}
    """
    
    interface_json = llm.generate_json(prompt)
    return CircuitInterface.from_json(interface_json)
```

### 2. Composition Planning

```python
def plan_composition(available_circuits, target_description, llm):
    """Use LLM to plan how to compose available circuits to achieve a target function."""
    # Create circuit catalog
    circuit_catalog = "\n".join([
        f"Circuit: {circuit.name}\n"
        f"Function: {circuit.description}\n"
        f"Inputs: {circuit.interface.input_dims}\n"
        f"Outputs: {circuit.interface.output_dims}\n"
        for circuit in available_circuits
    ])
    
    prompt = f"""
    I have the following neural circuits available:
    
    {circuit_catalog}
    
    I want to create a composed circuit that: {target_description}
    
    Provide a composition plan with the following:
    1. Which circuits to use
    2. How to connect them (sequential, parallel, recurrent, or hierarchical)
    3. Any adapter functions needed for compatibility
    4. The expected functionality of the final composed circuit
    
    Return the plan in JSON format.
    """
    
    composition_plan = llm.generate_json(prompt)
    return composition_plan
```

### 3. Adapter Generation

```python
def generate_adapter(source_circuit, target_circuit, connection_point, llm):
    """Use LLM to generate an adapter between two circuits."""
    prompt = f"""
    I need to connect the output '{connection_point}' from {source_circuit.name} 
    to the input '{connection_point}' of {target_circuit.name}.
    
    Source output shape: {source_circuit.interface.output_dims[connection_point]}
    Target input shape: {target_circuit.interface.input_dims[connection_point]}
    
    Source circuit function: {source_circuit.description}
    Target circuit function: {target_circuit.description}
    
    Generate Python code for an adapter function that will transform the source output
    to be compatible with the target input. The function should have the signature:
    
    def adapter(source_activation):
        # transformation code
        return transformed_activation
    """
    
    adapter_code = llm.generate_code(prompt)
    adapter_function = compile_function_from_code(adapter_code)
    return adapter_function
```

## Testing and Validation

Our framework includes tools for testing composed circuits:

### Functional Testing

```python
def test_composed_circuit(composed_circuit, test_cases):
    """Test a composed circuit against expected behaviors."""
    results = []
    
    for test_case in test_cases:
        # Run the circuit on the test input
        actual_output = composed_circuit(test_case.input)
        
        # Check if output matches expected
        success = is_output_matching(actual_output, test_case.expected_output)
        
        results.append({
            "test_case": test_case.id,
            "success": success,
            "expected": test_case.expected_output,
            "actual": actual_output,
            "error_margin": calculate_error(actual_output, test_case.expected_output)
        })
    
    return results
```

### Composition Validation

```python
def validate_composition(composition_plan, available_circuits):
    """Validate that a composition plan is feasible with the available circuits."""
    validation_results = {
        "is_valid": True,
        "issues": []
    }
    
    # Check that all referenced circuits exist
    for circuit_ref in composition_plan.get_circuit_references():
        if not circuit_exists(circuit_ref, available_circuits):
            validation_results["is_valid"] = False
            validation_results["issues"].append(f"Referenced circuit '{circuit_ref}' not found")
    
    # Check interface compatibility
    for connection in composition_plan.get_connections():
        source_circuit = get_circuit(connection.source_circuit, available_circuits)
        target_circuit = get_circuit(connection.target_circuit, available_circuits)
        
        if not are_interfaces_compatible(
            source_circuit, target_circuit, 
            connection.source_point, connection.target_point
        ):
            validation_results["is_valid"] = False
            validation_results["issues"].append(
                f"Incompatible connection from {connection.source_circuit}.{connection.source_point} "
                f"to {connection.target_circuit}.{connection.target_point}"
            )
    
    return validation_results
```

## Implementation Status

Current status of implementation:

- **Basic Composition Framework**: Implemented (v0.7)
- **Sequential & Parallel Composition**: Implemented (v0.8)
- **Recurrent & Hierarchical Composition**: Prototype (v0.4)
- **LLM-Assisted Composition**: Early development (v0.3)
- **Validation Tools**: Design phase (v0.2)

## Future Directions

Our roadmap for the composition framework:

1. **Self-Adaptive Interfaces**: Circuits that automatically adapt to connection partners
2. **Composition Verification**: Formal verification of circuit compositions
3. **Dynamic Reconfiguration**: Runtime modification of circuit connections
4. **Multi-Objective Optimization**: Optimizing compositions for multiple competing objectives 
5. **Learned Composition Rules**: Data-driven approaches to discovering effective composition patterns 