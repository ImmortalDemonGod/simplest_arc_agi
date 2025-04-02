# ARC Evaluation Protocol

This document outlines our approach for evaluating the Neural Circuit Extraction and Modular Composition Framework in alignment with the principles of the Abstraction and Reasoning Corpus (ARC) and François Chollet's insights on general intelligence from "On the Measure of Intelligence."

## Principles of ARC and Intelligence Measurement

ARC evaluates systems based on their ability to solve novel, abstract reasoning tasks from just a few examples. Following Chollet's paper, true intelligence should be measured by:

1. **Skill-Acquisition Efficiency**: Intelligence is the efficiency with which a system can turn experience (training data) into skill, not just the final skill level achieved.
2. **Core Knowledge**: Human-like systems should rely on similar prior knowledge (objectness, basic physics, number sense, geometry, etc.) rather than extensive domain-specific training.
3. **Few-Shot Generalization**: Systems should learn from very few examples (3-5 input/output pairs).
4. **Developer-Aware Evaluation**: The evaluation tasks must be novel to both the system and its developers.

## Dedicated Evaluation Dataset

We've created a specialized evaluation framework:

### Dataset Creation
- **Private Test Set**: 100-200+ ARC-like tasks created specifically for evaluation
- **Novel Compositions**: Tasks focus on novel combinations of Core Knowledge primitives
- **Strict Isolation**: Tasks must not overlap with those used for training models or developing the circuit extraction/composition mechanisms
- **Secrecy**: The dataset remains strictly private until the moment of evaluation

### Task Format
Tasks follow the ARC format, with each task consisting of:
- Multiple demonstration pairs (input/output grid examples)
- One or more test inputs requiring correct predictions

## Standardized Solver Interface

Our evaluation uses a standardized interface:

```python
def solve_arc_task(
    example_pairs: List[Tuple[Grid, Grid]],  # Demonstration pairs
    test_inputs: List[Grid]                  # Test inputs for evaluation
) -> List[Optional[Grid]]:                   # Predictions or None for failures
```

The system receives all demonstration pairs and test inputs for a task simultaneously, with no intermediate feedback.

## Evaluation Protocol

Our protocol mirrors ARC's approach:

1. For each task in the private evaluation set:
   - Call `solve_arc_task` with the demonstration pairs and test inputs
   - The solver processes the examples and makes predictions
   - No feedback is provided on test input predictions

2. A task is considered solved only if the function returns the exact correct output grid for all test inputs.

## Primary Evaluation Metrics

We track several key metrics:

### Primary Metric
- **Accuracy**: Percentage of tasks solved correctly in the private evaluation set

### Secondary Metrics
- **Few-Shot Learning Curve**: Performance as a function of example count (1, 2, 3...)
- **Trial Efficiency**: Average number of hypotheses generated/tested before finding the correct one
- **Resource Consumption**: LLM token usage, circuit calls, and wall-clock time per task
- **Solution Complexity**: Complexity of the generated circuit compositions

## Addressing the "Priors" Problem

To ensure fair evaluation aligned with Core Knowledge assumptions:

### Circuit Library Constraints
- **Core Knowledge Tagging**: All circuits are tagged with their alignment to core cognitive capacities
- **Strict Circuit Access**: During evaluation, the system can only access circuits tagged as Core Knowledge primitives
- **Prior Penalties**: Non-Core or Compound circuits may incur penalties in hypothesis selection

### LLM Constraints
- **Structured Prompting**: LLMs are explicitly instructed to rely only on Core Knowledge primitives
- **External Knowledge Prohibition**: Instructions forbid leveraging external knowledge beyond examples
- **Knowledge Auditing**: LLM interactions are logged and audited for suspicious domain knowledge

## Rule Inference Mechanism

Our framework implements a specific approach to inferring rules from few examples:

1. **Feature Extraction**: For each example pair, we run analyzer circuits to extract features
2. **Differential Analysis**: We identify consistent changes between input and output grids
3. **Hypothesis Generation**: The LLM suggests potential transformations based on the analysis
4. **Circuit Query**: The system queries the database for relevant primitive circuits
5. **Candidate Generation**: The LLM generates candidate compositions to implement the hypothesized rule
6. **Verification**: Candidates are tested against example pairs
7. **Selection**: The simplest viable hypothesis is selected and applied to test inputs

## Developer-Aware Generalization

To maintain integrity of the evaluation:

- **Strict Task Isolation**: Tasks are generated after system freeze (training complete, circuits extracted)
- **LLM Blinding**: No task identifiers or metadata that could link to online ARC discussions
- **Interaction Auditing**: All LLM prompts and responses are logged and checked for suspicious knowledge
- **Training Data Cutoffs**: We prefer LLMs with documented training data cutoffs
- **Human Benchmarking**: Human participants solve the same tasks to establish fair comparison baselines

## Implementation Phases

Our evaluation protocol will be implemented in these phases:

1. **Basic Task Validation**: Initial testing with simple ARC-style tasks (currently implemented)
2. **Few-Shot Learning Optimization**: Fine-tuning the LLM-circuit composition interface (in progress)
3. **Full Protocol Implementation**: Complete system with all constraints and metrics (planned)
4. **Human Comparative Studies**: Side-by-side evaluation with human participants (future)

## Alignment with Chollet's Framework

Our approach aligns with Chollet's formal definition of intelligence:
- `Span(ARC)` represents the space of problems constructed from Core Knowledge
- `Spec(ARC)` is measured by our ARC-aligned evaluation set
- `C(S, .)` is approximated by our metrics on hypothesis complexity
- `Scope` is measured by the breadth of tasks our system can solve
- `Generality` is assessed through the few-shot learning curve 