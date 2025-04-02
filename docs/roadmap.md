# Implementation Roadmap

This roadmap outlines our phased approach to developing the Neural Circuit Extraction and Modular Composition Framework.

## Phase 1: Foundation & Baseline (Months 1-3)

In this initial phase, we establish the core infrastructure and validate the basic approach:

### Data Generator v1
- Implement core `arc_types` functions for basic operations
- Develop generator for binary operation datasets (modular arithmetic)
- Create basic validation and testing procedures

### Transformer Prototype v1
- Build small baseline transformer (2-layer, 128-width)
- Incorporate FlashAttention for performance optimization
- Validate on simple tasks like `add_mod_97`

### Training Framework v1
- Set up basic infrastructure using Ray + W&B for experiment tracking
- Implement tools for launching, monitoring, and logging single training jobs
- Tune hyperparameters for efficient generalization (grokking) on baseline tasks

**Milestones:**
- ✅ Successfully train models that exhibit grokking on simple algorithmic tasks
- ✅ Basic training infrastructure operational
- ✅ Demonstrable generalization on held-out test data

## Phase 2: Parallelization, Efficiency & Optimization (Months 4-6)

In this phase, we focus on scaling our infrastructure and improving model efficiency:

### Concurrent Training
- Scale the framework to manage multiple concurrent training jobs
- Implement resource management across different tasks and hyperparameter settings
- Develop tools for comparing results across experiments

### Efficiency Techniques
- Integrate and systematically evaluate model pruning
- Implement LoRA adapters for efficient fine-tuning
- Analyze impact on generalization speed, final accuracy, parameter count, and inference latency

### HPO Integration
- Integrate automated hyperparameter optimization tools (Optuna, Ray Tune)
- Develop search spaces and optimization metrics
- Validate HPO effectiveness across different tasks

**Milestones:**
- ✅ Demonstrate concurrent training of 10+ experiments
- ✅ Achieve 30%+ parameter reduction with minimal accuracy loss through pruning
- ✅ Show successful task-specific adaptation using LoRA

## Phase 3: Extraction, Cataloging & Visualization (Months 7-9)

This phase focuses on extracting and understanding the circuits within our trained models:

### Circuit Extraction v1
- Implement CLT/dictionary learning pipeline
- Develop attribution graph generation tools
- Extract and analyze circuits from baseline models (and pruned/LoRA variants)

### Database v1
- Design and implement the database schema with `interface_definition`
- Create a functional system for storing task, model, training, and circuit data
- Develop APIs for querying and retrieving circuits

### Visualization v1
- Develop tools for visualizing loss/accuracy curves
- Create interactive attribution graph visualizers
- Implement feature activation pattern visualization

**Milestones:**
- ✅ Successfully extract interpretable circuits from at least 5 different algorithmic tasks
- ✅ Operational database with 20+ cataloged circuits
- ✅ Interactive visualization tools for exploring circuits

## Phase 4: Modular Composition R&D (Months 10-12+)

In this phase, we focus on composing circuits to solve more complex tasks:

### Interface Definition & Standardization
- Analyze extracted representations to understand interface characteristics
- Define standards/methods for circuit interfaces
- Document interface requirements in the database

### Composition Engine v0.1
- Develop prototype methods for composing 2-3 circuits functionally
- Implement hardcoded static pipelines for testing
- Validate on slightly more complex tasks (e.g., `(a+b)*c`)

### Code Representation v0.1
- Experiment with representing compositions in a code-like format
- Develop a simple DSL (Domain-Specific Language) for circuit composition
- Create parsers/interpreters for the composition language

### Validation & Intervention
- Develop methods to validate composed circuits
- Implement targeted testing and intervention experiments
- Create tools for patching intermediate results

### LLM Integration (Exploratory)
- Begin experiments using LLMs to query the circuit database
- Test LLM's ability to suggest compositions based on natural language descriptions
- Explore prompting strategies for effective LLM assistance

**Milestones:**
- ✅ Demonstrate successful composition of 2+ circuits to solve a new task
- ✅ Working prototype of code-like representation for compositions
- ✅ Initial proof-of-concept for LLM-assisted composition

## Phase 5: Scaling, Refinement & Automation (Ongoing)

This ongoing phase focuses on expanding capabilities and increasing automation:

### Expanded Task Library
- Add more diverse and complex algorithmic problems
- Generate larger datasets for more challenging tasks
- Develop curriculum learning approaches

### Model Scaling
- Scale model sizes as needed for more complex tasks
- Optimize training infrastructure for larger models
- Investigate scaling relationships for algorithmic learning

### Advanced Extraction
- Refine circuit extraction techniques for better interpretability
- Improve fidelity of extracted circuits
- Develop automated circuit discovery methods

### Sophisticated Composition
- Develop dynamic routing mechanisms
- Implement automated interface adaptation
- Create tools for handling errors and uncertainty in compositions

### Enhanced LLM Integration
- Deepen LLM integration for automated composition
- Implement verification of LLM-suggested compositions
- Explore LLM generation of glue code for interfaces

**Milestones:**
- Demonstrate solving complex algorithmic tasks beyond individual model capacity
- Achieve high automation in circuit extraction and composition
- Show robust performance across a diverse set of tasks 