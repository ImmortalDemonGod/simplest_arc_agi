# Self-Modeling Visualization Tools

This set of scripts allows you to train and visualize transformer models with self-modeling applied at different layers. The visualization helps understand how the location of self-modeling affects training dynamics and model performance. The scripts use an industry-standard 70/15/15 train/validation/test split.

## Dataset Split Roles

The three dataset splits serve distinct purposes in the training and evaluation process:

1. **Training Set (70%)**: Used to train the model parameters through backpropagation. The model learns patterns from this data.

2. **Validation Set (15%)**: Used for:
   - Early stopping to prevent overfitting
   - Hyperparameter tuning
   - Model selection (choosing the best checkpoint)
   - Monitoring generalization during training
   - The validation set is not used for gradient updates, only for evaluation

3. **Test Set (15%)**: Used only for final evaluation to assess how well the model generalizes to completely unseen data. The test set is never used for model selection or training decisions.

This separation ensures unbiased evaluation of model performance and helps prevent overfitting.

## Implementation Details

In our implementation:

1. **Early Stopping**: The validation accuracy is used to determine when to stop training if the `use_val_for_stopping` flag is enabled. This prevents overfitting by stopping training when performance on the validation set no longer improves.

2. **Model Selection**: The best model checkpoint is saved based on validation accuracy rather than training accuracy, ensuring we select models that generalize well.

3. **Target Accuracy**: When a target accuracy is specified (e.g., 80%), the system can use either validation or test accuracy to determine when this threshold has been reached.

4. **Visualization**: The validation metrics are plotted alongside training and test metrics, allowing for easy comparison of how well the model generalizes during training.

This approach provides a more robust evaluation of different self-modeling techniques by ensuring we're measuring their true generalization capabilities rather than just their ability to memorize the training data.

## Scripts Overview

1. **compare_self_modeling_locations.py**: Trains multiple transformer models with self-modeling applied at different layers:
   - Baseline (no self-modeling)
   - First layer (layer_0)
   - Middle layer
   - Last hidden layer
   - All layers (first, middle, and last layers simultaneously)

2. **visualize_self_modeling_results.py**: Creates visualizations from saved checkpoints, comparing metrics like:
   - Test/train accuracy
   - Test/train loss
   - Weight standard deviation
   - Real Log Canonical Threshold (RLCT)
   - Self-modeling loss

3. **run_self_modeling_comparison.py**: A convenience script that runs both training and visualization in one go.

## Usage

### Option 1: Run the complete pipeline

```bash
python run_self_modeling_comparison.py
```

This will:
1. Train all model variants (baseline, first_layer, middle_layer, last_layer, all_layers)
2. Generate comparison visualizations
3. Save results to `checkpoints/self_modeling_comparison/`

### Option 2: Run with custom parameters

```bash
python run_self_modeling_comparison.py --max_epochs 200 --samples 10000 --target_accuracy 0.8 --seed 42 --base_dir my_results
```

Parameters:
- `--max_epochs`: Maximum number of training epochs (default: 150)
- `--samples`: Number of training samples to generate (default: 5000)
- `--target_accuracy`: Stop training when this test accuracy is reached (e.g., 0.8 for 80%)
- `--seed`: Random seed for reproducibility (default: 42)
- `--base_dir`: Directory to save results (default: checkpoints/self_modeling_comparison)
- `--skip_training`: Skip training and only run visualization on existing checkpoints
- `--show_plots`: Display plots after generating them

### Option 3: Run training and visualization separately

Train models:
```bash
python compare_self_modeling_locations.py --max_epochs 200 --samples 10000 --target_accuracy 0.8 --seed 42 --base_dir my_results
```

Visualize results:
```bash
python visualize_self_modeling_results.py --base_dir my_results
```

## Understanding the Results

The visualization generates several plots:

1. **Test/Train/Validation Accuracy**: Shows how quickly each model variant learns
2. **Test/Train/Validation Loss**: Shows how the loss decreases during training
3. **Weight Standard Deviation**: Indicates how the weight distribution changes
4. **RLCT (Real Log Canonical Threshold)**: A measure of model complexity
5. **Self-Modeling Loss**: Shows how well each model predicts its own activations
6. **Learning Dynamics**: Plots accuracy vs. loss to show learning trajectories for both test and validation sets

## Key Insights

By comparing these visualizations, you can observe:

1. How self-modeling affects convergence speed
2. Whether applying self-modeling to different layers has different effects
3. How self-modeling influences the model's internal representations
4. The relationship between self-modeling loss and primary task performance
5. How validation metrics compare to test metrics, providing insight into generalization
6. Whether certain self-modeling approaches generalize better than others

### Self-Modeling Evaluation with Validation Data

The validation set is particularly valuable for evaluating self-modeling approaches because:

1. **Overfitting Detection**: Self-modeling introduces additional parameters and complexity. The validation set helps detect if a particular self-modeling approach is causing the model to overfit.
2. **Generalization Assessment**: Different self-modeling locations (first layer, middle layer, last layer) or combinations (all layers) may affect generalization differently. The validation metrics help identify which approach generalizes best.
2. **Generalization Assessment**: Different self-modeling locations (first layer, middle layer, last layer) may affect generalization differently. The validation metrics help identify which approach generalizes best.

3. **Early Convergence**: Some self-modeling approaches may lead to faster convergence on the validation set, indicating better learning dynamics.

4. **Stability Measurement**: The gap between training and validation metrics can indicate the stability of different self-modeling approaches.

By comparing these metrics across different self-modeling configurations, you can make more informed decisions about where in the model self-modeling is most effective.

This can provide insights into which layer is most beneficial for applying self-modeling in transformer architectures.

## Multi-Layer Self-Modeling

The "all_layers" variant applies self-modeling to multiple layers simultaneously (first, middle, and last). This approach:

1. **Creates multiple self-modeling heads**: Each layer gets its own dedicated prediction head
2. **Captures representations at different abstraction levels**: From low-level features in early layers to high-level features in later layers
3. **Provides a more comprehensive self-modeling signal**: The model must learn to predict its own activations at multiple points in the network

This variant allows us to investigate whether self-modeling benefits are additive across layers or if there are diminishing returns. It also helps determine if certain combinations of layers work better together than individual layers alone.

The implementation uses a dictionary-based approach to track activations and losses for each layer, making it easy to analyze the contribution of each layer to the overall self-modeling performance.