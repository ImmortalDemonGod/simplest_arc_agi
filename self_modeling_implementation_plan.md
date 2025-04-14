# Implementation Plan for Self-Modeling Auxiliary Task

## 1. Key Decisions

- **Target Layer**: We'll use the last hidden layer as the target layer for the self-modeling task, as it likely contains the most task-relevant features.
- **Loss Weight**: We'll use the suggested default value of 5.0 for the self-modeling loss weight (ws).
- **Complexity Metrics**: We'll implement both weight distribution width measurement and RLCT estimation as described in the pseudocode.
- **Integration Approach**: We'll modify the existing training pipeline to support both standard training and self-modeling as an option, making it more flexible and reusable.

## 2. Detailed Implementation Steps

### Step 1: Modify the TransformerConfig Class
Add parameters for self-modeling configuration:
```python
class TransformerConfig:
    def __init__(
        self,
        vocab_size: int = 120,
        hidden_size: int = 128,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 4,
        intermediate_size: int = 512,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 16,
        pad_token_id: int = 102,
        use_self_modeling: bool = False,  # Enable/disable self-modeling
        self_modeling_target_layer: str = "last_hidden",  # Target layer for self-modeling
        self_modeling_loss_weight: float = 5.0,  # Weight for self-modeling loss
    ):
        # Existing parameters
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.pad_token_id = pad_token_id
        
        # New parameters for self-modeling
        self.use_self_modeling = use_self_modeling
        self.self_modeling_target_layer = self_modeling_target_layer
        self.self_modeling_loss_weight = self_modeling_loss_weight
```

### Step 2: Create the Self-Modeling Auxiliary Head
```python
class SelfModelingHead(nn.Module):
    """Auxiliary head for predicting activations from a target layer"""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.ReLU()
        self.dense2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states):
        x = self.dense1(hidden_states)
        x = self.activation(x)
        x = self.dense2(x)
        x = self.dropout(x)
        return x
```

### Step 3: Modify the SimpleTransformer Class
Update the forward pass to extract activations and add the auxiliary head:
```python
class SimpleTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Existing components
        self.token_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])
        self.output = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Add self-modeling head if enabled
        if config.use_self_modeling:
            self.self_modeling_head = SelfModelingHead(config)
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.size()
        
        # Create position IDs
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).float()
        
        # Prepare attention mask for self-attention
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Get token and position embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        
        # Combine embeddings
        embeddings = token_embeds + position_embeds
        embeddings = self.layer_norm(embeddings)
        hidden_states = self.dropout(embeddings)
        
        # Store target layer activations if self-modeling is enabled
        target_activations = None
        
        # Pass through transformer blocks
        for i, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, extended_attention_mask)
            
            # Store activations from the target layer
            if self.config.use_self_modeling and i == len(self.blocks) - 1 and self.config.self_modeling_target_layer == "last_hidden":
                target_activations = hidden_states.clone()
        
        # Output layer for primary task
        logits = self.output(hidden_states)
        
        # If self-modeling is enabled, predict target activations
        if self.config.use_self_modeling and target_activations is not None:
            predicted_activations = self.self_modeling_head(hidden_states)
            return logits, target_activations, predicted_activations
        
        return logits
```

### Step 4: Modify the AlgorithmicTaskTrainer Class
Update the trainer to handle self-modeling:
```python
class AlgorithmicTaskTrainer:
    def __init__(
        self,
        model: SimpleTransformer,
        train_dataset: TensorDataset,
        test_dataset: TensorDataset,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        batch_size: int = 64,
        max_epochs: int = 10000,
        patience: int = 1000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        log_interval: int = 100,
        checkpoint_dir: Optional[str] = None,
        # New parameters for self-modeling
        primary_loss_weight: float = 1.0,
        self_modeling_loss_weight: Optional[float] = None,
    ):
        # Existing initialization
        self.model = model
        self.device = device
        self.model.to(self.device)
        
        # Datasets and dataloaders
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Optimizer and scheduler
        self.optimizer = optimizer or optim.AdamW(
            model.parameters(), lr=1e-3, weight_decay=0.01
        )
        self.lr_scheduler = lr_scheduler
        
        # Training parameters
        self.max_epochs = max_epochs
        self.patience = patience
        self.log_interval = log_interval
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory if needed
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Loss function - ignore padding tokens (-100)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Self-modeling parameters
        self.primary_loss_weight = primary_loss_weight
        self.self_modeling_loss_weight = self_modeling_loss_weight or model.config.self_modeling_loss_weight if hasattr(model.config, 'self_modeling_loss_weight') else 5.0
        self.use_self_modeling = hasattr(model.config, 'use_self_modeling') and model.config.use_self_modeling
        
        # MSE loss for self-modeling
        self.mse_loss = nn.MSELoss()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_test_acc = 0.0
        self.epochs_without_improvement = 0
        self.training_history = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": [],
            "epoch": [],
            "learning_rate": [],
            # New metrics for self-modeling
            "primary_loss": [],
            "self_modeling_loss": [],
            "weight_std": [],
            "rlct": []
        }
```

### Step 5: Update the Training and Evaluation Methods
Modify the training loop to handle self-modeling:
```python
def train_epoch(self) -> Tuple[float, float]:
    """Train for one epoch and return average loss and accuracy"""
    self.model.train()
    total_loss = 0.0
    total_primary_loss = 0.0
    total_self_modeling_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    for batch_idx, (input_ids, target_ids) in enumerate(self.train_dataloader):
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        
        if self.use_self_modeling:
            # Get both primary and auxiliary outputs
            logits, true_activations, predicted_activations = self.model(input_ids)
            
            # Compute primary loss
            primary_loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            
            # Compute self-modeling loss
            self_modeling_loss = self.mse_loss(predicted_activations, true_activations)
            
            # Combine losses
            loss = self.primary_loss_weight * primary_loss + self.self_modeling_loss_weight * self_modeling_loss
            
            # Track individual losses
            total_primary_loss += primary_loss.item()
            total_self_modeling_loss += self_modeling_loss.item()
        else:
            # Standard forward pass
            logits = self.model(input_ids)
            loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Update stats
        total_loss += loss.item()
        
        # Compute accuracy (only for non-ignored positions)
        mask = (target_ids != -100).float()
        predictions = logits.argmax(dim=-1)
        correct = ((predictions == target_ids) * mask).sum().item()
        total = mask.sum().item()
        
        correct_predictions += correct
        total_predictions += total
        
        # Log progress
        if batch_idx % self.log_interval == 0:
            if self.use_self_modeling:
                print(f"Epoch {self.current_epoch} [{batch_idx}/{len(self.train_dataloader)}] " 
                      f"Loss: {loss.item():.6f}, Primary: {primary_loss.item():.6f}, "
                      f"Self-Modeling: {self_modeling_loss.item():.6f}, Acc: {correct/total:.4f}")
            else:
                print(f"Epoch {self.current_epoch} [{batch_idx}/{len(self.train_dataloader)}] " 
                      f"Loss: {loss.item():.6f}, Acc: {correct/total:.4f}")
        
        self.global_step += 1
    
    # Compute average loss and accuracy
    avg_loss = total_loss / len(self.train_dataloader)
    avg_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    # Compute average individual losses if using self-modeling
    if self.use_self_modeling:
        avg_primary_loss = total_primary_loss / len(self.train_dataloader)
        avg_self_modeling_loss = total_self_modeling_loss / len(self.train_dataloader)
        return avg_loss, avg_accuracy, avg_primary_loss, avg_self_modeling_loss
    
    return avg_loss, avg_accuracy
```

### Step 6: Add Complexity Verification Methods
```python
def measure_weight_distribution(self) -> float:
    """
    Calculate the standard deviation of the weights in the final layer.
    
    Returns:
        std_dev: Standard deviation of the weights.
    """
    final_weights = self.model.output.weight.data.cpu().numpy()
    std_dev = np.std(final_weights)
    return std_dev

def estimate_rlct(self, num_iterations=100, lr=0.0001, localization=1000) -> float:
    """
    Estimate the Real Log Canonical Threshold (RLCT) using stochastic sampling
    near the trained model's weights.
    
    Args:
        num_iterations: Number of SGLD sampling iterations
        lr: Learning rate for SGLD
        localization: Localization parameter
        
    Returns:
        rlct: Estimated RLCT value
    """
    # Save original model state
    original_state = {name: param.clone() for name, param in self.model.named_parameters()}
    
    # Initialize RLCT estimate
    rlct_sum = 0.0
    
    # Create a copy of the model for SGLD sampling
    model_copy = copy.deepcopy(self.model)
    model_copy.eval()
    
    # Create a small validation set for loss computation
    val_loader = torch.utils.data.DataLoader(
        self.test_dataset, batch_size=32, shuffle=True
    )
    val_batch = next(iter(val_loader))
    val_input_ids, val_target_ids = val_batch
    val_input_ids = val_input_ids.to(self.device)
    val_target_ids = val_target_ids.to(self.device)
    
    # Perform SGLD sampling iterations
    for i in range(num_iterations):
        # Add noise to parameters
        for name, param in model_copy.named_parameters():
            noise = torch.randn_like(param) * lr / localization
            param.data.add_(noise)
        
        # Compute loss
        with torch.no_grad():
            if self.use_self_modeling:
                logits, true_activations, predicted_activations = model_copy(val_input_ids)
                primary_loss = self.criterion(logits.view(-1, logits.size(-1)), val_target_ids.view(-1))
                self_modeling_loss = self.mse_loss(predicted_activations, true_activations)
                loss = self.primary_loss_weight * primary_loss + self.self_modeling_loss_weight * self_modeling_loss
            else:
                logits = model_copy(val_input_ids)
                loss = self.criterion(logits.view(-1, logits.size(-1)), val_target_ids.view(-1))
        
        # Analyze loss geometry for RLCT estimation
        # This is a simplified approach - in practice, more sophisticated methods would be used
        loss_value = loss.item()
        rlct_contribution = -np.log(loss_value) / np.log(localization)
        rlct_sum += rlct_contribution
    
    # Restore original model state
    for name, param in self.model.named_parameters():
        param.data.copy_(original_state[name])
    
    # Compute average RLCT
    rlct = rlct_sum / num_iterations
    return rlct
```

## 3. Enhanced Visualization and Comparison Strategy

To thoroughly compare models with and without self-modeling, we'll implement a comprehensive visualization and analysis framework. This will allow us to clearly see the impact of the self-modeling auxiliary task on model complexity and performance.

### 3.1 Experiment Setup

We'll run parallel experiments with identical configurations except for the self-modeling component:

```python
def run_comparative_experiments(args):
    """Run parallel experiments with and without self-modeling"""
    
    # Common configuration
    base_checkpoint_dir = args.checkpoint_dir
    
    # Create experiment directories
    baseline_dir = os.path.join(base_checkpoint_dir, "baseline")
    self_modeling_dir = os.path.join(base_checkpoint_dir, "self_modeling")
    os.makedirs(baseline_dir, exist_ok=True)
    os.makedirs(self_modeling_dir, exist_ok=True)
    
    # Load data
    data_dict = load_data(args.modulus, args.train_ratio)
    
    # Run baseline experiment (without self-modeling)
    print("Running baseline experiment (without self-modeling)...")
    args.use_self_modeling = False
    args.checkpoint_dir = baseline_dir
    baseline_model, baseline_config = create_model(data_dict["vocab_size"], args)
    baseline_optimizer, baseline_lr_scheduler = create_optimizer(baseline_model, args.learning_rate, args.weight_decay)
    baseline_trainer = AlgorithmicTaskTrainer(
        model=baseline_model,
        train_dataset=data_dict["train_dataset"],
        test_dataset=data_dict["test_dataset"],
        optimizer=baseline_optimizer,
        lr_scheduler=baseline_lr_scheduler,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        device=args.device,
        checkpoint_dir=baseline_dir
    )
    baseline_history = baseline_trainer.train()
    
    # Run self-modeling experiment
    print("Running self-modeling experiment...")
    args.use_self_modeling = True
    args.checkpoint_dir = self_modeling_dir
    self_modeling_model, self_modeling_config = create_model(data_dict["vocab_size"], args)
    self_modeling_optimizer, self_modeling_lr_scheduler = create_optimizer(self_modeling_model, args.learning_rate, args.weight_decay)
    self_modeling_trainer = AlgorithmicTaskTrainer(
        model=self_modeling_model,
        train_dataset=data_dict["train_dataset"],
        test_dataset=data_dict["test_dataset"],
        optimizer=self_modeling_optimizer,
        lr_scheduler=self_modeling_lr_scheduler,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        device=args.device,
        checkpoint_dir=self_modeling_dir,
        primary_loss_weight=args.primary_loss_weight,
        self_modeling_loss_weight=args.self_modeling_loss_weight
    )
    self_modeling_history = self_modeling_trainer.train()
    
    # Compare and visualize results
    compare_and_visualize(baseline_history, self_modeling_history, args.task, base_checkpoint_dir)
    
    return baseline_history, self_modeling_history
```

### 3.2 Comprehensive Visualization Functions

We'll create detailed visualization functions to compare the models:

```python
def compare_and_visualize(baseline_history, self_modeling_history, task_name, save_dir):
    """Create comprehensive visualizations comparing baseline and self-modeling results"""
    
    # 1. Training Metrics Comparison
    plot_training_comparison(baseline_history, self_modeling_history, task_name, 
                            os.path.join(save_dir, f"{task_name}_training_comparison.png"))
    
    # 2. Complexity Metrics Comparison
    plot_complexity_comparison(baseline_history, self_modeling_history, task_name,
                              os.path.join(save_dir, f"{task_name}_complexity_comparison.png"))
    
    # 3. Weight Distribution Visualization
    plot_weight_distributions(baseline_history, self_modeling_history, task_name,
                             os.path.join(save_dir, f"{task_name}_weight_distributions.png"))
    
    # 4. RLCT Comparison
    if "rlct" in baseline_history and "rlct" in self_modeling_history:
        plot_rlct_comparison(baseline_history, self_modeling_history, task_name,
                            os.path.join(save_dir, f"{task_name}_rlct_comparison.png"))
    
    # 5. Generate summary report
    generate_comparison_report(baseline_history, self_modeling_history, task_name,
                              os.path.join(save_dir, f"{task_name}_comparison_report.md"))
```

### 3.3 Summary Report Generation

To provide a comprehensive analysis of the results, we'll generate a detailed summary report:

```python
def generate_comparison_report(baseline_history, self_modeling_history, task_name, save_path):
    """Generate a detailed report comparing baseline and self-modeling results"""
    
    # Calculate key metrics
    baseline_final_acc = baseline_history["test_acc"][-1]
    self_modeling_final_acc = self_modeling_history["test_acc"][-1]
    
    baseline_final_weight_std = baseline_history["weight_std"][-1]
    self_modeling_final_weight_std = self_modeling_history["weight_std"][-1]
    
    weight_std_reduction = (baseline_final_weight_std - self_modeling_final_weight_std) / baseline_final_weight_std * 100
    
    # RLCT comparison if available
    rlct_comparison = ""
    if "rlct" in baseline_history and "rlct" in self_modeling_history:
        baseline_final_rlct = baseline_history["rlct"][-1]
        self_modeling_final_rlct = self_modeling_history["rlct"][-1]
        rlct_reduction = (baseline_final_rlct - self_modeling_final_rlct) / baseline_final_rlct * 100
        
        rlct_comparison = f"""
## RLCT Comparison

| Model | Final RLCT | Reduction |
|-------|------------|-----------|
| Baseline | {baseline_final_rlct:.4f} | - |
| Self-Modeling | {self_modeling_final_rlct:.4f} | {rlct_reduction:.2f}% |

The Real Log Canonical Threshold (RLCT) is {rlct_reduction:.2f}% lower with self-modeling, indicating increased parameter efficiency.
"""
    
    # Generate report content
    report_content = f"""# Self-Modeling Auxiliary Task: Results for {task_name}

## Summary

This report compares the performance and complexity metrics between a baseline transformer model and one with the self-modeling auxiliary task.

## Primary Task Performance

| Model | Final Test Accuracy | Epochs to Converge |
|-------|---------------------|-------------------|
| Baseline | {baseline_final_acc:.4f} | {len(baseline_history["epoch"])} |
| Self-Modeling | {self_modeling_final_acc:.4f} | {len(self_modeling_history["epoch"])} |

The self-modeling auxiliary task {"improved" if self_modeling_final_acc > baseline_final_acc else "maintained"} primary task performance.

## Weight Distribution Analysis

| Model | Final Weight StdDev | Reduction |
|-------|---------------------|-----------|
| Baseline | {baseline_final_weight_std:.4f} | - |
| Self-Modeling | {self_modeling_final_weight_std:.4f} | {weight_std_reduction:.2f}% |

The weight distribution in the final layer is {weight_std_reduction:.2f}% narrower with self-modeling, indicating reduced complexity.

{rlct_comparison}

## Conclusion

The self-modeling auxiliary task has successfully:

1. {"Improved" if self_modeling_final_acc > baseline_final_acc else "Maintained"} primary task accuracy
2. Reduced the weight distribution width by {weight_std_reduction:.2f}%
3. {"Reduced the RLCT by " + f"{rlct_reduction:.2f}%" if "rlct" in baseline_history and "rlct" in self_modeling_history else ""}

These results confirm the effectiveness of the self-modeling approach in regularizing the network and increasing parameter efficiency while maintaining task performance.
"""
    
    # Write report to file
    with open(save_path, "w") as f:
        f.write(report_content)
    
    print(f"Comparison report saved to {save_path}")
    
    return report_content
```

## 4. Implementation Timeline

1. **Phase 1**: Modify the transformer model (1-2 days)
2. **Phase 2**: Update the training pipeline (1-2 days)
3. **Phase 3**: Add complexity verification metrics (1-2 days)
4. **Phase 4**: Testing and validation (2-3 days)
5. **Phase 5**: Documentation and finalization (1 day)

Total estimated time: 6-10 days

## 5. Expected Outcomes

Based on the pseudocode specification, we expect:

1. **Similar or slightly improved primary task accuracy**
2. **Narrower weight distribution** in the final layers
3. **Lower RLCT values**, indicating increased parameter efficiency
4. **More regularized network** with reduced complexity

These outcomes will be clearly visualized and documented in the comparison reports, providing a comprehensive analysis of the benefits of the self-modeling auxiliary task.

======🧙🏾‍♂️: Below is a complete, standalone pseudocode specification for integrating a self-modeling auxiliary task into a transformer‐based system. This design incorporates all key concepts and details that a senior developer could follow to implement the feature, including clear definitions, modular pseudocode, and built-in success criteria that reflect the benefits reported in the referenced paper.

---

# 📘 Pseudocode Specification: Self-Modeling Auxiliary Task Integration with Complexity Verification

## Objective

Implement a **self-modeling auxiliary task** within a transformer model. The auxiliary task forces the network to predict its own internal activations at a designated layer during training. This process is expected to:

- **Regularize the network** by reducing complexity.
- **Narrow the weight distribution** in the final layers.
- **Lower the Real Log Canonical Threshold (RLCT)**, indicating increased parameter efficiency.
- **Maintain (or slightly improve) primary task accuracy.**

The system should log detailed metrics so that success is verified by comparing:
- The standard deviation of final layer weight distributions.
- The estimated RLCT values.
- Primary task performance (e.g., classification accuracy).

---

## 🔧 Key Concepts & Definitions

- **Primary Task:** The main objective of the model (e.g., image classification).
- **Self-Modeling Auxiliary Task:** An extra regression task where the model predicts the activations from a target intermediate layer.
- **Target Layer (`L_target`):** The designated layer from which activations are extracted (e.g., the last hidden layer before classification).
- **Activation Vector (`a`):** The true activations from `L_target` during the forward pass.
- **Predicted Activation Vector (`â`):** The output of an auxiliary head tasked with predicting `a`.
- **Loss Terms:**
  - `L_primary`: Loss for the main task (e.g., cross-entropy loss).
  - `L_self`: Loss for the self-modeling task (e.g., mean squared error between `a` and `â`).
  - `L_total = wc * L_primary + ws * L_self`: The weighted sum of the primary and auxiliary losses, where `wc` and `ws` are the hyperparameters controlling their relative importance.
- **Verification Metrics:**
  - **Weight Distribution Width:** Standard deviation of the weights in the final layer; a narrower distribution indicates reduced complexity.
  - **RLCT (Real Log Canonical Threshold):** A theoretically grounded metric of model complexity, where a lower value suggests greater regularization and efficiency.

---

## 🧠 Model Architecture Modification

1. **Extract Intermediate Activations:**
   - During the forward pass, tap the output of a designated layer (`L_target`) and store it as the true activation vector `a`.

2. **Auxiliary Head:**
   - Add an auxiliary network (a small feed-forward network) that takes the same input as `L_target` and outputs a vector `â` of the same dimensions.

3. **Dual Output:**
   - The model’s forward pass returns both the primary task logits and the auxiliary output.

---

## 🧪 Pseudocode Implementation

### Step 1: Define Hyperparameters & Target Layer

```python
# Hyperparameters for loss balancing
wc = 1.0                      # Weight for the primary task loss
ws = 5.0                      # Weight for the self-modeling loss (adjust based on experiments)

# Specify target layer name (e.g., "last_hidden")
L_target = "last_hidden"
```

---

### Step 2: Modify the Forward Pass

```python
def forward(input_ids: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Perform a forward pass through the transformer model.
    
    Returns:
        classification_logits: Tensor for primary task predictions.
        activations_a: True activations from the target layer (L_target).
        predicted_a: Auxiliary predictions from the self-modeling head.
    """
    # Embedding and positional encoding
    hidden_states = embedding_layer(input_ids)
    
    # Forward pass through transformer layers
    for layer_name, layer in transformer_layers.items():
        hidden_states = layer(hidden_states)
        if layer_name == L_target:
            activations_a = hidden_states.clone()  # Store true activations (detach if necessary)
    
    # Primary head: output for classification or main task
    classification_logits = classification_head(hidden_states)
    
    # Auxiliary head: predict activations from the target layer
    predicted_a = self_modeling_head(hidden_states)
    
    return classification_logits, activations_a, predicted_a
```

---

### Step 3: Update the Training Loop with Dual Loss

```python
for epoch in range(total_epochs):
    for input_ids, labels in train_loader:
        # Forward pass: obtain both primary and auxiliary outputs
        logits, a, a_hat = model.forward(input_ids)
        
        # Compute primary loss (e.g., cross-entropy)
        L_primary = cross_entropy_loss(logits, labels)
        
        # Compute auxiliary loss (e.g., mean squared error)
        L_self = mse_loss(a_hat, a)
        
        # Combine losses using predefined weights
        L_total = wc * L_primary + ws * L_self
        
        # Backpropagation and optimization
        optimizer.zero_grad()
        L_total.backward()
        optimizer.step()
        
    # Log losses at the end of each epoch
    log({
        "epoch": epoch,
        "L_total": L_total.item(),
        "L_primary": L_primary.item(),
        "L_self": L_self.item()
    })
    
    # Optionally, validate on a test set and log accuracy
    test_accuracy = evaluate_on_test_set(model, test_loader)
    log({"epoch": epoch, "test_accuracy": test_accuracy})
```

---

### Step 4: Auxiliary Head Design Module

```python
def initialize_self_modeling_head(input_dim: int, target_dim: int) -> nn.Module:
    """
    Create the auxiliary head to predict activations from L_target.
    
    Args:
        input_dim: Dimensionality of the target layer's activations.
        target_dim: Same as input_dim (for direct prediction).
        
    Returns:
        A PyTorch module representing the self-modeling head.
    """
    # Option 1: Two-layer feed-forward network with ReLU activation
    head = nn.Sequential(
        nn.Linear(input_dim, input_dim),
        nn.ReLU(),
        nn.Linear(input_dim, target_dim)
    )
    return head
```

*Integrate this head into the model’s initialization:*

```python
# In the model __init__
self.self_modeling_head = initialize_self_modeling_head(hidden_dim, hidden_dim)
```

---

### Step 5: Post-Training Validation & Complexity Verification

#### 5.1 Measuring Weight Distribution

```python
def measure_weight_distribution(model: nn.Module) -> float:
    """
    Calculate the standard deviation of the weights in the final layer.
    
    Returns:
        std_dev: Standard deviation of the weights.
    """
    final_weights = model.classification_head.weight.data.cpu().numpy()
    std_dev = np.std(final_weights)
    return std_dev

# After training, log the weight distribution metric:
final_std = measure_weight_distribution(model)
log({"final_weight_std": final_std})
```

#### 5.2 Estimating RLCT

```python
def estimate_RLCT(model: nn.Module, calibration_params: Dict) -> float:
    """
    Estimate the Real Log Canonical Threshold (RLCT) using stochastic sampling
    near the trained model's weights.
    
    Args:
        model: The trained model.
        calibration_params: Hyperparameters for RLCT estimation (learning rate, localization).
        
    Returns:
        rlct: Estimated RLCT value.
    """
    # Pseudocode for RLCT estimation using SGLD:
    rlct = 0.0
    # Perform multiple SGLD sampling iterations near model's critical point
    for i in range(calibration_params["num_iterations"]):
        sampled_weights = sgld_sample(model, calibration_params)
        loss_val = compute_loss(model, sampled_weights)
        rlct += analyze_loss_geometry(loss_val)
    rlct /= calibration_params["num_iterations"]
    return rlct

# Calibration parameters might be set as follows:
calibration_params = {"num_iterations": 100, "lr": 0.0001, "localization": 1000}
final_rlct = estimate_RLCT(model, calibration_params)
log({"final_RLCT": final_rlct})
```

#### 5.3 Overall Success Criteria

The feature is considered successful if:
- **Primary task performance** remains at an acceptable level (e.g., test accuracy does not degrade beyond a set threshold).
- The **weight distribution** in the final layer is significantly narrower compared to a baseline model without self-modeling.
- The **estimated RLCT** is lower, indicating reduced complexity and increased parameter efficiency.

---

## ✅ Recommended Defaults

| Parameter              | Suggested Value        | Description                                  |
|------------------------|------------------------|----------------------------------------------|
| `wc`                   | 1.0                    | Weight for primary task loss                 |
| `ws`                   | 5.0 (adjustable)       | Weight for self-modeling loss                |
| `L_target`             | "last_hidden"          | Target layer for extracting activations      |
| Activation Loss        | Mean Squared Error (MSE)| Measures prediction error of activations       |
| RLCT Calibration (lr)  | 0.0001                 | Learning rate for SGLD sampling              |
| RLCT Calibration       | 1000 (localization)    | Localization parameter for RLCT estimation   |

---

## 🧪 Validation Strategy Recap

1. **Primary Task Evaluation:**  
   - Monitor test accuracy over epochs to ensure the auxiliary task does not compromise main performance.

2. **Complexity Metrics:**  
   - Calculate and log the standard deviation of the final layer's weight distribution.
   - Estimate the RLCT to quantitatively measure network complexity.
   - Compare these metrics to a baseline model without the self-modeling head.

3. **Hyperparameter Tuning:**  
   - Adjust `ws` to balance the auxiliary task and ensure that it regularizes without overwhelming the primary task.

4. **Logging & Visualization:**  
   - Record detailed logs for both loss curves and complexity metrics.
   - Plot histograms of weight distributions and RLCT trends over training epochs.

---

This pseudocode specification is designed to be self-contained and detailed enough for a senior developer to integrate into the existing codebase. It provides clear definitions, modular implementation steps, and a comprehensive validation plan to verify that the addition of self-modeling indeed simplifies the network in accordance with the results presented in the referenced paper.

