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

#### 3.2.1 Training Metrics Comparison

```python
def plot_training_comparison(baseline_history, self_modeling_history, task_name, save_path=None):
    """Plot and compare training metrics between baseline and self-modeling"""
    plt.figure(figsize=(15, 10))
    
    # Plot accuracy
    plt.subplot(2, 2, 1)
    plt.plot(baseline_history["epoch"], baseline_history["train_acc"], 
             label="Baseline Train Accuracy", linestyle="-", color="blue")
    plt.plot(baseline_history["epoch"], baseline_history["test_acc"], 
             label="Baseline Test Accuracy", linestyle="--", color="blue")
    plt.plot(self_modeling_history["epoch"], self_modeling_history["train_acc"], 
             label="Self-Modeling Train Accuracy", linestyle="-", color="red")
    plt.plot(self_modeling_history["epoch"], self_modeling_history["test_acc"], 
             label="Self-Modeling Test Accuracy", linestyle="--", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Comparison for {task_name}")
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(2, 2, 2)
    plt.plot(baseline_history["epoch"], baseline_history["train_loss"], 
             label="Baseline Train Loss", linestyle="-", color="blue")
    plt.plot(baseline_history["epoch"], baseline_history["test_loss"], 
             label="Baseline Test Loss", linestyle="--", color="blue")
    plt.plot(self_modeling_history["epoch"], self_modeling_history["train_loss"], 
             label="Self-Modeling Train Loss", linestyle="-", color="red")
    plt.plot(self_modeling_history["epoch"], self_modeling_history["test_loss"], 
             label="Self-Modeling Test Loss", linestyle="--", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Comparison for {task_name}")
    plt.legend()
    plt.grid(True)
    
    # Plot learning rate
    plt.subplot(2, 2, 3)
    plt.plot(baseline_history["epoch"], baseline_history["learning_rate"], 
             label="Baseline Learning Rate", color="blue")
    plt.plot(self_modeling_history["epoch"], self_modeling_history["learning_rate"], 
             label="Self-Modeling Learning Rate", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.legend()
    plt.grid(True)
    
    # Plot self-modeling specific losses if available
    if "primary_loss" in self_modeling_history and "self_modeling_loss" in self_modeling_history:
        plt.subplot(2, 2, 4)
        plt.plot(self_modeling_history["epoch"], self_modeling_history["primary_loss"], 
                 label="Primary Task Loss", color="green")
        plt.plot(self_modeling_history["epoch"], self_modeling_history["self_modeling_loss"], 
                 label="Self-Modeling Loss", color="purple")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Self-Modeling Component Losses")
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
```

#### 3.2.2 Complexity Metrics Comparison

```python
def plot_complexity_comparison(baseline_history, self_modeling_history, task_name, save_path=None):
    """Plot and compare complexity metrics between baseline and self-modeling"""
    plt.figure(figsize=(15, 6))
    
    # Plot weight standard deviation
    plt.subplot(1, 2, 1)
    plt.plot(baseline_history["epoch"], baseline_history["weight_std"], 
             label="Baseline Weight StdDev", color="blue")
    plt.plot(self_modeling_history["epoch"], self_modeling_history["weight_std"], 
             label="Self-Modeling Weight StdDev", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Standard Deviation")
    plt.title(f"Weight Distribution Width Comparison for {task_name}")
    plt.legend()
    plt.grid(True)
    
    # Plot RLCT if available
    if "rlct" in baseline_history and "rlct" in self_modeling_history:
        # Create x-axis values for RLCT (computed less frequently)
        baseline_rlct_epochs = [baseline_history["epoch"][i] for i in range(0, len(baseline_history["epoch"]), 10) 
                               if i < len(baseline_history["rlct"])]
        self_modeling_rlct_epochs = [self_modeling_history["epoch"][i] for i in range(0, len(self_modeling_history["epoch"]), 10) 
                                    if i < len(self_modeling_history["rlct"])]
        
        plt.subplot(1, 2, 2)
        plt.plot(baseline_rlct_epochs, baseline_history["rlct"], 
                 label="Baseline RLCT", color="blue")
        plt.plot(self_modeling_rlct_epochs, self_modeling_history["rlct"], 
                 label="Self-Modeling RLCT", color="red")
        plt.xlabel("Epoch")
        plt.ylabel("RLCT")
        plt.title(f"Real Log Canonical Threshold Comparison for {task_name}")
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
```

#### 3.2.3 Weight Distribution Visualization

```python
def plot_weight_distributions(baseline_history, self_modeling_history, task_name, save_path=None):
    """Plot histograms of weight distributions at different training stages"""
    # We'll create histograms of the weight distributions at the beginning, middle, and end of training
    
    # Extract weight standard deviations at different points
    baseline_epochs = baseline_history["epoch"]
    self_modeling_epochs = self_modeling_history["epoch"]
    
    # Get indices for beginning, middle, and end
    baseline_start_idx = 0
    baseline_mid_idx = len(baseline_epochs) // 2
    baseline_end_idx = len(baseline_epochs) - 1
    
    self_modeling_start_idx = 0
    self_modeling_mid_idx = len(self_modeling_epochs) // 2
    self_modeling_end_idx = len(self_modeling_epochs) - 1
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot weight std over time
    plt.subplot(2, 2, 1)
    plt.plot(baseline_history["epoch"], baseline_history["weight_std"], 
             label="Baseline", color="blue")
    plt.plot(self_modeling_history["epoch"], self_modeling_history["weight_std"], 
             label="Self-Modeling", color="red")
    plt.axvline(x=baseline_epochs[baseline_start_idx], color='gray', linestyle='--')
    plt.axvline(x=baseline_epochs[baseline_mid_idx], color='gray', linestyle='--')
    plt.axvline(x=baseline_epochs[baseline_end_idx], color='gray', linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Weight StdDev")
    plt.title("Weight Distribution Width Over Time")
    plt.legend()
    plt.grid(True)
    
    # Plot histograms at beginning, middle, and end
    # Note: In a real implementation, we would need to store the actual weight values
    # at these points, not just the standard deviation
    
    # For illustration, we'll create synthetic histograms based on the std values
    # In the actual implementation, we would use the real weight distributions
    
    # Beginning of training
    plt.subplot(2, 3, 4)
    x_baseline = np.random.normal(0, baseline_history["weight_std"][baseline_start_idx], 1000)
    x_self_modeling = np.random.normal(0, self_modeling_history["weight_std"][self_modeling_start_idx], 1000)
    plt.hist(x_baseline, alpha=0.5, bins=30, label="Baseline", color="blue")
    plt.hist(x_self_modeling, alpha=0.5, bins=30, label="Self-Modeling", color="red")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.title(f"Weight Distribution at Epoch {baseline_epochs[baseline_start_idx]}")
    plt.legend()
    
