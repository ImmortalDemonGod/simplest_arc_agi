import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import os
import sys
import copy
from typing import Dict, List, Tuple, Optional, Callable, Union

# Add src to path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.transformer import SimpleTransformer, TransformerConfig

class AlgorithmicTaskTrainer:
    """Trainer class for algorithmic tasks using transformer models"""
    
    def __init__(
        self,
        model: SimpleTransformer,
        train_dataset: TensorDataset,
        test_dataset: TensorDataset,
        val_dataset: Optional[TensorDataset] = None,
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
        self_modeling_loss_weight: Optional[float] = None
    ):
        self.model = model
        self.device = device
        self.model.to(self.device)
        
        # Datasets and dataloaders
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset  # May be None
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
        self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset is not None else None
        
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
            "rlct": [],
            # For multi-layer self-modeling
            "layer_losses": {}
        }
    
    def train_epoch(self) -> Union[Tuple[float, float], Tuple[float, float, float, float], Tuple[float, float, float, float, Dict[str, float]]]:
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
                outputs = self.model(input_ids)
                
                # Handle both single-layer and multi-layer self-modeling
                if isinstance(outputs[1], dict):
                    # Multi-layer self-modeling
                    logits, true_activations_dict, predicted_activations_dict = outputs
                    
                    # Compute primary loss
                    primary_loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                    
                    # Initialize total self-modeling loss and layer-specific losses
                    self_modeling_loss = 0.0
                    layer_losses_dict = {}
                    
                    # Compute self-modeling loss for each layer
                    for layer_name, true_activations in true_activations_dict.items():
                        predicted_activations = predicted_activations_dict[layer_name]
                        
                        # Normalize activations
                        true_mean = true_activations.mean(dim=-1, keepdim=True)
                        true_std = true_activations.std(dim=-1, keepdim=True) + 1e-6
                        pred_mean = predicted_activations.mean(dim=-1, keepdim=True)
                        pred_std = predicted_activations.std(dim=-1, keepdim=True) + 1e-6
                        
                        normalized_true = (true_activations - true_mean) / true_std
                        normalized_pred = (predicted_activations - pred_mean) / pred_std
                        
                        # Compute layer-specific loss
                        layer_loss = self.mse_loss(normalized_pred, normalized_true)
                        layer_losses_dict[layer_name] = layer_loss.item()
                        
                        # Add to total self-modeling loss
                        self_modeling_loss += layer_loss
                    
                    # Average the self-modeling loss across layers
                    if len(layer_losses_dict) > 0:
                        self_modeling_loss /= len(layer_losses_dict)
                    
                    # Combine losses
                    loss = self.primary_loss_weight * primary_loss + self.self_modeling_loss_weight * self_modeling_loss
                    
                    # Track individual losses
                    total_primary_loss += primary_loss.item()
                    total_self_modeling_loss += float(self_modeling_loss)
                else:
                    # Single-layer self-modeling (backward compatibility)
                    logits, true_activations, predicted_activations = outputs
                    
                    # Compute primary loss
                    primary_loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                    
                    # Compute self-modeling loss with normalization
                    # Normalize activations to have zero mean and unit variance
                    true_mean = true_activations.mean(dim=-1, keepdim=True)
                    true_std = true_activations.std(dim=-1, keepdim=True) + 1e-6
                    pred_mean = predicted_activations.mean(dim=-1, keepdim=True)
                    pred_std = predicted_activations.std(dim=-1, keepdim=True) + 1e-6
                    
                    normalized_true = (true_activations - true_mean) / true_std
                    normalized_pred = (predicted_activations - pred_mean) / pred_std
                    
                    self_modeling_loss = self.mse_loss(normalized_pred, normalized_true)
                    
                    # Combine losses
                    loss = self.primary_loss_weight * primary_loss + self.self_modeling_loss_weight * self_modeling_loss
                    
                    # Track individual losses
                    total_primary_loss += primary_loss.item()
                    total_self_modeling_loss += float(self_modeling_loss)
                    
                    # No layer_losses needed for single-layer self-modeling
                    pass
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
                    # For multi-layer self-modeling, we just show the total loss
                    print(f"Epoch {self.current_epoch} [{batch_idx}/{len(self.train_dataloader)}] "
                          f"Loss: {loss.item():.6f}, Primary: {primary_loss.item():.6f}, "
                          f"Self-Modeling: {float(self_modeling_loss):.6f}, Acc: {correct/total:.4f}")
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
            
            # For multi-layer self-modeling, also return layer-specific losses
            if 'layer_losses_dict' in locals():
                # Average layer losses across batches
                avg_layer_losses = {}
                for layer_name in layer_losses_dict:
                    avg_layer_losses[layer_name] = layer_losses_dict[layer_name] / len(self.train_dataloader)
                return avg_loss, avg_accuracy, avg_primary_loss, avg_self_modeling_loss, avg_layer_losses
            else:
                return avg_loss, avg_accuracy, avg_primary_loss, avg_self_modeling_loss
        
        return avg_loss, avg_accuracy
    
    def evaluate(self) -> Tuple[float, float]:
        """Evaluate model on test set"""
        self.model.eval()
        total_loss = 0.0
        total_primary_loss = 0.0
        total_self_modeling_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for input_ids, target_ids in self.test_dataloader:
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                # Forward pass
                if self.use_self_modeling:
                    # Get both primary and auxiliary outputs
                    outputs = self.model(input_ids)
                    
                    # Handle both single-layer and multi-layer self-modeling
                    if isinstance(outputs[1], dict):
                        # Multi-layer self-modeling
                        logits, true_activations_dict, predicted_activations_dict = outputs
                        
                        # Compute primary loss
                        primary_loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                        
                        # Use the first layer's activations for simplicity in evaluation
                        layer_name = next(iter(true_activations_dict.keys()))
                        true_activations = true_activations_dict[layer_name]
                        predicted_activations = predicted_activations_dict[layer_name]
                    else:
                        # Single-layer self-modeling
                        logits, true_activations, predicted_activations = outputs
                        
                        # Compute primary loss
                        primary_loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                    
                    # Compute self-modeling loss with normalization
                    # Normalize activations to have zero mean and unit variance
                    true_mean = true_activations.mean(dim=-1, keepdim=True)
                    true_std = true_activations.std(dim=-1, keepdim=True) + 1e-6
                    pred_mean = predicted_activations.mean(dim=-1, keepdim=True)
                    pred_std = predicted_activations.std(dim=-1, keepdim=True) + 1e-6
                    
                    normalized_true = (true_activations - true_mean) / true_std
                    normalized_pred = (predicted_activations - pred_mean) / pred_std
                    
                    self_modeling_loss = self.mse_loss(normalized_pred, normalized_true)
                    
                    # Combine losses
                    loss = self.primary_loss_weight * primary_loss + self.self_modeling_loss_weight * self_modeling_loss
                    
                    # Track individual losses
                    total_primary_loss += primary_loss.item()
                    total_self_modeling_loss += float(self_modeling_loss)
                else:
                    # Standard forward pass
                    logits = self.model(input_ids)
                    loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                
                total_loss += loss.item()
                
                # Compute accuracy (only for non-ignored positions)
                mask = (target_ids != -100).float()
                predictions = logits.argmax(dim=-1)
                correct = ((predictions == target_ids) * mask).sum().item()
                total = mask.sum().item()
                
                correct_predictions += correct
                total_predictions += total
        
        # Compute average loss and accuracy
        avg_loss = total_loss / len(self.test_dataloader)
        avg_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        # Also store self-modeling metrics if enabled
        if self.use_self_modeling:
            avg_primary_loss = total_primary_loss / len(self.test_dataloader)
            avg_self_modeling_loss = total_self_modeling_loss / len(self.test_dataloader)
            self._last_primary_loss = avg_primary_loss
            self._last_self_modeling_loss = avg_self_modeling_loss
        
        return avg_loss, avg_accuracy
    
    def train(self, verbose: bool = True) -> Dict:
        """
        Train the model for self.max_epochs epochs or until early stopping condition is met
        Returns the training history
        """
        print(f"Starting training on {self.device}")
        start_time = time.time()
        
        while self.current_epoch < self.max_epochs:
            epoch_start_time = time.time()
            
            # Train one epoch
            if self.use_self_modeling:
                result = self.train_epoch()
                if len(result) == 5:
                    # Multi-layer self-modeling
                    train_loss, train_acc, primary_loss, self_modeling_loss, layer_losses = result
                    # Store layer-specific losses
                    for layer_name, layer_loss in layer_losses.items():
                        if layer_name not in self.training_history["layer_losses"]:
                            self.training_history["layer_losses"][layer_name] = []
                        self.training_history["layer_losses"][layer_name].append(layer_loss)
                else:
                    # Single-layer self-modeling
                    train_loss, train_acc, primary_loss, self_modeling_loss = result
                
                # Update self-modeling specific metrics
                self.training_history["primary_loss"].append(primary_loss)
                self.training_history["self_modeling_loss"].append(self_modeling_loss)
            else:
                result = self.train_epoch()
                train_loss, train_acc = result
            
            # Evaluate
            test_loss, test_acc = self.evaluate()
            
            # Measure weight distribution
            weight_std = self.measure_weight_distribution()
            self.training_history["weight_std"].append(weight_std)
            
            # Estimate RLCT (less frequently)
            if self.current_epoch % 10 == 0:
                rlct = self.estimate_rlct()
                self.training_history["rlct"].append(rlct)
            
            # Step learning rate scheduler if provided
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            # Update training history
            self.training_history["train_loss"].append(train_loss)
            self.training_history["train_acc"].append(train_acc)
            self.training_history["test_loss"].append(test_loss)
            self.training_history["test_acc"].append(test_acc)
            self.training_history["epoch"].append(self.current_epoch)
            self.training_history["learning_rate"].append(current_lr)
            
            epoch_time = time.time() - epoch_start_time
            
            if verbose:
                print(f"Epoch {self.current_epoch}: "
                      f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f}, "
                      f"Test Loss: {test_loss:.6f}, Test Acc: {test_acc:.4f}, "
                      f"LR: {current_lr:.6f}, Time: {epoch_time:.2f}s")
            
            # Check for improvement
            if test_acc > self.best_test_acc:
                self.best_test_acc = test_acc
                self.epochs_without_improvement = 0
                
                # Save checkpoint if directory provided
                if self.checkpoint_dir is not None:
                    self.save_checkpoint(os.path.join(self.checkpoint_dir, "best_model.pt"))
                    
                if verbose:
                    print(f"New best model with test accuracy: {test_acc:.4f}")
            else:
                self.epochs_without_improvement += 1
                
                # Early stopping
                if self.epochs_without_improvement >= self.patience:
                    print(f"Early stopping after {self.current_epoch} epochs without improvement")
                    break
            
            # Save checkpoint every 1000 epochs
            if self.checkpoint_dir is not None and self.current_epoch % 1000 == 0:
                self.save_checkpoint(os.path.join(self.checkpoint_dir, f"model_epoch_{self.current_epoch}.pt"))
            
            self.current_epoch += 1
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f}s")
        
        # Save final model
        if self.checkpoint_dir is not None:
            self.save_checkpoint(os.path.join(self.checkpoint_dir, "final_model.pt"))
        
        return self.training_history
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_test_acc": self.best_test_acc,
            "training_history": self.training_history
        }
        
        if self.lr_scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.lr_scheduler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str, load_optimizer: bool = True) -> None:
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.lr_scheduler is not None and "scheduler_state_dict" in checkpoint:
                self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.current_epoch = checkpoint["current_epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_test_acc = checkpoint["best_test_acc"]
        self.training_history = checkpoint["training_history"]
    
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
                    outputs = model_copy(val_input_ids)
                    
                    # Handle both single-layer and multi-layer self-modeling
                    if isinstance(outputs[1], dict):
                        # Multi-layer self-modeling
                        logits, true_activations_dict, predicted_activations_dict = outputs
                        
                        # Compute primary loss
                        primary_loss = self.criterion(logits.view(-1, logits.size(-1)), val_target_ids.view(-1))
                        
                        # Use the first layer's activations for simplicity in evaluation
                        layer_name = next(iter(true_activations_dict.keys()))
                        true_activations = true_activations_dict[layer_name]
                        predicted_activations = predicted_activations_dict[layer_name]
                    else:
                        # Single-layer self-modeling
                        logits, true_activations, predicted_activations = outputs
                        
                        # Compute primary loss
                        primary_loss = self.criterion(logits.view(-1, logits.size(-1)), val_target_ids.view(-1))
                    
                    # Normalize activations
                    true_mean = true_activations.mean(dim=-1, keepdim=True)
                    true_std = true_activations.std(dim=-1, keepdim=True) + 1e-6
                    pred_mean = predicted_activations.mean(dim=-1, keepdim=True)
                    pred_std = predicted_activations.std(dim=-1, keepdim=True) + 1e-6
                    
                    normalized_true = (true_activations - true_mean) / true_std
                    normalized_pred = (predicted_activations - pred_mean) / pred_std
                    
                    self_modeling_loss = self.mse_loss(normalized_pred, normalized_true)
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

if __name__ == "__main__":
    # Example usage to test the trainer
    
    # Create a simple random dataset
    batch_size = 32
    vocab_size = 20
    
    # Create dummy data
    train_inputs = torch.randint(0, vocab_size-1, (1000, 5))
    train_targets = torch.full((1000, 5), -100)  # Start with all ignored
    train_targets[:, 4] = torch.randint(0, vocab_size-1, (1000,))  # Only last token is target
    
    test_inputs = torch.randint(0, vocab_size-1, (200, 5))
    test_targets = torch.full((200, 5), -100)
    test_targets[:, 4] = torch.randint(0, vocab_size-1, (200,))
    
    train_dataset = TensorDataset(train_inputs, train_targets)
    test_dataset = TensorDataset(test_inputs, test_targets)
    
    # Create model
    config = TransformerConfig(vocab_size=vocab_size, hidden_size=64, num_hidden_layers=2)
    model = SimpleTransformer(config)
    
    # Create trainer
    trainer = AlgorithmicTaskTrainer(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        max_epochs=5,  # Just a few epochs for testing
        log_interval=5
    )
    
    # Train
    history = trainer.train()
    print("Final test accuracy:", history["test_acc"][-1]) 