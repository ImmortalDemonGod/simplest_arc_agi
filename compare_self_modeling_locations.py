#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import random
from typing import Optional, Dict, List, Any, cast, Union
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import os
import time
import numpy as np
from collections import defaultdict

# Set fixed random seed for reproducibility
def set_seed(seed=42):
    """Set seed for all random number generators for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Import project modules
from src.data_generation.binary_ops import generate_modular_addition_data, format_for_transformer
from src.models.transformer import SimpleTransformer, TransformerConfig
from src.training.trainer import AlgorithmicTaskTrainer

def setup_directories(base_dir):
    """Create checkpoint directories for each model variant"""
    os.makedirs(base_dir, exist_ok=True)
    
    # Create subdirectories for each model variant
    variants = ["baseline", "first_layer", "middle_layer", "last_layer", "all_layers"]
    for variant in variants:
        os.makedirs(os.path.join(base_dir, variant), exist_ok=True)
    
    return variants

def load_data(modulus, train_ratio=0.7, val_ratio=0.15, num_samples=None):
    """
    Generate and format data for modular addition with industry-standard split (70/15/15)
    If num_samples is provided, generate multiple copies of the data to simulate larger dataset
    """
    # Special tokens
    START_TOKEN = modulus
    SEP_TOKEN = modulus + 1
    PAD_TOKEN = modulus + 2
    
    # Generate the data with industry-standard split
    data = generate_modular_addition_data(modulus=modulus, train_ratio=train_ratio, val_ratio=val_ratio)
    
    # Calculate the original size
    original_size = len(data["train_inputs"])
    
    # If num_samples specified, expand the dataset by duplication
    if num_samples and num_samples > len(data["train_inputs"]):
        # Calculate how many times to repeat the data
        repeat_factor = (num_samples // len(data["train_inputs"])) + 1
        
        # Repeat training data
        repeated_train_inputs = data["train_inputs"].repeat(repeat_factor, 1)
        repeated_train_targets = data["train_targets"].repeat(repeat_factor)
        
        # Trim to desired size
        data["train_inputs"] = repeated_train_inputs[:num_samples]
        data["train_targets"] = repeated_train_targets[:num_samples]
        
        print(f"Expanded training dataset from {original_size} to {len(data['train_inputs'])} examples")
    
    # Format for transformer
    train_inputs, train_targets = format_for_transformer(
        data["train_inputs"], data["train_targets"], START_TOKEN, SEP_TOKEN, PAD_TOKEN
    )
    val_inputs, val_targets = format_for_transformer(
        data["val_inputs"], data["val_targets"], START_TOKEN, SEP_TOKEN, PAD_TOKEN
    )
    test_inputs, test_targets = format_for_transformer(
        data["test_inputs"], data["test_targets"], START_TOKEN, SEP_TOKEN, PAD_TOKEN
    )
    
    # Create datasets
    train_dataset = TensorDataset(train_inputs, train_targets)
    val_dataset = TensorDataset(val_inputs, val_targets)
    test_dataset = TensorDataset(test_inputs, test_targets)
    
    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "vocab_size": modulus + 3  # Numbers 0 to (modulus-1) + 3 special tokens
    }

def create_model(vocab_size, hidden_size, num_layers, num_heads, use_self_modeling=False, target_layer: Union[str, List[str], None] = "last_hidden"):
    """Create and initialize the transformer model with specified self-modeling configuration"""
    if isinstance(target_layer, list):
        # Multi-layer self-modeling
        config = TransformerConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            pad_token_id=vocab_size - 1,  # PAD_TOKEN
            use_self_modeling=use_self_modeling,
            self_modeling_target_layer="last_hidden",  # Default value, will be overridden
            self_modeling_target_layers=target_layer,  # Set the list of target layers
            self_modeling_loss_weight=1.0  # Use consistent weight for fair comparison
        )
    else:
        # Single-layer self-modeling
        config = TransformerConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            pad_token_id=vocab_size - 1,  # PAD_TOKEN
            use_self_modeling=use_self_modeling,
            self_modeling_target_layer=target_layer if target_layer is not None else "last_hidden",
            self_modeling_loss_weight=1.0  # Use consistent weight for fair comparison
        )
    
    model = SimpleTransformer(config)
    return model, config

def create_optimizer(model, lr, weight_decay):
    """Create optimizer and learning rate scheduler"""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Create the scheduler - we'll use None instead to avoid type issues
    # The trainer will handle this case
    lr_scheduler = None
    return optimizer, lr_scheduler

def train_model_variant(variant_name, target_layer, data_dict, params, base_dir, target_accuracy=None):
    """Train a specific model variant and return its training history"""
    print(f"\n=== Training {variant_name} Model ===")
    
    # Determine if self-modeling should be used
    use_self_modeling = variant_name != "baseline"
    
    # Create model
    model, config = create_model(
        data_dict["vocab_size"],
        params["hidden_size"],
        params["num_layers"],
        params["num_heads"],
        use_self_modeling=use_self_modeling,
        target_layer=target_layer
    )
    
    # Create optimizer
    optimizer, lr_scheduler = create_optimizer(
        model,
        params["learning_rate"],
        params["weight_decay"]
    )
    
    # Create a custom trainer class that stops at target accuracy if specified
    class TargetAccuracyTrainer(AlgorithmicTaskTrainer):
        def train(self, verbose=True, target_accuracy=None):
            """Train until reaching target accuracy or max epochs"""
            print(f"Starting training on {self.device}" +
                  (f", targeting {target_accuracy:.1%} test accuracy" if target_accuracy else ""))
            start_time = time.time()
            
            # Use validation set for early stopping if available
            use_val_for_stopping = self.val_dataloader is not None
            
            while self.current_epoch < self.max_epochs:
                epoch_start_time = time.time()
                
                # Train one epoch
                train_epoch_result = self.train_epoch()
                if self.use_self_modeling and len(train_epoch_result) == 4:
                    train_loss, train_acc, primary_loss, self_modeling_loss = train_epoch_result
                    # Update self-modeling specific metrics
                    self.training_history["primary_loss"].append(primary_loss)
                    self.training_history["self_modeling_loss"].append(self_modeling_loss)
                else:
                    train_loss, train_acc = train_epoch_result[:2]
                
                # Evaluate on test set
                test_loss, test_acc = self.evaluate()
                
                # Evaluate on validation set if available
                val_acc = 0.0
                val_loss = 0.0  # Initialize val_loss
                if use_val_for_stopping and self.val_dataloader is not None:
                    # Use parent class's evaluate method but with validation dataloader
                    self.model.eval()
                    val_loss = 0.0
                    val_correct = 0
                    val_total = 0
                    
                    with torch.no_grad():
                        for input_ids, target_ids in self.val_dataloader:
                            input_ids = input_ids.to(self.device)
                            target_ids = target_ids.to(self.device)
                            
                            # Forward pass
                            if self.use_self_modeling:
                                logits, _, _ = self.model(input_ids)
                            else:
                                logits = self.model(input_ids)
                                
                            loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                            val_loss += loss.item()
                            
                            # Compute accuracy
                            mask = (target_ids != -100).float()
                            predictions = logits.argmax(dim=-1)
                            correct = ((predictions == target_ids) * mask).sum().item()
                            total = mask.sum().item()
                            
                            val_correct += correct
                            val_total += total
                    
                    if len(self.val_dataloader) > 0:
                        val_loss = val_loss / len(self.val_dataloader)
                        val_acc = val_correct / val_total if val_total > 0 else 0.0
                
                # Measure weight distribution
                weight_std = self.measure_weight_distribution()
                self.training_history["weight_std"].append(weight_std)
                
                # Estimate RLCT (less frequently)
                if self.current_epoch % 10 == 0:
                    rlct = self.estimate_rlct()
                    self.training_history["rlct"].append(rlct)
                
                # Track validation accuracy if available
                if use_val_for_stopping:
                    if "val_acc" not in self.training_history:
                        self.training_history["val_acc"] = []
                    if "val_loss" not in self.training_history:
                        self.training_history["val_loss"] = []
                    self.training_history["val_acc"].append(val_acc)
                    self.training_history["val_loss"].append(val_loss)
                
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
                
                # Check if we've reached target accuracy - use validation accuracy if available
                eval_acc_for_target = val_acc if use_val_for_stopping else test_acc
                if target_accuracy is not None and eval_acc_for_target >= target_accuracy:
                    if use_val_for_stopping:
                        print(f"Target accuracy {target_accuracy:.1%} reached at epoch {self.current_epoch} (validation)")
                    else:
                        print(f"Target accuracy {target_accuracy:.1%} reached at epoch {self.current_epoch}")
                    
                    # Save checkpoint if directory provided
                    if self.checkpoint_dir is not None:
                        self.save_checkpoint(os.path.join(self.checkpoint_dir, "target_accuracy_model.pt"))
                    
                    break
                
                # Check for improvement - use validation accuracy if available, otherwise test accuracy
                eval_acc = val_acc if use_val_for_stopping else test_acc
                
                if eval_acc > self.best_test_acc:
                    self.best_test_acc = eval_acc
                    self.epochs_without_improvement = 0
                    
                    # Save checkpoint if directory provided
                    if self.checkpoint_dir is not None:
                        self.save_checkpoint(os.path.join(self.checkpoint_dir, "best_model.pt"))
                        
                    if verbose:
                        if use_val_for_stopping:
                            print(f"New best model with validation accuracy: {val_acc:.4f}")
                        else:
                            print(f"New best model with test accuracy: {test_acc:.4f}")
                else:
                    self.epochs_without_improvement += 1
                    
                    # Early stopping
                    if self.epochs_without_improvement >= self.patience:
                        print(f"Early stopping after {self.current_epoch} epochs without improvement")
                        break
                
                # Save checkpoint every 100 epochs
                if self.checkpoint_dir is not None and self.current_epoch % 100 == 0:
                    self.save_checkpoint(os.path.join(self.checkpoint_dir, f"model_epoch_{self.current_epoch}.pt"))
                
                self.current_epoch += 1
            
            total_time = time.time() - start_time
            print(f"Training completed in {total_time:.2f}s")
            
            # Save final model
            if self.checkpoint_dir is not None:
                self.save_checkpoint(os.path.join(self.checkpoint_dir, "final_model.pt"))
            
            return self.training_history
    
    # Create trainer
    trainer = TargetAccuracyTrainer(
        model=model,
        train_dataset=data_dict["train_dataset"],
        test_dataset=data_dict["test_dataset"],
        val_dataset=data_dict["val_dataset"],  # Pass validation dataset
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        batch_size=params["batch_size"],
        max_epochs=params["max_epochs"],
        device=params["device"],
        checkpoint_dir=os.path.join(base_dir, variant_name),
        primary_loss_weight=1.0,
        self_modeling_loss_weight=1.0 if use_self_modeling else None
    )
    
    # Train model
    start_time = time.time()
    history = trainer.train(target_accuracy=target_accuracy)
    training_time = time.time() - start_time
    
    # Print results
    print(f"{variant_name} training completed in {training_time:.2f}s")
    print(f"Final test accuracy: {history['test_acc'][-1]:.4f}")
    
    return history

def plot_comparison(histories, variants, base_dir):
    """Create comparison plots for all model variants"""
    # Define metrics to plot
    metrics = [
        {"name": "test_acc", "title": "Test Accuracy", "ylabel": "Accuracy"},
        {"name": "train_acc", "title": "Train Accuracy", "ylabel": "Accuracy"},
        {"name": "test_loss", "title": "Test Loss", "ylabel": "Loss"},
        {"name": "train_loss", "title": "Train Loss", "ylabel": "Loss"},
        {"name": "weight_std", "title": "Weight Standard Deviation", "ylabel": "Std Dev"},
    ]
    
    # Check if validation metrics are available in any history
    if any("val_acc" in h for h in histories.values()):
        metrics.append({"name": "val_acc", "title": "Validation Accuracy", "ylabel": "Accuracy"})
    if any("val_loss" in h for h in histories.values()):
        metrics.append({"name": "val_loss", "title": "Validation Loss", "ylabel": "Loss"})
    
    # Check if RLCT is available in all histories
    if all("rlct" in h for h in histories.values()):
        metrics.append({"name": "rlct", "title": "Real Log Canonical Threshold (RLCT)", "ylabel": "RLCT"})
    
    # Check if self-modeling loss is available in relevant histories
    if all("self_modeling_loss" in h for name, h in histories.items() if name != "baseline"):
        metrics.append({"name": "self_modeling_loss", "title": "Self-Modeling Loss", "ylabel": "Loss"})
    
    # Create a figure for each metric
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        for variant in variants:
            # Skip self_modeling_loss for baseline
            if metric["name"] == "self_modeling_loss" and variant == "baseline":
                continue
                
            history = histories[variant]
            
            # For RLCT, we need to handle sparse data points
            if metric["name"] == "rlct":
                # RLCT is measured every 10 epochs, so we need to create corresponding x-axis values
                rlct_epochs = [epoch for i, epoch in enumerate(history["epoch"]) if i % 10 == 0]
                if len(history[metric["name"]]) > 0:  # Only plot if there's data
                    plt.plot(rlct_epochs, history[metric["name"]], label=variant.replace("_", " ").title())
            else:
                if len(history[metric["name"]]) > 0:  # Only plot if there's data
                    plt.plot(history["epoch"], history[metric["name"]], label=variant.replace("_", " ").title())
        
        plt.xlabel("Epoch")
        plt.ylabel(metric["ylabel"])
        plt.title(f"{metric['title']} Comparison")
        plt.legend()
        plt.grid(True)
        
        # Save figure
        plt.savefig(os.path.join(base_dir, f"comparison_{metric['name']}.png"))
    
    # Create a combined plot for test/validation accuracy and loss
    plt.figure(figsize=(12, 10))
    
    # Test/validation accuracy subplot
    plt.subplot(2, 1, 1)
    for variant in variants:
        if variant not in histories:
            continue
        history = histories[variant]
        if len(history["test_acc"]) > 0:
            plt.plot(history["epoch"], history["test_acc"], label=f"{variant.replace('_', ' ').title()} (Test)")
        # Add validation accuracy if available
        if "val_acc" in history and len(history["val_acc"]) > 0:
            plt.plot(history["epoch"], history["val_acc"], label=f"{variant.replace('_', ' ').title()} (Val)", linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test/Validation Accuracy Comparison")
    plt.legend()
    plt.grid(True)
    
    # Test/validation loss subplot
    plt.subplot(2, 1, 2)
    for variant in variants:
        if variant not in histories:
            continue
        history = histories[variant]
        if len(history["test_loss"]) > 0:
            plt.plot(history["epoch"], history["test_loss"], label=f"{variant.replace('_', ' ').title()} (Test)")
        # Add validation loss if available
        if "val_loss" in history and len(history["val_loss"]) > 0:
            plt.plot(history["epoch"], history["val_loss"], label=f"{variant.replace('_', ' ').title()} (Val)", linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Test/Validation Loss Comparison")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "comparison_combined.png"))
    
    # Create a learning dynamics plot (accuracy vs. loss)
    plt.figure(figsize=(10, 8))
    for variant in variants:
        if variant not in histories:
            continue
        history = histories[variant]
        if len(history["test_loss"]) > 0 and len(history["test_acc"]) > 0:
            plt.plot(history["test_loss"], history["test_acc"], 'o-', label=f"{variant.replace('_', ' ').title()} (Test)", alpha=0.7)
        # Add validation dynamics if available
        if "val_acc" in history and "val_loss" in history and len(history["val_acc"]) > 0 and len(history["val_loss"]) > 0:
            plt.plot(history["val_loss"], history["val_acc"], 'o--', label=f"{variant.replace('_', ' ').title()} (Val)", alpha=0.7)
    plt.xlabel("Loss")
    plt.ylabel("Accuracy")
    plt.title("Learning Dynamics: Accuracy vs. Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(base_dir, "learning_dynamics.png"))
    
    # If self-modeling loss is available, create a plot comparing it across variants
    if all("self_modeling_loss" in h for name, h in histories.items() if name != "baseline"):
        plt.figure(figsize=(10, 6))
        for variant in variants:
            if variant == "baseline":
                continue
            history = histories[variant]
            if len(history["self_modeling_loss"]) > 0:
                plt.plot(history["epoch"], history["self_modeling_loss"], label=variant.replace("_", " ").title())
        plt.xlabel("Epoch")
        plt.ylabel("Self-Modeling Loss")
        plt.title("Self-Modeling Loss Comparison")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(base_dir, "comparison_self_modeling_loss.png"))
    
    print(f"Comparison plots saved to {base_dir}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Compare self-modeling at different locations in transformer models")
    parser.add_argument("--max_epochs", type=int, default=150,
                        help="Maximum number of epochs to train each model")
    parser.add_argument("--samples", type=int, default=5000,
                        help="Number of training samples to generate")
    parser.add_argument("--base_dir", type=str, default="checkpoints/self_modeling_comparison",
                        help="Base directory for saving checkpoints and visualizations")
    parser.add_argument("--modulus", type=int, default=19,
                        help="Modulus for the addition task")
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="Hidden size for the transformer model")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of transformer layers")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--target_accuracy", type=float, default=None,
                        help="Target test accuracy to stop training (e.g., 0.8 for 80%%)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Configuration
    params = {
        "modulus": args.modulus,
        "train_ratio": 0.8,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "num_heads": 4,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    # Base directory for checkpoints and plots
    base_dir = args.base_dir
    
    # Setup directories and get variant names
    variants = setup_directories(base_dir)
    
    print(f"Using device: {params['device']}")
    
    # Generate data
    print("Generating dataset...")
    data_dict = load_data(params["modulus"], params["train_ratio"], num_samples=args.samples)
    print(f"Generated {len(data_dict['train_dataset'])} training examples and {len(data_dict['test_dataset'])} test examples")
    
    # Define model variants with their target layers
    variant_configs = {
        "baseline": None,  # No self-modeling
        "first_layer": "layer_0",  # First layer
        "middle_layer": "middle",  # Middle layer
        "last_layer": "last_hidden",  # Last layer
        "all_layers": ["layer_0", "middle", "last_hidden"]  # All three layers
    }
    
    # Train all model variants
    histories = {}
    for variant, target_layer in variant_configs.items():
        history = train_model_variant(
            variant,
            target_layer,
            data_dict,
            params,
            base_dir,
            target_accuracy=args.target_accuracy
        )
        histories[variant] = history
    
    # Create comparison plots
    plot_comparison(histories, variants, base_dir)
    
    # Print final comparison
    print("\n=== Final Comparison ===")
    for variant in variants:
        print(f"{variant.replace('_', ' ').title()} final test accuracy: {histories[variant]['test_acc'][-1]:.4f}")
    
    print("\nResults and plots saved to", base_dir)

if __name__ == "__main__":
    main()