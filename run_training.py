#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import os
import time
import sys

# Import project modules
from src.data_generation.binary_ops import generate_modular_addition_data, format_for_transformer
from src.models.transformer import SimpleTransformer, TransformerConfig
from src.training.trainer import AlgorithmicTaskTrainer

def setup_directories(checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)

def load_data(modulus, train_ratio, num_samples=None):
    """
    Generate and format data for modular addition
    If num_samples is provided, generate multiple copies of the data to simulate larger dataset
    """
    # Special tokens
    START_TOKEN = modulus
    SEP_TOKEN = modulus + 1
    PAD_TOKEN = modulus + 2
    
    # Generate the data
    data = generate_modular_addition_data(modulus=modulus, train_ratio=train_ratio)
    
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
    test_inputs, test_targets = format_for_transformer(
        data["test_inputs"], data["test_targets"], START_TOKEN, SEP_TOKEN, PAD_TOKEN
    )
    
    # Create datasets
    train_dataset = TensorDataset(train_inputs, train_targets)
    test_dataset = TensorDataset(test_inputs, test_targets)
    
    return {
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "special_tokens": {
            "START_TOKEN": START_TOKEN,
            "SEP_TOKEN": SEP_TOKEN,
            "PAD_TOKEN": PAD_TOKEN
        },
        "vocab_size": modulus + 3  # Numbers 0 to (modulus-1) + 3 special tokens
    }

def create_model(vocab_size, hidden_size=128, num_layers=2, num_heads=4):
    """Create and initialize the transformer model"""
    config = TransformerConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        pad_token_id=vocab_size - 1  # PAD_TOKEN
    )
    
    model = SimpleTransformer(config)
    return model, config

def create_optimizer(model, lr=1e-3, weight_decay=0.01):
    """Create optimizer and learning rate scheduler"""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-5)
    return optimizer, lr_scheduler

def plot_training_history(history, task_name, save_path=None):
    """Plot and optionally save training history"""
    plt.figure(figsize=(12, 8))
    
    # Plot accuracy
    plt.subplot(2, 1, 1)
    plt.plot(history["epoch"], history["train_acc"], label="Train Accuracy")
    plt.plot(history["epoch"], history["test_acc"], label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy over time for {task_name}")
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(2, 1, 2)
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["test_loss"], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss over time for {task_name}")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

class CustomTrainer(AlgorithmicTaskTrainer):
    """Custom trainer that runs until reaching a target accuracy"""
    
    def train(self, target_accuracy=0.8, max_epochs=10000, verbose=True):
        """
        Train the model until reaching target accuracy or max_epochs
        
        Args:
            target_accuracy: Target test accuracy to reach
            max_epochs: Maximum number of epochs to train
            verbose: Whether to print progress
            
        Returns:
            Training history
        """
        print(f"Starting training on {self.device}, targeting {target_accuracy:.1%} test accuracy")
        start_time = time.time()
        
        while self.current_epoch < max_epochs:
            epoch_start_time = time.time()
            
            # Train one epoch
            train_loss, train_acc = self.train_epoch()
            
            # Evaluate
            test_loss, test_acc = self.evaluate()
            
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
            
            # Check if we've reached target accuracy
            if test_acc >= target_accuracy:
                print(f"Target accuracy {target_accuracy:.1%} reached at epoch {self.current_epoch}")
                
                # Save checkpoint if directory provided
                if self.checkpoint_dir is not None:
                    self.save_checkpoint(os.path.join(self.checkpoint_dir, "target_accuracy_model.pt"))
                
                break
            
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

def main():
    # Configuration
    modulus = 19  # Using modular addition (mod 19) as our single focused task
    train_ratio = 0.8  # More training data
    hidden_size = 128
    num_layers = 2
    num_heads = 4
    max_epochs = 20000  # Increased max epochs to allow more time to reach 100%
    target_accuracy = 1.0  # Run until 100% test accuracy
    batch_size = 32
    learning_rate = 1e-3
    weight_decay = 0.01
    checkpoint_dir = "checkpoints/mod19_target100"
    task_name = f"add_mod_{modulus}"
    
    # Set device to MPS (Metal Performance Shaders) for Apple Silicon
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Setup directories
    setup_directories(checkpoint_dir)
    
    print(f"Starting pipeline for task {task_name} on {device}")
    
    # Generate much more data (50,000 training examples)
    print("1. Generating expanded dataset...")
    data_dict = load_data(modulus, train_ratio, num_samples=50000)
    print(f"Generated {len(data_dict['train_dataset'])} training examples and {len(data_dict['test_dataset'])} test examples")
    
    print(f"2. Creating model with {num_layers} layers, {hidden_size} hidden size...")
    model, config = create_model(data_dict["vocab_size"], hidden_size, num_layers, num_heads)
    optimizer, lr_scheduler = create_optimizer(model, learning_rate, weight_decay)
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {total_params:,} parameters")
    
    print(f"3. Training model until {target_accuracy*100}% test accuracy (or max {max_epochs} epochs)...")
    trainer = CustomTrainer(
        model=model,
        train_dataset=data_dict["train_dataset"],
        test_dataset=data_dict["test_dataset"],
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        batch_size=batch_size,
        max_epochs=max_epochs,  # Just as a safeguard
        device=device,
        checkpoint_dir=checkpoint_dir,
        patience=5000  # Increased patience to allow more time without improvement
    )
    
    # Start training with target accuracy
    start_time = time.time()
    history = trainer.train(target_accuracy=target_accuracy, max_epochs=max_epochs)
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f}s")
    print(f"Final test accuracy: {history['test_acc'][-1]:.4f}")
    
    # Plot and save training history
    plot_path = os.path.join(checkpoint_dir, f"{task_name}_training_history.png")
    plot_training_history(history, task_name, save_path=plot_path)
    
    # Print final metrics
    best_epoch = history["epoch"][history["test_acc"].index(max(history["test_acc"]))]
    best_accuracy = max(history["test_acc"])
    final_epoch = history["epoch"][-1]
    print(f"Best test accuracy: {best_accuracy:.4f} at epoch {best_epoch}")
    print(f"Completed {final_epoch+1} epochs at {(final_epoch+1) / training_time:.2f} epochs/second")
    print(f"Results saved to {checkpoint_dir}")

if __name__ == "__main__":
    main() 