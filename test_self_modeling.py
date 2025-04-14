#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
import os
import time
import numpy as np

# Import project modules
from src.data_generation.binary_ops import generate_modular_addition_data, format_for_transformer
from src.models.transformer import SimpleTransformer, TransformerConfig
from src.training.trainer import AlgorithmicTaskTrainer

def main():
    # Configuration - updated to match run_training.py
    modulus = 19  # Using modular addition (mod 19) as in run_training.py
    train_ratio = 0.8
    hidden_size = 128  # Increased from 64 to 128
    num_layers = 2
    num_heads = 4  # Increased from 2 to 4
    max_epochs = 150  # Increased from 100 to 150 for better observation of effects
    batch_size = 32
    learning_rate = 1e-3
    weight_decay = 0.01
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create checkpoint directories
    baseline_dir = "checkpoints/baseline"
    self_modeling_dir = "checkpoints/self_modeling"
    os.makedirs(baseline_dir, exist_ok=True)
    os.makedirs(self_modeling_dir, exist_ok=True)
    
    print(f"Using device: {device}")
    
    # Generate data with expanded dataset
    print("Generating expanded dataset...")
    data_dict = load_data(modulus, train_ratio, num_samples=5000)  # Using 5000 samples instead of 50000 for faster testing
    print(f"Generated {len(data_dict['train_dataset'])} training examples and {len(data_dict['test_dataset'])} test examples")
    
    # Train baseline model (without self-modeling)
    print("\n=== Training Baseline Model (without self-modeling) ===")
    baseline_model, baseline_config = create_model(
        data_dict["vocab_size"], 
        hidden_size, 
        num_layers, 
        num_heads, 
        use_self_modeling=False
    )
    baseline_optimizer, baseline_lr_scheduler = create_optimizer(
        baseline_model, 
        learning_rate, 
        weight_decay
    )
    baseline_trainer = AlgorithmicTaskTrainer(
        model=baseline_model,
        train_dataset=data_dict["train_dataset"],
        test_dataset=data_dict["test_dataset"],
        optimizer=baseline_optimizer,
        lr_scheduler=baseline_lr_scheduler,
        batch_size=batch_size,
        max_epochs=max_epochs,
        device=device,
        checkpoint_dir=baseline_dir
    )
    baseline_history = baseline_trainer.train()
    
    # Train self-modeling model
    print("\n=== Training Self-Modeling Model ===")
    self_modeling_model, self_modeling_config = create_model(
        data_dict["vocab_size"], 
        hidden_size, 
        num_layers, 
        num_heads, 
        use_self_modeling=True
    )
    self_modeling_optimizer, self_modeling_lr_scheduler = create_optimizer(
        self_modeling_model, 
        learning_rate, 
        weight_decay
    )
    self_modeling_trainer = AlgorithmicTaskTrainer(
        model=self_modeling_model,
        train_dataset=data_dict["train_dataset"],
        test_dataset=data_dict["test_dataset"],
        optimizer=self_modeling_optimizer,
        lr_scheduler=self_modeling_lr_scheduler,
        batch_size=batch_size,
        max_epochs=max_epochs,
        device=device,
        checkpoint_dir=self_modeling_dir,
        primary_loss_weight=1.0,
        self_modeling_loss_weight=1.0  # Reduced from 5.0 to match our new default
    )
    self_modeling_history = self_modeling_trainer.train()
    
    # Compare results
    print("\n=== Comparison ===")
    print(f"Baseline final test accuracy: {baseline_history['test_acc'][-1]:.4f}")
    print(f"Self-modeling final test accuracy: {self_modeling_history['test_acc'][-1]:.4f}")
    
    print(f"Baseline final weight std: {baseline_history['weight_std'][-1]:.4f}")
    print(f"Self-modeling final weight std: {self_modeling_history['weight_std'][-1]:.4f}")
    
    if "rlct" in baseline_history and "rlct" in self_modeling_history:
        print(f"Baseline final RLCT: {baseline_history['rlct'][-1]:.4f}")
        print(f"Self-modeling final RLCT: {self_modeling_history['rlct'][-1]:.4f}")
    
    print("\nResults saved to checkpoints/baseline and checkpoints/self_modeling")

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
        "vocab_size": modulus + 3  # Numbers 0 to (modulus-1) + 3 special tokens
    }

def create_model(vocab_size, hidden_size, num_layers, num_heads, use_self_modeling):
    """Create and initialize the transformer model"""
    config = TransformerConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        pad_token_id=vocab_size - 1,  # PAD_TOKEN
        use_self_modeling=use_self_modeling,
        self_modeling_target_layer="middle",  # Use middle layer instead of last_hidden
        self_modeling_loss_weight=1.0  # Use lower weight for self-modeling loss
    )
    
    model = SimpleTransformer(config)
    return model, config

def create_optimizer(model, lr, weight_decay):
    """Create optimizer and learning rate scheduler"""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-5)
    return optimizer, lr_scheduler

if __name__ == "__main__":
    main()