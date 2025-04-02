import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np

from src.data_generation.binary_ops import generate_modular_addition_data, format_for_transformer
from src.models.transformer import SimpleTransformer, TransformerConfig
from src.training.trainer import AlgorithmicTaskTrainer
from src.database.circuit_database import CircuitDatabase
from src.utils import set_seed, get_device

def parse_args():
    parser = argparse.ArgumentParser(description="Train and extract circuits for algorithmic tasks")
    parser.add_argument("--task", type=str, default="add_mod_11", help="Task name")
    parser.add_argument("--modulus", type=int, default=11, help="Modulus for modular addition")
    parser.add_argument("--train_ratio", type=float, default=0.5, help="Fraction of examples for training")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size of transformer")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--max_epochs", type=int, default=10000, help="Maximum number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--db_path", type=str, default="circuits.db", help="Path to circuit database")
    parser.add_argument("--device", type=str, default=get_device(), help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def setup_directories(checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)

def load_data(modulus, train_ratio):
    """Generate and format data for modular addition"""
    # Special tokens
    START_TOKEN = modulus
    SEP_TOKEN = modulus + 1
    PAD_TOKEN = modulus + 2
    
    # Generate the data
    data = generate_modular_addition_data(modulus=modulus, train_ratio=train_ratio)
    
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

def create_model(vocab_size, config_args):
    """Create and initialize the transformer model"""
    config = TransformerConfig(
        vocab_size=vocab_size,
        hidden_size=config_args.hidden_size,
        num_hidden_layers=config_args.num_layers,
        num_attention_heads=config_args.num_heads,
        pad_token_id=vocab_size - 1  # PAD_TOKEN
    )
    
    model = SimpleTransformer(config)
    return model, config

def create_optimizer(model, lr, weight_decay):
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

def extract_simple_circuit(model, task_name, modulus):
    """
    Extract a simplified circuit representation
    This is a placeholder for more sophisticated extraction techniques
    In a complete implementation, this would use techniques like:
    - Cross-layer transcoding/dictionary learning
    - Attribution graph construction
    """
    # In a real implementation, we would:
    # 1. Apply sparse autoencoder/dictionary learning to find features
    # 2. Construct attribution graphs to link features
    # 3. Prune to identify the minimal circuit
    
    # For now, we'll create a simplified representation based on the model structure
    n_layers = len(model.blocks)
    n_heads = model.blocks[0].attention.num_attention_heads
    
    # Create node list - include embedding, attention heads, FFNs, and output
    nodes = [{"id": "embedding", "type": "embedding", "layer": 0}]
    
    # Add attention heads and FFNs for each layer
    for l in range(n_layers):
        for h in range(n_heads):
            nodes.append({"id": f"attn_{l}_{h}", "type": "attention_head", "layer": l, "head_idx": h})
        nodes.append({"id": f"ffn_{l}", "type": "ffn", "layer": l})
    
    nodes.append({"id": "output", "type": "output", "layer": n_layers})
    
    # Create edge list - simplified connectivity
    edges = []
    
    # Connect embedding to all first layer attention heads
    for h in range(n_heads):
        edges.append({"from": "embedding", "to": f"attn_0_{h}", "weight": 1.0 / n_heads})
    
    # Connect attention heads to FFN in same layer
    for l in range(n_layers):
        for h in range(n_heads):
            edges.append({"from": f"attn_{l}_{h}", "to": f"ffn_{l}", "weight": 1.0 / n_heads})
    
    # Connect each layer's FFN to next layer's attention heads
    for l in range(n_layers - 1):
        for h in range(n_heads):
            edges.append({"from": f"ffn_{l}", "to": f"attn_{l+1}_{h}", "weight": 1.0 / n_heads})
    
    # Connect final FFN to output
    edges.append({"from": f"ffn_{n_layers-1}", "to": "output", "weight": 1.0})
    
    # Create circuit structure
    circuit_structure = {
        "nodes": nodes,
        "edges": edges
    }
    
    # Create interface definition
    interface_definition = {
        "input_format": "sequence",
        "input_tokens": ["START", "num_a", "SEP", "num_b"],
        "output_tokens": ["result"],
        "value_ranges": {"num_a": f"0-{modulus-1}", "num_b": f"0-{modulus-1}", "result": f"0-{modulus-1}"}
    }
    
    # Create model architecture description
    model_architecture = {
        "type": "transformer",
        "hidden_size": model.config.hidden_size,
        "num_layers": model.config.num_hidden_layers,
        "num_heads": model.config.num_attention_heads,
        "params": sum(p.numel() for p in model.parameters())
    }
    
    # Create interpretation
    interpretation = f"This circuit implements addition modulo {modulus}. The model learns to map input pairs (a, b) to their sum modulo {modulus}."
    
    return {
        "circuit_structure": circuit_structure,
        "interface_definition": interface_definition,
        "model_architecture": model_architecture,
        "interpretation": interpretation
    }

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Setup directories
    setup_directories(args.checkpoint_dir)
    
    print(f"Starting pipeline for task {args.task} on {args.device}")
    print("1. Generating data...")
    data_dict = load_data(args.modulus, args.train_ratio)
    
    print(f"2. Creating model with {args.num_layers} layers, {args.hidden_size} hidden size...")
    model, config = create_model(data_dict["vocab_size"], args)
    optimizer, lr_scheduler = create_optimizer(model, args.learning_rate, args.weight_decay)
    
    print(f"3. Training model...")
    trainer = AlgorithmicTaskTrainer(
        model=model,
        train_dataset=data_dict["train_dataset"],
        test_dataset=data_dict["test_dataset"],
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Start training
    start_time = time.time()
    history = trainer.train()
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f}s")
    print(f"Final test accuracy: {history['test_acc'][-1]:.4f}")
    
    # Plot and save training history
    plot_path = os.path.join(args.checkpoint_dir, f"{args.task}_training_history.png")
    plot_training_history(history, args.task, save_path=plot_path)
    
    # Extract circuit (placeholder for more sophisticated extraction)
    print("4. Extracting circuit...")
    circuit_data = extract_simple_circuit(model, args.task, args.modulus)
    
    # Store in database
    print("5. Storing circuit in database...")
    db = CircuitDatabase(args.db_path)
    
    # Create a unique ID for the circuit
    circuit_id = f"{args.task}_layers{args.num_layers}_hidden{args.hidden_size}_heads{args.num_heads}"
    
    # Add to database
    success = db.add_circuit(
        circuit_id=circuit_id,
        task_name=args.task,
        model_architecture=circuit_data["model_architecture"],
        circuit_structure=circuit_data["circuit_structure"],
        interface_definition=circuit_data["interface_definition"],
        task_description=f"Addition modulo {args.modulus}: (a + b) % {args.modulus}",
        training_details={
            "epochs": trainer.current_epoch,
            "final_accuracy": float(history["test_acc"][-1]),
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "training_time": training_time
        },
        interpretation=circuit_data["interpretation"],
        fidelity=float(history["test_acc"][-1]),
        extraction_method="simple_connectivity",
        tags=["Arithmetic", "Core_Number", "Primitive"]
    )
    
    if success:
        print(f"Circuit stored successfully with ID: {circuit_id}")
    else:
        print("Failed to store circuit in database")
    
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main() 