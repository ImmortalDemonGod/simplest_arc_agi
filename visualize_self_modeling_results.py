#!/usr/bin/env python3
import torch
import matplotlib.pyplot as plt
import os
import argparse
import glob
import numpy as np
from collections import defaultdict

def load_checkpoint(checkpoint_path):
    """Load training history from a checkpoint file"""
    try:
        # Set weights_only=False to handle numpy scalar objects in the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        return checkpoint.get('training_history', {})
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        print("Note: If you're using PyTorch 2.6+, you may need to add numpy scalar to safe globals.")
        print("For now, we'll continue without loading this checkpoint.")
        return {}

def find_latest_checkpoint(directory):
    """Find the latest checkpoint in a directory"""
    checkpoints = glob.glob(os.path.join(directory, "*.pt"))
    if not checkpoints:
        return None
    
    # Sort by modification time (newest first)
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    return checkpoints[0]

def load_histories_from_checkpoints(base_dir, variants=None):
    """Load training histories from checkpoint files for each variant"""
    if variants is None:
        # Try to detect variants from subdirectories
        variants = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    histories = {}
    for variant in variants:
        variant_dir = os.path.join(base_dir, variant)
        if not os.path.isdir(variant_dir):
            print(f"Warning: {variant_dir} is not a directory, skipping")
            continue
        
        checkpoint_path = find_latest_checkpoint(variant_dir)
        if checkpoint_path:
            print(f"Loading checkpoint for {variant}: {os.path.basename(checkpoint_path)}")
            history = load_checkpoint(checkpoint_path)
            if history:
                histories[variant] = history
        else:
            print(f"No checkpoint found for {variant}")
    
    return histories

def plot_comparison(histories, variants, output_dir=None):
    """Create comparison plots for all model variants"""
    if not output_dir:
        output_dir = "."
    os.makedirs(output_dir, exist_ok=True)
    
    # Define metrics to plot
    metrics = [
        {"name": "test_acc", "title": "Test Accuracy", "ylabel": "Accuracy"},
        {"name": "train_acc", "title": "Train Accuracy", "ylabel": "Accuracy"},
        {"name": "test_loss", "title": "Test Loss", "ylabel": "Loss"},
        {"name": "train_loss", "title": "Train Loss", "ylabel": "Loss"},
        {"name": "weight_std", "title": "Weight Standard Deviation", "ylabel": "Std Dev"},
    ]
    
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
            if variant not in histories:
                continue
                
            # Skip self_modeling_loss for baseline
            if metric["name"] == "self_modeling_loss" and variant == "baseline":
                continue
                
            history = histories[variant]
            
            # Skip if metric not in history
            if metric["name"] not in history:
                continue
                
            # For RLCT, we need to handle sparse data points
            if metric["name"] == "rlct":
                # RLCT is measured every 10 epochs, so we need to create corresponding x-axis values
                rlct_epochs = [epoch for i, epoch in enumerate(history["epoch"]) if i % 10 == 0]
                if len(rlct_epochs) > 0 and len(history[metric["name"]]) > 0:
                    plt.plot(rlct_epochs, history[metric["name"]], label=variant.replace("_", " ").title())
            else:
                if len(history["epoch"]) > 0 and len(history[metric["name"]]) > 0:
                    plt.plot(history["epoch"], history[metric["name"]], label=variant.replace("_", " ").title())
        
        plt.xlabel("Epoch")
        plt.ylabel(metric["ylabel"])
        plt.title(f"{metric['title']} Comparison")
        plt.legend()
        plt.grid(True)
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f"comparison_{metric['name']}.png"))
        plt.close()
    
    # Create a combined plot for test accuracy and loss
    plt.figure(figsize=(12, 10))
    
    # Test accuracy subplot
    plt.subplot(2, 1, 1)
    for variant in variants:
        if variant not in histories:
            continue
        history = histories[variant]
        if len(history["epoch"]) > 0 and len(history["test_acc"]) > 0:
            plt.plot(history["epoch"], history["test_acc"], label=variant.replace("_", " ").title())
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy Comparison")
    plt.legend()
    plt.grid(True)
    
    # Test loss subplot
    plt.subplot(2, 1, 2)
    for variant in variants:
        if variant not in histories:
            continue
        history = histories[variant]
        if len(history["epoch"]) > 0 and len(history["test_loss"]) > 0:
            plt.plot(history["epoch"], history["test_loss"], label=variant.replace("_", " ").title())
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Test Loss Comparison")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_combined.png"))
    plt.close()
    
    # Create a learning dynamics plot (accuracy vs. loss)
    plt.figure(figsize=(10, 8))
    for variant in variants:
        if variant not in histories:
            continue
        history = histories[variant]
        if len(history["test_loss"]) > 0 and len(history["test_acc"]) > 0:
            plt.plot(history["test_loss"], history["test_acc"], 'o-', label=variant.replace("_", " ").title(), alpha=0.7)
    plt.xlabel("Test Loss")
    plt.ylabel("Test Accuracy")
    plt.title("Learning Dynamics: Accuracy vs. Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "learning_dynamics.png"))
    plt.close()
    
    # If self-modeling loss is available, create a plot comparing it across variants
    if any("self_modeling_loss" in h for h in histories.values()):
        plt.figure(figsize=(10, 6))
        for variant in variants:
            if variant not in histories or variant == "baseline":
                continue
            history = histories[variant]
            if "self_modeling_loss" not in history:
                print(f"Warning: self_modeling_loss not found in {variant} history")
                continue
            if len(history["epoch"]) > 0 and len(history["self_modeling_loss"]) > 0:
                plt.plot(history["epoch"], history["self_modeling_loss"], label=variant.replace("_", " ").title())
            else:
                print(f"Warning: Empty epoch or self_modeling_loss arrays for {variant}")
        plt.xlabel("Epoch")
        plt.ylabel("Self-Modeling Loss")
        plt.title("Self-Modeling Loss Comparison")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "comparison_self_modeling_loss.png"))
        plt.close()
    
    print(f"Comparison plots saved to {output_dir}")

def print_final_metrics(histories, variants):
    """Print a table of final metrics for each variant"""
    print("\n=== Final Metrics ===")
    
    # Print header
    metrics = ["Test Acc", "Train Acc", "Test Loss", "Train Loss"]
    if any("val_acc" in h for h in histories.values()):
        metrics.append("Val Acc")
    if any("val_loss" in h for h in histories.values()):
        metrics.append("Val Loss")
    if any("weight_std" in h for h in histories.values()):
        metrics.append("Weight Std")
    if any("rlct" in h for h in histories.values()):
        metrics.append("RLCT")
    if any("self_modeling_loss" in h for h in histories.values()):
        metrics.append("Self-Modeling Loss")
    
    header = "Variant".ljust(20)
    for metric in metrics:
        header += metric.ljust(15)
    print(header)
    print("-" * len(header))
    
    # Print each variant's metrics
    for variant in variants:
        if variant not in histories:
            continue
            
        history = histories[variant]
        line = variant.replace("_", " ").title().ljust(20)
        
        # Add each metric
        if "test_acc" in history:
            line += f"{history['test_acc'][-1]:.4f}".ljust(15)
        else:
            line += "N/A".ljust(15)
            
        if "train_acc" in history:
            line += f"{history['train_acc'][-1]:.4f}".ljust(15)
        else:
            line += "N/A".ljust(15)
            
        if "test_loss" in history:
            line += f"{history['test_loss'][-1]:.4f}".ljust(15)
        else:
            line += "N/A".ljust(15)
            
        if "train_loss" in history:
            line += f"{history['train_loss'][-1]:.4f}".ljust(15)
        else:
            line += "N/A".ljust(15)
            
        if "Val Acc" in metrics and "val_acc" in history:
            line += f"{history['val_acc'][-1]:.4f}".ljust(15)
        elif "Val Acc" in metrics:
            line += "N/A".ljust(15)
            
        if "Val Loss" in metrics and "val_loss" in history:
            line += f"{history['val_loss'][-1]:.4f}".ljust(15)
        elif "Val Loss" in metrics:
            line += "N/A".ljust(15)
            
        if "weight_std" in metrics and "weight_std" in history:
            line += f"{history['weight_std'][-1]:.4f}".ljust(15)
        elif "weight_std" in metrics:
            line += "N/A".ljust(15)
            
        if "rlct" in metrics and "rlct" in history:
            line += f"{history['rlct'][-1]:.4f}".ljust(15)
        elif "rlct" in metrics:
            line += "N/A".ljust(15)
            
        if "self_modeling_loss" in metrics and "self_modeling_loss" in history:
            line += f"{history['self_modeling_loss'][-1]:.4f}".ljust(15)
        elif "self_modeling_loss" in metrics:
            line += "N/A".ljust(15)
            
        print(line)

def main():
    parser = argparse.ArgumentParser(description="Visualize self-modeling training results")
    parser.add_argument("--base_dir", type=str, default="checkpoints/self_modeling_comparison",
                        help="Base directory containing model checkpoints")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save visualization plots (defaults to base_dir)")
    parser.add_argument("--variants", type=str, nargs="+",
                        default=["baseline", "first_layer", "middle_layer", "last_layer", "all_layers"],
                        help="Model variants to include in visualization")
    
    args = parser.parse_args()
    
    # Set output directory to base_dir if not specified
    if args.output_dir is None:
        args.output_dir = args.base_dir
    
    print(f"Loading checkpoints from {args.base_dir}")
    histories = load_histories_from_checkpoints(args.base_dir, args.variants)
    
    if not histories:
        print("No valid checkpoints found. Run compare_self_modeling_locations.py first.")
        return
    
    # Get the list of variants that have valid histories
    valid_variants = list(histories.keys())
    print(f"Found valid histories for: {', '.join(valid_variants)}")
    
    # Print final metrics
    print_final_metrics(histories, valid_variants)
    
    # Create visualization plots
    plot_comparison(histories, valid_variants, args.output_dir)
    
    print(f"Visualization complete. Plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()