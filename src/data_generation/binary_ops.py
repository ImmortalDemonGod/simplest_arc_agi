import numpy as np
import torch
from typing import Dict, List, Tuple

def generate_modular_addition_data(modulus: int, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Dict[str, torch.Tensor]:
    """
    Generate a dataset for modular addition: (a + b) % modulus with industry-standard split
    
    Args:
        modulus: The modulus for the addition operation
        train_ratio: Fraction of examples to use for training (default: 0.7 or 70%)
        val_ratio: Fraction of examples to use for validation (default: 0.15 or 15%)
                  The remaining (1 - train_ratio - val_ratio) will be used for testing
    
    Returns:
        Dictionary containing input and target tensors for train, validation, and test sets
    """
    # Generate all possible a,b pairs
    all_pairs = [(a, b) for a in range(modulus) for b in range(modulus)]
    all_inputs = torch.tensor(all_pairs, dtype=torch.long)
    
    # Calculate targets: (a + b) % modulus
    all_targets = (all_inputs[:, 0] + all_inputs[:, 1]) % modulus
    
    # Shuffle the data
    indices = torch.randperm(len(all_pairs))
    all_inputs = all_inputs[indices]
    all_targets = all_targets[indices]
    
    # Calculate split sizes
    total_size = len(all_pairs)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    # test_size will be the remainder
    
    # Split into train, validation, and test sets
    train_inputs = all_inputs[:train_size]
    train_targets = all_targets[:train_size]
    
    val_inputs = all_inputs[train_size:train_size+val_size]
    val_targets = all_targets[train_size:train_size+val_size]
    
    test_inputs = all_inputs[train_size+val_size:]
    test_targets = all_targets[train_size+val_size:]
    
    return {
        "train_inputs": train_inputs,
        "train_targets": train_targets,
        "val_inputs": val_inputs,
        "val_targets": val_targets,
        "test_inputs": test_inputs,
        "test_targets": test_targets
    }

def format_for_transformer(inputs: torch.Tensor, targets: torch.Tensor, 
                          start_token: int, sep_token: int, pad_token: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Format binary operation data for a transformer model
    
    Args:
        inputs: Tensor of shape [N, 2] containing input pairs
        targets: Tensor of shape [N] containing target values
        start_token: Token ID for sequence start
        sep_token: Token ID for separating inputs
        pad_token: Token ID for padding
    
    Returns:
        input_seqs: Tensor of shape [N, seq_len] with format [start, a, sep, b, pad, ...]
        target_seqs: Tensor of shape [N, seq_len] with format [pad, pad, pad, pad, c, pad, ...]
    """
    batch_size = inputs.shape[0]
    seq_len = 5  # [start, a, sep, b, target]
    
    input_seqs = torch.full((batch_size, seq_len), pad_token, dtype=torch.long)
    target_seqs = torch.full((batch_size, seq_len), -100, dtype=torch.long)  # -100 is ignore_index for CrossEntropyLoss
    
    # Fill input sequences: [start, a, sep, b, pad]
    input_seqs[:, 0] = start_token
    input_seqs[:, 1] = inputs[:, 0]
    input_seqs[:, 2] = sep_token
    input_seqs[:, 3] = inputs[:, 1]
    
    # Fill target sequences: [-100, -100, -100, -100, c]
    # Only predict the final output token
    target_seqs[:, 4] = targets
    
    return input_seqs, target_seqs

if __name__ == "__main__":
    # Example usage
    data = generate_modular_addition_data(modulus=11, train_ratio=0.5)
    
    # Define special tokens
    START_TOKEN = 100
    SEP_TOKEN = 101
    PAD_TOKEN = 102
    
    # Format for transformer
    train_input_seqs, train_target_seqs = format_for_transformer(
        data["train_inputs"], data["train_targets"], START_TOKEN, SEP_TOKEN, PAD_TOKEN
    )
    
    print(f"Generated {len(data['train_inputs'])} training examples and {len(data['test_inputs'])} test examples")
    print(f"Input sequence shape: {train_input_seqs.shape}")
    print(f"Target sequence shape: {train_target_seqs.shape}")
    print(f"Sample input sequence: {train_input_seqs[0]}")
    print(f"Sample target sequence: {train_target_seqs[0]}") 