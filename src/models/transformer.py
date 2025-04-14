import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class TransformerConfig:
    """Configuration class for a small transformer model"""
    def __init__(
        self,
        vocab_size: int = 120,  # Enough for modular math with small moduli + special tokens
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
        self_modeling_target_layers: Optional[list[str]] = None,  # List of target layers for multi-layer self-modeling
        self_modeling_loss_weight: float = 1.0,  # Weight for self-modeling loss (reduced from 5.0)
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.pad_token_id = pad_token_id
        
        # Self-modeling parameters
        self.use_self_modeling = use_self_modeling
        self.self_modeling_target_layer = self_modeling_target_layer
        self.self_modeling_target_layers = self_modeling_target_layers
        self.self_modeling_loss_weight = self_modeling_loss_weight


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module"""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        
    def transpose_for_scores(self, x):
        """Reshape for multi-head attention computation"""
        batch_size, seq_len = x.size(0), x.size(1)
        x = x.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        # (batch_size, num_heads, seq_len, head_size)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None):
        # hidden_states: (batch_size, seq_len, hidden_size)
        batch_size, seq_len = hidden_states.size(0), hidden_states.size(1)
        
        # Linear projections
        query = self.query(hidden_states)  # (batch_size, seq_len, all_head_size)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        
        # Reshape for multi-head attention
        query = self.transpose_for_scores(query)  # (batch_size, num_heads, seq_len, head_size)
        key = self.transpose_for_scores(key)
        value = self.transpose_for_scores(value)
        
        # Compute attention scores: (batch_size, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask: (batch_size, 1, 1, seq_len) with 0 for masked positions and 1 elsewhere
            attention_scores = attention_scores + attention_mask
        
        # Apply softmax and dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, value)  # (batch_size, num_heads, seq_len, head_size)
        context = context.permute(0, 2, 1, 3).contiguous()  # (batch_size, seq_len, num_heads, head_size)
        context = context.view(batch_size, seq_len, self.all_head_size)  # (batch_size, seq_len, hidden_size)
        
        # Final linear projection
        output = self.out(context)
        
        return output


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward layers"""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )
        
    def forward(self, hidden_states, attention_mask=None):
        # Self-attention with residual connection and layer norm
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = self.layer_norm1(hidden_states + attention_output)
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(hidden_states)
        hidden_states = self.layer_norm2(hidden_states + ff_output)
        
        return hidden_states

class SelfModelingHead(nn.Module):
    """Enhanced auxiliary head for predicting activations from a target layer"""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        # More sophisticated architecture with intermediate size and layer normalization
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.layer_norm1 = nn.LayerNorm(config.intermediate_size)
        # Using GELU activation instead of ReLU for better performance
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        
        # Intermediate layer
        self.dense2 = nn.Linear(config.intermediate_size, config.intermediate_size)
        self.layer_norm2 = nn.LayerNorm(config.intermediate_size)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)
        
        # Output layer
        self.dense3 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm3 = nn.LayerNorm(config.hidden_size)
        self.dropout3 = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states):
        # First layer
        x = self.dense1(hidden_states)
        x = self.layer_norm1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        
        # Intermediate layer with residual connection
        residual = x
        x = self.dense2(x)
        x = self.layer_norm2(x)
        x = self.activation(x)
        x = self.dropout2(x)
        x = x + residual  # Residual connection
        
        # Output layer
        x = self.dense3(x)
        x = self.layer_norm3(x)
        x = self.dropout3(x)
        
        return x


class SimpleTransformer(nn.Module):
    """Basic transformer model for algorithmic tasks"""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        
        # Position embeddings
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Output layer
        self.output = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Add self-modeling heads if enabled
        if config.use_self_modeling:
            if config.self_modeling_target_layers is not None:
                # Create a separate head for each target layer
                self.self_modeling_heads = nn.ModuleDict()
                for layer_name in config.self_modeling_target_layers:
                    self.self_modeling_heads[layer_name] = SelfModelingHead(config)
            else:
                # Single head for backward compatibility
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
        # (batch_size, 1, 1, seq_len) with 0 for masked positions and -10000 elsewhere
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Get token and position embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        
        # Combine embeddings
        embeddings = token_embeds + position_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Store target layer activations if self-modeling is enabled
        target_activations = {}
        
        # Pass through transformer blocks
        hidden_states = embeddings
        for i, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, extended_attention_mask)
            
            # Store activations from target layers
            if hasattr(self.config, 'use_self_modeling') and self.config.use_self_modeling:
                layer_name = f"layer_{i}"
                
                # For multi-layer self-modeling
                if self.config.self_modeling_target_layers is not None:
                    if layer_name in self.config.self_modeling_target_layers:
                        target_activations[layer_name] = hidden_states.clone()
                    if "middle" in self.config.self_modeling_target_layers and i == len(self.blocks) // 2:
                        target_activations["middle"] = hidden_states.clone()
                    if "last_hidden" in self.config.self_modeling_target_layers and i == len(self.blocks) - 1:
                        target_activations["last_hidden"] = hidden_states.clone()
                # For single-layer self-modeling (backward compatibility)
                else:
                    if (self.config.self_modeling_target_layer == "last_hidden" and i == len(self.blocks) - 1) or \
                       (self.config.self_modeling_target_layer == layer_name) or \
                       (self.config.self_modeling_target_layer == "middle" and i == len(self.blocks) // 2):
                        target_activations[self.config.self_modeling_target_layer] = hidden_states.clone()
        
        # Output layer
        logits = self.output(hidden_states)
        
        # If self-modeling is enabled, predict target activations
        if hasattr(self.config, 'use_self_modeling') and self.config.use_self_modeling and target_activations:
            if self.config.self_modeling_target_layers is not None:
                # Multi-layer self-modeling
                predicted_activations = {}
                for layer_name, activation in target_activations.items():
                    predicted_activations[layer_name] = self.self_modeling_heads[layer_name](hidden_states)
            else:
                # Single-layer self-modeling (backward compatibility)
                layer_name = next(iter(target_activations.keys()))
                predicted_activations = {layer_name: self.self_modeling_head(hidden_states)}
            
            return logits, target_activations, predicted_activations
        
        return logits


if __name__ == "__main__":
    # Example usage
    config = TransformerConfig(
        vocab_size=120,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=512
    )
    
    model = SimpleTransformer(config)
    
    # Create a sample batch
    batch_size = 4
    seq_len = 5
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    attention_mask = (input_ids != config.pad_token_id).float()
    
    # Forward pass
    logits = model(input_ids, attention_mask)
    
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}") 