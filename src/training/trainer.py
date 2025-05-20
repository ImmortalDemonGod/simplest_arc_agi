import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import os
import sys
from typing import Dict, List, Tuple, Optional, Callable

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
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        batch_size: int = 64,
        max_epochs: int = 10000,
        patience: int = 1000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        log_interval: int = 100,
        checkpoint_dir: Optional[str] = None
    ):
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
            "learning_rate": []
        }
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch and return average loss and accuracy"""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, (input_ids, target_ids) in enumerate(self.train_dataloader):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(input_ids)
            
            # Compute loss - reshape logits for cross entropy
            # logits: [batch_size, seq_len, vocab_size]
            # target_ids: [batch_size, seq_len]
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
                print(f"Epoch {self.current_epoch} [{batch_idx}/{len(self.train_dataloader)}] " 
                      f"Loss: {loss.item():.6f}, Acc: {correct/total:.4f}")
            
            self.global_step += 1
        
        # Compute average loss and accuracy
        avg_loss = total_loss / len(self.train_dataloader)
        avg_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return avg_loss, avg_accuracy
    
    def evaluate(self) -> Tuple[float, float]:
        """Evaluate model on test set"""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for input_ids, target_ids in self.test_dataloader:
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                # Forward pass
                logits = self.model(input_ids)
                
                # Compute loss
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
        num_batches = len(self.test_dataloader)
        if num_batches == 0:
            return float("nan"), 0.0

        avg_loss = total_loss / num_batches
        avg_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
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