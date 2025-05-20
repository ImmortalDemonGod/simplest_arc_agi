import math
import pytest

# Skip tests if torch is not installed
torch = pytest.importorskip("torch")
from torch.utils.data import TensorDataset

from src.models.transformer import SimpleTransformer, TransformerConfig
from src.training.trainer import AlgorithmicTaskTrainer


def test_evaluate_returns_nan_for_empty_test_dataloader():
    config = TransformerConfig(vocab_size=10, hidden_size=8, num_hidden_layers=1,
                               num_attention_heads=2, intermediate_size=32)
    model = SimpleTransformer(config)

    # minimal train dataset so dataloader can initialize
    train_inputs = torch.randint(0, config.vocab_size - 1, (1, 2))
    train_targets = torch.randint(0, config.vocab_size - 1, (1, 2))
    train_dataset = TensorDataset(train_inputs, train_targets)

    # empty test dataset
    test_inputs = torch.empty((0, 2), dtype=torch.long)
    test_targets = torch.empty((0, 2), dtype=torch.long)
    test_dataset = TensorDataset(test_inputs, test_targets)

    trainer = AlgorithmicTaskTrainer(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=1,
        max_epochs=1,
    )
    loss, acc = trainer.evaluate()

    assert math.isnan(loss)
    assert acc == 0.0
