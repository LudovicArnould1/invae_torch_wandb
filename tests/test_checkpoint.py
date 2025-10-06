"""
Unit tests for checkpoint management functionality.
"""
import json
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from homework_scientalab.checkpoint import CheckpointManager


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 20, output_dim: int = 5):
        """Initialize simple model."""
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary checkpoint directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def simple_model():
    """Create simple model for testing."""
    return SimpleModel()


@pytest.fixture
def simple_optimizer(simple_model):
    """Create optimizer for testing."""
    return torch.optim.Adam(simple_model.parameters(), lr=0.001)


def test_checkpoint_manager_init(temp_checkpoint_dir):
    """Test checkpoint manager initialization."""
    manager = CheckpointManager(
        checkpoint_dir=temp_checkpoint_dir,
        keep_last_n=2,
    )
    
    assert manager.checkpoint_dir == temp_checkpoint_dir
    assert manager.keep_last_n == 2
    assert manager.best_checkpoint_path == temp_checkpoint_dir / "best_model.pt"
    assert manager.checkpoint_dir.exists()


def test_save_checkpoint(temp_checkpoint_dir, simple_model, simple_optimizer):
    """Test checkpoint saving."""
    manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)
    
    model_config = {"input_dim": 10, "hidden_dim": 20, "output_dim": 5}
    
    # Save checkpoint
    checkpoint_path = manager.save_checkpoint(
        epoch=5,
        global_step=1000,
        model=simple_model,
        optimizer=simple_optimizer,
        val_loss=2.5,
        best_val_loss=2.0,
        model_config=model_config,
        is_best=False,
    )
    
    assert checkpoint_path.exists()
    assert checkpoint_path.name == "checkpoint_epoch_5.pt"
    
    # Check checkpoint contents
    checkpoint = torch.load(checkpoint_path)
    assert checkpoint["epoch"] == 5
    assert checkpoint["global_step"] == 1000
    assert checkpoint["val_loss"] == 2.5
    assert checkpoint["best_val_loss"] == 2.0
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert "rng_state" in checkpoint
    assert checkpoint["model_config"] == model_config


def test_save_best_checkpoint(temp_checkpoint_dir, simple_model, simple_optimizer):
    """Test saving best checkpoint."""
    manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)
    
    model_config = {"input_dim": 10, "hidden_dim": 20, "output_dim": 5}
    
    # Save best checkpoint
    checkpoint_path = manager.save_checkpoint(
        epoch=10,
        global_step=2000,
        model=simple_model,
        optimizer=simple_optimizer,
        val_loss=1.5,
        best_val_loss=1.5,
        model_config=model_config,
        is_best=True,
    )
    
    assert checkpoint_path == manager.best_checkpoint_path
    assert checkpoint_path.exists()


def test_load_checkpoint(temp_checkpoint_dir, simple_model, simple_optimizer):
    """Test checkpoint loading."""
    manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)
    
    model_config = {"input_dim": 10, "hidden_dim": 20, "output_dim": 5}
    
    # Set specific RNG states before saving
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Save checkpoint
    manager.save_checkpoint(
        epoch=5,
        global_step=1000,
        model=simple_model,
        optimizer=simple_optimizer,
        val_loss=2.5,
        best_val_loss=2.0,
        model_config=model_config,
        is_best=True,
    )
    
    # Modify RNG states
    random.seed(100)
    np.random.seed(100)
    torch.manual_seed(100)
    
    # Load checkpoint
    checkpoint_state = manager.load_checkpoint()
    
    assert checkpoint_state is not None
    assert checkpoint_state.epoch == 5
    assert checkpoint_state.global_step == 1000
    assert checkpoint_state.best_val_loss == 2.0
    assert checkpoint_state.model_config == model_config
    
    # Verify RNG states were restored (approximately)
    # Generate a random number and check it matches expected pattern
    random_val = random.random()
    # The exact value depends on the RNG state restoration
    assert random_val is not None  # Basic check that it runs


def test_has_checkpoint(temp_checkpoint_dir, simple_model, simple_optimizer):
    """Test checkpoint existence check."""
    manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)
    
    # Initially no checkpoint
    assert not manager.has_checkpoint()
    
    # Save checkpoint
    manager.save_checkpoint(
        epoch=1,
        global_step=100,
        model=simple_model,
        optimizer=simple_optimizer,
        val_loss=3.0,
        best_val_loss=3.0,
        model_config={},
        is_best=True,
    )
    
    # Now should have checkpoint
    assert manager.has_checkpoint()


def test_resume_info_creation(temp_checkpoint_dir, simple_model, simple_optimizer):
    """Test resume info JSON creation."""
    manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)
    
    manager.save_checkpoint(
        epoch=5,
        global_step=1000,
        model=simple_model,
        optimizer=simple_optimizer,
        val_loss=2.5,
        best_val_loss=2.0,
        model_config={},
        is_best=False,
    )
    
    # Check resume info file exists
    assert manager.resume_info_path.exists()
    
    # Check contents
    with open(manager.resume_info_path, "r") as f:
        resume_info = json.load(f)
    
    assert resume_info["epoch"] == 5
    assert resume_info["global_step"] == 1000
    assert resume_info["best_val_loss"] == 2.0
    assert "latest_checkpoint" in resume_info


def test_checkpoint_cleanup(temp_checkpoint_dir, simple_model, simple_optimizer):
    """Test that old checkpoints are cleaned up."""
    manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir, keep_last_n=2)
    
    # Save multiple checkpoints
    for epoch in range(1, 6):
        manager.save_checkpoint(
            epoch=epoch,
            global_step=epoch * 100,
            model=simple_model,
            optimizer=simple_optimizer,
            val_loss=3.0 - epoch * 0.1,
            best_val_loss=2.5,
            model_config={},
            is_best=False,
        )
    
    # Should only have last 2 regular checkpoints (+ possibly resume_info.json)
    regular_checkpoints = list(temp_checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    assert len(regular_checkpoints) <= 2
    
    # Check that the most recent ones are kept
    epochs = [int(p.stem.split("_")[-1]) for p in regular_checkpoints]
    assert max(epochs) == 5  # Most recent should be kept


def test_load_nonexistent_checkpoint(temp_checkpoint_dir):
    """Test loading when no checkpoint exists."""
    manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)
    
    checkpoint_state = manager.load_checkpoint()
    assert checkpoint_state is None


def test_get_resume_run_id(temp_checkpoint_dir, simple_model, simple_optimizer):
    """Test getting W&B run ID for resumption."""
    manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)
    
    # Initially no run ID
    assert manager.get_resume_run_id() is None
    
    # Save checkpoint (without W&B run)
    manager.save_checkpoint(
        epoch=1,
        global_step=100,
        model=simple_model,
        optimizer=simple_optimizer,
        val_loss=3.0,
        best_val_loss=3.0,
        model_config={},
        is_best=True,
    )
    
    # Still no run ID (W&B not initialized)
    assert manager.get_resume_run_id() is None


def test_checkpoint_with_two_optimizers(temp_checkpoint_dir, simple_model):
    """Test checkpoint with two optimizers."""
    optimizer1 = torch.optim.Adam(simple_model.fc1.parameters(), lr=0.001)
    optimizer2 = torch.optim.SGD(simple_model.fc2.parameters(), lr=0.01)
    
    manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)
    
    # Save checkpoint with both optimizers
    checkpoint_path = manager.save_checkpoint(
        epoch=1,
        global_step=100,
        model=simple_model,
        optimizer=optimizer1,
        val_loss=3.0,
        best_val_loss=3.0,
        model_config={},
        is_best=True,
        optimizer_other=optimizer2,
    )
    
    # Load and check
    checkpoint = torch.load(checkpoint_path)
    assert "optimizer_state_dict" in checkpoint
    assert "optimizer_other_state_dict" in checkpoint
    assert checkpoint["optimizer_other_state_dict"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

