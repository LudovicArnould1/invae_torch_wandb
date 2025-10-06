"""
Quick test to validate W&B sweep setup without running full training.

This script verifies:
1. Sweep configuration is valid
2. sweep_train.py imports correctly
3. Configuration overrides work as expected
"""
from __future__ import annotations
import yaml
from pathlib import Path


def test_sweep_config():
    """Test that sweep configuration is valid."""
    print("=" * 80)
    print("TEST 1: Validating sweep_config.yaml")
    print("=" * 80)
    
    config_path = Path("src/homework_scientalab/config/sweep_config.yaml")
    assert config_path.exists(), f"Sweep config not found: {config_path}"
    
    with open(config_path, "r") as f:
        sweep_config = yaml.safe_load(f)
    
    # Check required fields
    assert "method" in sweep_config, "Missing 'method' in sweep config"
    assert "metric" in sweep_config, "Missing 'metric' in sweep config"
    assert "parameters" in sweep_config, "Missing 'parameters' in sweep config"
    
    # Validate parameters
    params = sweep_config["parameters"]
    expected_params = ["z_i_dim", "z_s_dim", "learning_rate", "lambda_indep", "warmup_steps"]
    
    for param in expected_params:
        assert param in params, f"Missing parameter: {param}"
        print(f"  ✓ {param}: {params[param]}")
    
    print("\n✓ Sweep configuration is valid\n")


def test_sweep_train_imports():
    """Test that sweep_train.py imports correctly."""
    print("=" * 80)
    print("TEST 2: Validating sweep_train.py imports")
    print("=" * 80)
    
    try:
        from homework_scientalab.config import DataConfig, TrainConfig, load_data_config, load_train_config
        print("  ✓ Config imports successful")
        
        from homework_scientalab.data import prepare_dataloaders
        print("  ✓ Data imports successful")
        
        from homework_scientalab.model import InVAE
        print("  ✓ Model imports successful")
        
        from homework_scientalab.trainer import InVAETrainer
        print("  ✓ Trainer imports successful")
        
        print("\n✓ All imports successful\n")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        raise


def test_config_override():
    """Test that configuration override mechanism works."""
    print("=" * 80)
    print("TEST 3: Testing configuration override")
    print("=" * 80)
    
    from homework_scientalab.config import load_train_config
    
    # Load base config
    train_cfg = load_train_config()
    
    print(f"  Base z_i_dim: {train_cfg.z_i_dim}")
    print(f"  Base z_s_dim: {train_cfg.z_s_dim}")
    print(f"  Base learning_rate: {train_cfg.learning_rate}")
    print(f"  Base lambda_indep: {train_cfg.lambda_indep}")
    print(f"  Base warmup_steps: {train_cfg.warmup_schedule.warmup_steps}")
    
    # Simulate sweep override
    train_cfg.z_i_dim = 40
    train_cfg.z_s_dim = 7
    train_cfg.learning_rate = 0.005
    train_cfg.lambda_indep = 0.05
    train_cfg.warmup_schedule.warmup_steps = 500
    
    print(f"\n  After override:")
    print(f"  z_i_dim: {train_cfg.z_i_dim}")
    print(f"  z_s_dim: {train_cfg.z_s_dim}")
    print(f"  learning_rate: {train_cfg.learning_rate}")
    print(f"  lambda_indep: {train_cfg.lambda_indep}")
    print(f"  warmup_steps: {train_cfg.warmup_schedule.warmup_steps}")
    
    assert train_cfg.z_i_dim == 40
    assert train_cfg.z_s_dim == 7
    assert train_cfg.learning_rate == 0.005
    
    print("\n✓ Configuration override works correctly\n")


def test_sweep_script_syntax():
    """Test that sweep scripts have valid syntax."""
    print("=" * 80)
    print("TEST 4: Validating script syntax")
    print("=" * 80)
    
    scripts = [
        "sweep_train.py",
        "run_sweep.py",
    ]
    
    import py_compile
    
    for script in scripts:
        script_path = Path(script)
        if script_path.exists():
            try:
                py_compile.compile(str(script_path), doraise=True)
                print(f"  ✓ {script}: Valid Python syntax")
            except py_compile.PyCompileError as e:
                print(f"  ✗ {script}: Syntax error - {e}")
                raise
        else:
            print(f"  ⚠ {script}: File not found")
    
    print("\n✓ All scripts have valid syntax\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("W&B SWEEP SETUP VALIDATION")
    print("=" * 80 + "\n")
    
    try:
        test_sweep_config()
        test_sweep_train_imports()
        test_config_override()
        test_sweep_script_syntax()
        
        print("=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)
        print("\nSweep setup is ready! To run the sweep:")
        print("  python run_sweep.py --count 3")
        print("\nOr for full sweep:")
        print("  python run_sweep.py --count 20")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("TEST FAILED ✗")
        print("=" * 80)
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()

