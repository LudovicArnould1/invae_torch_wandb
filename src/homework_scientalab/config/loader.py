"""
Utilities for loading configuration from YAML files.

Provides functions to load YAML configs and merge them with dataclass defaults.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Type, TypeVar, Optional
import yaml

from homework_scientalab.config.configs import (
    DataConfig,
    InVAEConfig,
    TrainConfig,
    WarmupSchedule,
)

T = TypeVar('T')


def load_yaml(yaml_path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML file and return its contents as a dictionary.
    
    Args:
        yaml_path: Path to YAML file
        
    Returns:
        Dictionary with YAML contents
        
    Raises:
        FileNotFoundError: If YAML file doesn't exist
        yaml.YAMLError: If YAML is malformed
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return config_dict or {}


def _convert_lists_to_tuples(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert list values to tuples for fields that expect tuples.
    
    YAML doesn't have tuples, so we represent them as lists and convert here.
    
    Args:
        config_dict: Configuration dictionary from YAML
        
    Returns:
        Dictionary with lists converted to tuples where appropriate
    """
    tuple_fields = {'enc_hidden', 'dec_hidden', 'prior_hidden'}
    
    result = {}
    for key, value in config_dict.items():
        if key in tuple_fields and isinstance(value, list):
            result[key] = tuple(value)
        elif isinstance(value, dict):
            # Recursively handle nested dicts (e.g., warmup_schedule)
            result[key] = _convert_lists_to_tuples(value)
        else:
            result[key] = value
    
    return result


def load_config(
    config_cls: Type[T],
    yaml_path: Optional[str | Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> T:
    """
    Load configuration from YAML file and/or overrides.
    
    Priority (highest to lowest):
    1. overrides dict
    2. YAML file
    3. dataclass defaults
    
    Args:
        config_cls: Dataclass type to instantiate
        yaml_path: Optional path to YAML config file
        overrides: Optional dictionary of config overrides
        
    Returns:
        Instance of config_cls with merged configuration
        
    Example:
        >>> cfg = load_config(DataConfig, "config/data.yaml", {"batch_size": 512})
    """
    # Start with empty dict
    config_dict = {}
    
    # Load from YAML if provided
    if yaml_path is not None:
        config_dict.update(load_yaml(yaml_path))
    
    # Apply overrides
    if overrides is not None:
        config_dict.update(overrides)
    
    # Convert lists to tuples where needed
    config_dict = _convert_lists_to_tuples(config_dict)
    
    # Handle nested configs (e.g., WarmupSchedule in TrainConfig)
    if config_cls == TrainConfig and 'warmup_schedule' in config_dict:
        warmup_dict = config_dict['warmup_schedule']
        if isinstance(warmup_dict, dict):
            config_dict['warmup_schedule'] = WarmupSchedule(**warmup_dict)
    
    # Instantiate dataclass
    return config_cls(**config_dict)


def load_data_config(
    yaml_path: Optional[str | Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> DataConfig:
    """
    Load DataConfig from YAML file.
    
    Args:
        yaml_path: Path to data config YAML file
        overrides: Optional dictionary of config overrides
        
    Returns:
        DataConfig instance
    """
    if yaml_path is None:
        # Default path relative to this file
        config_dir = Path(__file__).parent
        yaml_path = config_dir / "data_config.yaml"
    
    return load_config(DataConfig, yaml_path, overrides)


def load_model_config(
    yaml_path: Optional[str | Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> InVAEConfig:
    """
    Load InVAEConfig from YAML file.
    
    Note: x_dim, b_dim, t_dim are typically set dynamically from data,
    so you'll likely need to provide them via overrides.
    
    Args:
        yaml_path: Path to model config YAML file
        overrides: Optional dictionary of config overrides (must include x_dim, b_dim, t_dim)
        
    Returns:
        InVAEConfig instance
    """
    if yaml_path is None:
        # Default path relative to this file
        config_dir = Path(__file__).parent
        yaml_path = config_dir / "model_config.yaml"
    
    return load_config(InVAEConfig, yaml_path, overrides)


def load_train_config(
    yaml_path: Optional[str | Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> TrainConfig:
    """
    Load TrainConfig from YAML file.
    
    Args:
        yaml_path: Path to train config YAML file
        overrides: Optional dictionary of config overrides
        
    Returns:
        TrainConfig instance
    """
    if yaml_path is None:
        # Default path relative to this file
        config_dir = Path(__file__).parent
        yaml_path = config_dir / "train_config.yaml"
    
    return load_config(TrainConfig, yaml_path, overrides)

