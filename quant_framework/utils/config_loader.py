"""
Configuration loading utilities.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary with configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If file format is not supported
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Determine file type from extension
    if config_file.suffix in ['.yaml', '.yml']:
        return load_yaml(config_path)
    elif config_file.suffix == '.json':
        return load_json(config_path)
    else:
        raise ValueError(
            f"Unsupported config file format: {config_file.suffix}. "
            "Use .yaml, .yml, or .json"
        )


def load_yaml(yaml_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        yaml_path: Path to YAML file
        
    Returns:
        Dictionary with configuration
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config or {}


def load_json(json_path: str) -> Dict[str, Any]:
    """
    Load JSON configuration file.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Dictionary with configuration
    """
    with open(json_path, 'r') as f:
        config = json.load(f)
    
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save configuration
    """
    output_file = Path(output_path)
    
    # Determine file type from extension
    if output_file.suffix in ['.yaml', '.yml']:
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    elif output_file.suffix == '.json':
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=4)
    else:
        raise ValueError(
            f"Unsupported config file format: {output_file.suffix}. "
            "Use .yaml, .yml, or .json"
        )
    
    print(f"Configuration saved to {output_path}")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Override config values take precedence over base config.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """
    Validate that configuration has required keys.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If required keys are missing
    """
    missing_keys = set(required_keys) - set(config.keys())
    
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")
    
    return True

