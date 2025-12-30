"""
Configuration management for the Accurate ML application.
"""

import os
import toml
from pathlib import Path
from typing import Dict, Any
import streamlit as st

class Config:
    """Configuration manager for the application."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.uploads_dir = self.base_dir / "uploads"
        self.logs_dir = self.base_dir / "logs"
        self.saved_models_dir = self.base_dir / "saved_models"
        self.config_file = self.base_dir / "config.toml"
        
        # Ensure directories exist
        self._create_directories()
        
        # Load configuration
        self.config = self._load_config()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.uploads_dir, self.logs_dir, self.saved_models_dir]:
            directory.mkdir(exist_ok=True)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from TOML file."""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return toml.load(f)
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "app": {
                "title": "Accurate 🎯",
                "version": "2.0.0",
                "debug": False,
                "max_file_size_mb": 200,
                "supported_formats": ["csv", "xlsx", "json"],
                "random_state": 42
            },
            "models": {
                "default_test_size": 0.2,
                "cross_validation_folds": 5,
                "hyperparameter_tuning": True,
                "save_models": True
            },
            "visualization": {
                "figure_size": [12, 8],
                "dpi": 300,
                "style": "whitegrid"
            },
            "preprocessing": {
                "handle_missing": "auto",
                "scaling": "standard",
                "encoding": "auto"
            },
            "theme": {
                "primaryColor": "#1f77b4",
                "backgroundColor": "#f0f2f6",
                "secondaryBackgroundColor": "#e0e0e0",
                "textColor": "#000000",
                "font": "sans serif"
            }
        }
    
    def get(self, key: str, default=None):
        """Get configuration value by key."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def update(self, key: str, value: Any):
        """Update configuration value."""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
        self._save_config()
    
    def _save_config(self):
        """Save configuration to file."""
        with open(self.config_file, 'w') as f:
            toml.dump(self.config, f)

# Global configuration instance
config = Config()