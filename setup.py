"""
Utility script for initializing the Accurate ML platform.
"""

import os
import sys
from pathlib import Path

def create_directories():
    """Create necessary directories for the application."""
    directories = [
        'logs',
        'saved_models', 
        'uploads',
        'src/__pycache__',
        'tests/__pycache__'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def check_dependencies():
    """Check if all required dependencies are installed."""
    try:
        import streamlit
        import pandas
        import numpy
        import sklearn
        import matplotlib
        import seaborn
        import plotly
        import xgboost
        import lightgbm
        import catboost
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install -r requirements_new.txt")
        return False

def setup_environment():
    """Set up the development environment."""
    print("🚀 Setting up Accurate ML Platform...")
    
    # Create directories
    create_directories()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print("✅ Setup completed successfully!")
    print("\nTo run the application:")
    print("streamlit run app.py")

if __name__ == "__main__":
    setup_environment()