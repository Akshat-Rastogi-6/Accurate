"""
Unit tests for the model factory module.
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from src.models.model_factory import ModelRegistry, ModelManager

class TestModelRegistry:
    """Test cases for ModelRegistry."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.registry = ModelRegistry()
    
    def test_get_all_models(self):
        """Test getting all available models."""
        models = self.registry.get_all_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        assert "Random Forest" in models
        assert "XGBoost" in models
    
    def test_get_model_by_name(self):
        """Test getting a specific model."""
        model = self.registry.get_model("Random Forest")
        
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
    
    def test_get_models_by_category(self):
        """Test getting models by category."""
        tree_models = self.registry.get_models_by_category("Tree-based")
        
        assert isinstance(tree_models, list)
        assert "Random Forest" in tree_models
        assert "Decision Tree" in tree_models
    
    def test_invalid_model_name(self):
        """Test error handling for invalid model name."""
        with pytest.raises(ValueError):
            self.registry.get_model("NonexistentModel")
    
    def test_model_info(self):
        """Test getting model information."""
        info = self.registry.get_model_info("Random Forest")
        
        assert "name" in info
        assert "category" in info
        assert "class" in info
        assert info["supports_probability"]

class TestModelManager:
    """Test cases for ModelManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = ModelManager()
        
        # Create sample datasets
        self.X_class, self.y_class = make_classification(
            n_samples=100, n_features=10, n_classes=2, random_state=42
        )
        self.X_reg, self.y_reg = make_regression(
            n_samples=100, n_features=10, random_state=42
        )
    
    def test_train_classification_model(self):
        """Test training a classification model."""
        model = self.manager.train_model("Random Forest", self.X_class, self.y_class)
        
        assert model is not None
        assert hasattr(model, 'predict')
        assert "Random Forest" in self.manager.trained_models
    
    def test_train_regression_model(self):
        """Test training a regression model.""" 
        # Use a model that supports both classification and regression
        model = self.manager.train_model("Random Forest", self.X_reg, self.y_reg)
        
        assert model is not None
        assert hasattr(model, 'predict')
    
    def test_get_trained_model(self):
        """Test retrieving a trained model."""
        # First train a model
        self.manager.train_model("Random Forest", self.X_class, self.y_class)
        
        # Then retrieve it
        model = self.manager.get_trained_model("Random Forest")
        
        assert model is not None
        assert hasattr(model, 'predict')
    
    def test_get_untrained_model_error(self):
        """Test error when getting untrained model."""
        with pytest.raises(ValueError):
            self.manager.get_trained_model("UntrainedModel")
    
    def test_model_recommendations(self):
        """Test getting model recommendations."""
        recommendations = self.manager.get_model_recommendations("classification", 1000)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

if __name__ == "__main__":
    pytest.main([__file__])