"""
Machine learning models registry and factory for the Accurate ML application.
"""

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, 
    IsolationForest, RandomForestClassifier, ExtraTreesClassifier
)
from sklearn.linear_model import (
    LogisticRegression, PassiveAggressiveClassifier, Perceptron, 
    RidgeClassifier, SGDClassifier
)
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator
from typing import Dict, Any, List, Optional
import joblib
import os
from datetime import datetime
from ..utils.logger import logger
from ..config import config

class ModelRegistry:
    """Registry for all available machine learning models."""
    
    def __init__(self):
        self.models = self._initialize_models()
        self.model_categories = self._categorize_models()
    
    def _initialize_models(self) -> Dict[str, BaseEstimator]:
        """Initialize all available models."""
        return {
            # Tree-based models
            "Random Forest": RandomForestClassifier(random_state=config.get('app.random_state', 42)),
            "Decision Tree": DecisionTreeClassifier(random_state=config.get('app.random_state', 42)),
            "Extra Trees": ExtraTreesClassifier(random_state=config.get('app.random_state', 42)),
            
            # Boosting models
            "Gradient Boosting": GradientBoostingClassifier(random_state=config.get('app.random_state', 42)),
            "AdaBoost": AdaBoostClassifier(random_state=config.get('app.random_state', 42)),
            "XGBoost": XGBClassifier(random_state=config.get('app.random_state', 42), eval_metric='logloss'),
            "LightGBM": LGBMClassifier(random_state=config.get('app.random_state', 42), verbose=-1),
            "CatBoost": CatBoostClassifier(random_state=config.get('app.random_state', 42), verbose=False),
            
            # Linear models
            "Logistic Regression": LogisticRegression(random_state=config.get('app.random_state', 42), max_iter=1000),
            "Ridge Classifier": RidgeClassifier(random_state=config.get('app.random_state', 42)),
            "SGD Classifier": SGDClassifier(random_state=config.get('app.random_state', 42)),
            "Passive Aggressive": PassiveAggressiveClassifier(random_state=config.get('app.random_state', 42)),
            "Perceptron": Perceptron(random_state=config.get('app.random_state', 42)),
            
            # Naive Bayes
            "Gaussian Naive Bayes": GaussianNB(),
            "Bernoulli Naive Bayes": BernoulliNB(),
            "Multinomial Naive Bayes": MultinomialNB(),
            
            # Instance-based
            "K-Nearest Neighbors": KNeighborsClassifier(),
            
            # Neural Networks
            "Multi-layer Perceptron": MLPClassifier(random_state=config.get('app.random_state', 42), max_iter=1000),
            
            # Support Vector Machines
            "Support Vector Machine": SVC(random_state=config.get('app.random_state', 42), probability=True),
            
            # Ensemble methods
            "Bagging Classifier": BaggingClassifier(random_state=config.get('app.random_state', 42)),
            
            # Discriminant Analysis
            "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
            "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
            
            # Anomaly Detection
            "Isolation Forest": IsolationForest(random_state=config.get('app.random_state', 42)),
        }
    
    def _categorize_models(self) -> Dict[str, List[str]]:
        """Categorize models by type."""
        return {
            "Tree-based": [
                "Random Forest", "Decision Tree", "Extra Trees"
            ],
            "Boosting": [
                "Gradient Boosting", "AdaBoost", "XGBoost", "LightGBM", "CatBoost"
            ],
            "Linear": [
                "Logistic Regression", "Ridge Classifier", "SGD Classifier", 
                "Passive Aggressive", "Perceptron"
            ],
            "Probabilistic": [
                "Gaussian Naive Bayes", "Bernoulli Naive Bayes", "Multinomial Naive Bayes"
            ],
            "Instance-based": [
                "K-Nearest Neighbors"
            ],
            "Neural Networks": [
                "Multi-layer Perceptron"
            ],
            "Support Vector Machines": [
                "Support Vector Machine"
            ],
            "Ensemble": [
                "Bagging Classifier"
            ],
            "Discriminant Analysis": [
                "Linear Discriminant Analysis", "Quadratic Discriminant Analysis"
            ],
            "Anomaly Detection": [
                "Isolation Forest"
            ]
        }
    
    def get_model(self, model_name: str) -> BaseEstimator:
        """Get a model instance by name."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        
        # Return a fresh copy of the model
        model = self.models[model_name]
        return model.__class__(**model.get_params())
    
    def get_all_models(self) -> List[str]:
        """Get list of all available model names."""
        return list(self.models.keys())
    
    def get_models_by_category(self, category: str) -> List[str]:
        """Get models by category."""
        if category not in self.model_categories:
            raise ValueError(f"Category '{category}' not found. Available categories: {list(self.model_categories.keys())}")
        return self.model_categories[category]
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a model."""
        if model_name not in self.models:
            return {}
        
        model = self.models[model_name]
        
        # Find category
        category = "Other"
        for cat, models in self.model_categories.items():
            if model_name in models:
                category = cat
                break
        
        return {
            "name": model_name,
            "category": category,
            "class": model.__class__.__name__,
            "parameters": model.get_params(),
            "supports_probability": hasattr(model, 'predict_proba'),
            "supports_feature_importance": hasattr(model, 'feature_importances_'),
        }

class ModelManager:
    """Manager for training, saving, and loading models."""
    
    def __init__(self):
        self.registry = ModelRegistry()
        self.trained_models = {}
        self.model_history = []
    
    def train_model(
        self, 
        model_name: str, 
        X_train, 
        y_train, 
        model_params: Optional[Dict[str, Any]] = None
    ) -> BaseEstimator:
        """Train a model with given data."""
        try:
            logger.info(f"Training {model_name} model")
            
            # Get model instance
            model = self.registry.get_model(model_name)
            
            # Update parameters if provided
            if model_params:
                model.set_params(**model_params)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Store trained model
            self.trained_models[model_name] = model
            
            # Record training history
            self.model_history.append({
                'model_name': model_name,
                'timestamp': datetime.now(),
                'parameters': model.get_params(),
                'training_samples': len(X_train)
            })
            
            logger.info(f"Successfully trained {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to train {model_name}: {str(e)}")
            raise
    
    def save_model(self, model_name: str, filepath: Optional[str] = None) -> str:
        """Save a trained model to disk."""
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' has not been trained yet")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name.replace(' ', '_')}_{timestamp}.joblib"
            filepath = os.path.join(config.saved_models_dir, filename)
        
        try:
            joblib.dump(self.trained_models[model_name], filepath)
            logger.info(f"Model '{model_name}' saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save model '{model_name}': {str(e)}")
            raise
    
    def load_model(self, filepath: str, model_name: Optional[str] = None) -> BaseEstimator:
        """Load a model from disk."""
        try:
            model = joblib.load(filepath)
            
            if model_name:
                self.trained_models[model_name] = model
            
            logger.info(f"Model loaded from {filepath}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model from {filepath}: {str(e)}")
            raise
    
    def get_trained_model(self, model_name: str) -> BaseEstimator:
        """Get a trained model."""
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' has not been trained yet")
        return self.trained_models[model_name]
    
    def list_saved_models(self) -> List[str]:
        """List all saved model files."""
        model_dir = config.saved_models_dir
        if not os.path.exists(model_dir):
            return []
        
        return [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
    
    def get_model_recommendations(self, task_type: str, dataset_size: int) -> List[str]:
        """Get model recommendations based on task and dataset characteristics."""
        recommendations = []
        
        if task_type.lower() == 'classification':
            if dataset_size < 1000:
                # Small dataset
                recommendations = [
                    "Random Forest", "Gaussian Naive Bayes", "K-Nearest Neighbors"
                ]
            elif dataset_size < 10000:
                # Medium dataset
                recommendations = [
                    "Random Forest", "XGBoost", "Support Vector Machine", "Logistic Regression"
                ]
            else:
                # Large dataset
                recommendations = [
                    "XGBoost", "LightGBM", "Random Forest", "Multi-layer Perceptron"
                ]
        
        return recommendations

# Global instances
model_registry = ModelRegistry()
model_manager = ModelManager()