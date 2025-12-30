"""
Model evaluation utilities for the Accurate ML application.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, cohen_kappa_score, matthews_corrcoef,
    log_loss, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from typing import Dict, Any, List, Optional, Union
import streamlit as st
from ..utils.logger import logger
from ..config import config

class ModelEvaluator:
    """Comprehensive model evaluation utilities."""
    
    def __init__(self):
        self.evaluation_results = {}
        
    def evaluate_classification_model(
        self, 
        model, 
        X_test, 
        y_test, 
        model_name: str,
        cross_validate: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation for classification models.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model
            cross_validate: Whether to perform cross-validation
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            logger.info(f"Evaluating classification model: {model_name}")
            
            # Get predictions
            y_pred = model.predict(X_test)
            
            # Get prediction probabilities if available
            y_proba = None
            if hasattr(model, 'predict_proba'):
                try:
                    y_proba = model.predict_proba(X_test)
                except:
                    y_proba = None
            
            # Basic metrics
            results = {
                'model_name': model_name,
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
                'cohen_kappa': float(cohen_kappa_score(y_test, y_pred)),
                'matthews_corrcoef': float(matthews_corrcoef(y_test, y_pred)),
            }
            
            # ROC AUC (for binary and multiclass)
            if y_proba is not None:
                try:
                    if len(np.unique(y_test)) == 2:
                        # Binary classification
                        results['roc_auc'] = float(roc_auc_score(y_test, y_proba[:, 1]))
                    else:
                        # Multiclass classification
                        results['roc_auc'] = float(roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted'))
                except:
                    results['roc_auc'] = None
            else:
                results['roc_auc'] = None
            
            # Log loss
            if y_proba is not None:
                try:
                    results['log_loss'] = float(log_loss(y_test, y_proba))
                except:
                    results['log_loss'] = None
            else:
                results['log_loss'] = None
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            results['confusion_matrix'] = cm.tolist()
            
            # Classification report
            class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            results['classification_report'] = class_report
            
            # Cross-validation scores
            if cross_validate and len(X_test) > 10:  # Only if we have enough samples
                try:
                    cv_folds = min(5, len(X_test) // 2)  # Adjust folds based on data size
                    cv_scores = cross_val_score(
                        model, X_test, y_test, 
                        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                        scoring='accuracy'
                    )
                    results['cv_accuracy_mean'] = float(np.mean(cv_scores))
                    results['cv_accuracy_std'] = float(np.std(cv_scores))
                    results['cv_scores'] = cv_scores.tolist()
                except:
                    results['cv_accuracy_mean'] = None
                    results['cv_accuracy_std'] = None
            
            # Feature importance if available
            if hasattr(model, 'feature_importances_'):
                results['feature_importance'] = model.feature_importances_.tolist()
            
            self.evaluation_results[model_name] = results
            logger.info(f"Evaluation completed for {model_name}")
            
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed for {model_name}: {str(e)}")
            raise
    
    def evaluate_regression_model(
        self, 
        model, 
        X_test, 
        y_test, 
        model_name: str,
        cross_validate: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation for regression models.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model
            cross_validate: Whether to perform cross-validation
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            logger.info(f"Evaluating regression model: {model_name}")
            
            # Get predictions
            y_pred = model.predict(X_test)
            
            # Basic metrics
            results = {
                'model_name': model_name,
                'r2_score': float(r2_score(y_test, y_pred)),
                'mean_squared_error': float(mean_squared_error(y_test, y_pred)),
                'root_mean_squared_error': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                'mean_absolute_error': float(mean_absolute_error(y_test, y_pred)),
            }
            
            # Mean Absolute Percentage Error
            try:
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                results['mean_absolute_percentage_error'] = float(mape)
            except:
                results['mean_absolute_percentage_error'] = None
            
            # Cross-validation scores
            if cross_validate and len(X_test) > 10:
                try:
                    cv_folds = min(5, len(X_test) // 2)
                    cv_scores = cross_val_score(model, X_test, y_test, cv=cv_folds, scoring='r2')
                    results['cv_r2_mean'] = float(np.mean(cv_scores))
                    results['cv_r2_std'] = float(np.std(cv_scores))
                    results['cv_scores'] = cv_scores.tolist()
                except:
                    results['cv_r2_mean'] = None
                    results['cv_r2_std'] = None
            
            # Feature importance if available
            if hasattr(model, 'feature_importances_'):
                results['feature_importance'] = model.feature_importances_.tolist()
            
            self.evaluation_results[model_name] = results
            logger.info(f"Evaluation completed for {model_name}")
            
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed for {model_name}: {str(e)}")
            raise
    
    def compare_models(self, results_list: List[Dict[str, Any]]) -> pd.DataFrame:
        """Compare multiple model evaluation results."""
        if not results_list:
            return pd.DataFrame()
        
        # Determine if classification or regression
        is_classification = 'accuracy' in results_list[0]
        
        if is_classification:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'cohen_kappa']
        else:
            metrics = ['r2_score', 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error']
        
        comparison_data = []
        for result in results_list:
            row = {'Model': result['model_name']}
            for metric in metrics:
                value = result.get(metric)
                if value is not None:
                    row[metric.replace('_', ' ').title()] = value
                else:
                    row[metric.replace('_', ' ').title()] = 'N/A'
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def get_best_model(self, results_list: List[Dict[str, Any]], metric: str = 'auto') -> str:
        """Get the name of the best performing model."""
        if not results_list:
            return None
        
        # Auto-select metric based on task type
        if metric == 'auto':
            if 'accuracy' in results_list[0]:
                metric = 'f1_score'  # Use F1 score for classification
            else:
                metric = 'r2_score'  # Use R² for regression
        
        # Find model with best score
        best_score = None
        best_model = None
        
        higher_is_better = metric not in ['mean_squared_error', 'mean_absolute_error', 'log_loss']
        
        for result in results_list:
            score = result.get(metric)
            if score is not None:
                if best_score is None:
                    best_score = score
                    best_model = result['model_name']
                elif (higher_is_better and score > best_score) or (not higher_is_better and score < best_score):
                    best_score = score
                    best_model = result['model_name']
        
        return best_model
    
    def display_evaluation_results(self, results: Dict[str, Any]):
        """Display evaluation results in Streamlit."""
        st.subheader(f"📊 Evaluation Results: {results['model_name']}")
        
        # Determine if classification or regression
        is_classification = 'accuracy' in results
        
        if is_classification:
            # Classification metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Accuracy", f"{results.get('accuracy', 0):.4f}")
                st.metric("Precision", f"{results.get('precision', 0):.4f}")
            
            with col2:
                st.metric("Recall", f"{results.get('recall', 0):.4f}")
                st.metric("F1 Score", f"{results.get('f1_score', 0):.4f}")
            
            with col3:
                if results.get('roc_auc') is not None:
                    st.metric("ROC AUC", f"{results['roc_auc']:.4f}")
                st.metric("Cohen's Kappa", f"{results.get('cohen_kappa', 0):.4f}")
            
            # Cross-validation results
            if results.get('cv_accuracy_mean') is not None:
                st.subheader("Cross-Validation Results")
                st.metric(
                    "CV Accuracy", 
                    f"{results['cv_accuracy_mean']:.4f} ± {results.get('cv_accuracy_std', 0):.4f}"
                )
        
        else:
            # Regression metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("R² Score", f"{results.get('r2_score', 0):.4f}")
                st.metric("RMSE", f"{results.get('root_mean_squared_error', 0):.4f}")
            
            with col2:
                st.metric("MAE", f"{results.get('mean_absolute_error', 0):.4f}")
                if results.get('mean_absolute_percentage_error') is not None:
                    st.metric("MAPE", f"{results['mean_absolute_percentage_error']:.2f}%")
        
        # Feature importance
        if results.get('feature_importance') is not None:
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': [f'Feature_{i}' for i in range(len(results['feature_importance']))],
                'Importance': results['feature_importance']
            }).sort_values('Importance', ascending=False).head(10)
            
            st.bar_chart(importance_df.set_index('Feature'))

# Global evaluator instance
evaluator = ModelEvaluator()