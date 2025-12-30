"""
Advanced visualization utilities for the Accurate ML application.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import streamlit as st
from typing import Dict, Any, List, Optional, Tuple
import io
import base64
from ..utils.logger import logger
from ..config import config

class DataVisualizer:
    """Advanced data visualization utilities."""
    
    def __init__(self):
        self.figure_size = config.get('visualization.figure_size', [12, 8])
        self.dpi = config.get('visualization.dpi', 300)
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_data_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create comprehensive data overview visualizations."""
        try:
            logger.info("Creating data overview visualizations")
            
            visualizations = {}
            
            # Dataset shape and info
            info_buffer = io.StringIO()
            df.info(buf=info_buffer)
            visualizations['data_info'] = info_buffer.getvalue()
            
            # Missing values heatmap
            if df.isnull().sum().sum() > 0:
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
                plt.title('Missing Values Heatmap')
                plt.tight_layout()
                visualizations['missing_values_heatmap'] = self._fig_to_base64(fig)
                plt.close()
            
            # Data types distribution
            dtype_counts = df.dtypes.value_counts()
            fig, ax = plt.subplots(figsize=(8, 6))
            dtype_counts.plot(kind='bar', ax=ax)
            plt.title('Data Types Distribution')
            plt.xlabel('Data Type')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            visualizations['dtype_distribution'] = self._fig_to_base64(fig)
            plt.close()
            
            # Correlation matrix for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                fig, ax = plt.subplots(figsize=(12, 10))
                correlation_matrix = df[numeric_cols].corr()
                mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
                sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                           center=0, square=True, ax=ax)
                plt.title('Correlation Matrix')
                plt.tight_layout()
                visualizations['correlation_matrix'] = self._fig_to_base64(fig)
                plt.close()
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Failed to create data overview: {str(e)}")
            return {}
    
    def create_target_distribution(self, y: pd.Series, target_name: str) -> str:
        """Create target variable distribution visualization."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if pd.api.types.is_numeric_dtype(y):
                # Numeric target - histogram
                ax.hist(y, bins=30, edgecolor='black', alpha=0.7)
                ax.set_title(f'Distribution of {target_name}')
                ax.set_xlabel(target_name)
                ax.set_ylabel('Frequency')
            else:
                # Categorical target - bar plot
                value_counts = y.value_counts()
                ax.bar(value_counts.index.astype(str), value_counts.values)
                ax.set_title(f'Distribution of {target_name}')
                ax.set_xlabel(target_name)
                ax.set_ylabel('Count')
                
                # Add percentage labels
                total = len(y)
                for i, (label, count) in enumerate(value_counts.items()):
                    percentage = count / total * 100
                    ax.text(i, count + 0.01 * max(value_counts.values), 
                           f'{percentage:.1f}%', ha='center')
            
            plt.tight_layout()
            result = self._fig_to_base64(fig)
            plt.close()
            return result
            
        except Exception as e:
            logger.error(f"Failed to create target distribution: {str(e)}")
            return None
    
    def create_confusion_matrix(self, y_true, y_pred, labels=None, model_name: str = "") -> str:
        """Create an enhanced confusion matrix visualization."""
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=labels, yticklabels=labels, ax=ax)
            
            ax.set_title(f'Confusion Matrix - {model_name}')
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            
            # Add accuracy information
            accuracy = np.diag(cm).sum() / cm.sum()
            ax.text(0.02, 0.98, f'Accuracy: {accuracy:.3f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            result = self._fig_to_base64(fig)
            plt.close()
            return result
            
        except Exception as e:
            logger.error(f"Failed to create confusion matrix: {str(e)}")
            return None
    
    def create_roc_curves(self, y_true, y_proba, model_name: str = "") -> str:
        """Create ROC curves for binary/multiclass classification."""
        try:
            # Check if binary or multiclass
            unique_classes = np.unique(y_true)
            n_classes = len(unique_classes)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            if n_classes == 2:
                # Binary classification
                fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                
            else:
                # Multiclass classification
                colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
                
                for i, color in zip(range(n_classes), colors):
                    # One-vs-rest ROC
                    y_true_binary = (y_true == unique_classes[i]).astype(int)
                    fpr, tpr, _ = roc_curve(y_true_binary, y_proba[:, i])
                    roc_auc = auc(fpr, tpr)
                    
                    ax.plot(fpr, tpr, color=color, lw=2,
                           label=f'Class {unique_classes[i]} (AUC = {roc_auc:.2f})')
                
                ax.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curves - {model_name}')
            ax.legend(loc="lower right")
            
            plt.tight_layout()
            result = self._fig_to_base64(fig)
            plt.close()
            return result
            
        except Exception as e:
            logger.error(f"Failed to create ROC curves: {str(e)}")
            return None
    
    def create_precision_recall_curve(self, y_true, y_proba, model_name: str = "") -> str:
        """Create precision-recall curve."""
        try:
            # For binary classification
            if y_proba.shape[1] == 2:
                precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(recall, precision, color='b', lw=2)
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title(f'Precision-Recall Curve - {model_name}')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                result = self._fig_to_base64(fig)
                plt.close()
                return result
            
        except Exception as e:
            logger.error(f"Failed to create precision-recall curve: {str(e)}")
            return None
    
    def create_feature_importance_plot(self, feature_names: List[str], 
                                     importance_values: List[float], 
                                     model_name: str = "",
                                     top_n: int = 15) -> str:
        """Create feature importance visualization."""
        try:
            # Create DataFrame and sort by importance
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_values
            }).sort_values('importance', ascending=True).tail(top_n)
            
            fig, ax = plt.subplots(figsize=(10, max(6, len(importance_df) * 0.3)))
            
            bars = ax.barh(range(len(importance_df)), importance_df['importance'])
            ax.set_yticks(range(len(importance_df)))
            ax.set_yticklabels(importance_df['feature'])
            ax.set_xlabel('Importance')
            ax.set_title(f'Top {top_n} Feature Importances - {model_name}')
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2, 
                       f'{width:.3f}', ha='left', va='center')
            
            plt.tight_layout()
            result = self._fig_to_base64(fig)
            plt.close()
            return result
            
        except Exception as e:
            logger.error(f"Failed to create feature importance plot: {str(e)}")
            return None
    
    def create_learning_curves(self, train_scores: List[float], 
                             val_scores: List[float], 
                             model_name: str = "") -> str:
        """Create learning curves visualization."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            epochs = range(1, len(train_scores) + 1)
            
            ax.plot(epochs, train_scores, 'o-', label='Training Score', color='blue')
            ax.plot(epochs, val_scores, 'o-', label='Validation Score', color='red')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Score')
            ax.set_title(f'Learning Curves - {model_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            result = self._fig_to_base64(fig)
            plt.close()
            return result
            
        except Exception as e:
            logger.error(f"Failed to create learning curves: {str(e)}")
            return None
    
    def create_residuals_plot(self, y_true, y_pred, model_name: str = "") -> str:
        """Create residuals plot for regression models."""
        try:
            residuals = y_true - y_pred
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Residuals vs Predicted
            ax1.scatter(y_pred, residuals, alpha=0.6)
            ax1.axhline(y=0, color='red', linestyle='--')
            ax1.set_xlabel('Predicted Values')
            ax1.set_ylabel('Residuals')
            ax1.set_title(f'Residuals vs Predicted - {model_name}')
            ax1.grid(True, alpha=0.3)
            
            # Residuals histogram
            ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
            ax2.set_xlabel('Residuals')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Residuals Distribution')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            result = self._fig_to_base64(fig)
            plt.close()
            return result
            
        except Exception as e:
            logger.error(f"Failed to create residuals plot: {str(e)}")
            return None
    
    def create_model_comparison_chart(self, comparison_df: pd.DataFrame) -> str:
        """Create model comparison visualization."""
        try:
            # Determine if classification or regression based on columns
            is_classification = 'Accuracy' in comparison_df.columns
            
            if is_classification:
                metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
                metrics = [m for m in metrics if m in comparison_df.columns]
            else:
                metrics = ['R2 Score', 'Mean Squared Error', 'Mean Absolute Error']
                metrics = [m for m in metrics if m in comparison_df.columns]
            
            fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 6))
            if len(metrics) == 1:
                axes = [axes]
            
            for i, metric in enumerate(metrics):
                # Filter out 'N/A' values and convert to float
                data = comparison_df[['Model', metric]].copy()
                data = data[data[metric] != 'N/A']
                
                if not data.empty:
                    data[metric] = pd.to_numeric(data[metric], errors='coerce')
                    data = data.dropna()
                    
                    if not data.empty:
                        bars = axes[i].bar(data['Model'], data[metric])
                        axes[i].set_title(metric)
                        axes[i].set_xlabel('Model')
                        axes[i].set_ylabel(metric)
                        axes[i].tick_params(axis='x', rotation=45)
                        
                        # Add value labels on bars
                        for bar in bars:
                            height = bar.get_height()
                            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                                       f'{height:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            result = self._fig_to_base64(fig)
            plt.close()
            return result
            
        except Exception as e:
            logger.error(f"Failed to create model comparison chart: {str(e)}")
            return None
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        try:
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
            buffer.seek(0)
            
            # Encode to base64
            img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            buffer.close()
            
            return img_base64
            
        except Exception as e:
            logger.error(f"Failed to convert figure to base64: {str(e)}")
            return None
    
    def display_image_from_base64(self, img_base64: str, caption: str = "", width: int = None):
        """Display base64 encoded image in Streamlit."""
        if img_base64:
            st.image(f"data:image/png;base64,{img_base64}", caption=caption, width=width)
        else:
            st.error("Failed to generate visualization")

# Global visualizer instance
visualizer = DataVisualizer()