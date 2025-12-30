"""
Data preprocessing utilities for the Accurate ML application.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, 
    OneHotEncoder, OrdinalEncoder
)
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Optional, Tuple, List
import streamlit as st
from ..utils.logger import logger
from ..config import config

class DataPreprocessor:
    """Advanced data preprocessing utilities."""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.preprocessing_steps = []
    
    def preprocess_data(
        self, 
        df: pd.DataFrame, 
        target_column: str,
        test_size: float = 0.2,
        preprocessing_options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Comprehensive data preprocessing pipeline.
        
        Args:
            df: Input dataframe
            target_column: Name of target column
            test_size: Proportion of test set
            preprocessing_options: Optional preprocessing configuration
            
        Returns:
            X_train, X_test, y_train, y_test, preprocessing_info
        """
        try:
            if preprocessing_options is None:
                preprocessing_options = self._get_default_preprocessing_options()
            
            logger.info(f"Starting data preprocessing for target column: {target_column}")
            
            # Separate features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Handle missing values
            X = self._handle_missing_values(X, preprocessing_options.get('missing_strategy', 'auto'))
            
            # Handle categorical variables
            X = self._handle_categorical_variables(X, preprocessing_options.get('encoding_strategy', 'auto'))
            
            # Scale numerical features
            X = self._scale_numerical_features(X, preprocessing_options.get('scaling_strategy', 'standard'))
            
            # Handle target variable
            y, target_info = self._handle_target_variable(y)
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=config.get('app.random_state', 42),
                stratify=y if self._is_classification_task(y) else None
            )
            
            preprocessing_info = {
                'original_shape': df.shape,
                'final_shape': X.shape,
                'features_removed': set(df.columns) - set(X.columns) - {target_column},
                'preprocessing_steps': self.preprocessing_steps,
                'target_info': target_info,
                'test_size': test_size
            }
            
            logger.info(f"Data preprocessing completed. Final shape: {X.shape}")
            
            return X_train, X_test, y_train, y_test, preprocessing_info
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {str(e)}")
            logger.info("Attempting minimal preprocessing fallback...")
            
            # Try minimal preprocessing as fallback
            return self._minimal_preprocessing_fallback(df, target_column, test_size)
    
    def _get_default_preprocessing_options(self) -> Dict[str, Any]:
        """Get default preprocessing options."""
        return {
            'missing_strategy': 'auto',
            'encoding_strategy': 'auto',
            'scaling_strategy': 'standard',
            'remove_outliers': False,
            'feature_selection': False
        }
    
    def _handle_missing_values(self, X: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        if X.isnull().sum().sum() == 0:
            return X
        
        logger.info(f"Handling missing values with strategy: {strategy}")
        
        if strategy == 'auto':
            # Decide strategy based on missing percentage
            missing_percentage = X.isnull().sum().sum() / (X.shape[0] * X.shape[1])
            if missing_percentage < 0.05:
                strategy = 'drop'
            elif missing_percentage < 0.3:
                strategy = 'knn'
            else:
                strategy = 'mean_mode'
        
        if strategy == 'drop':
            X = X.dropna()
        
        elif strategy == 'mean_mode':
            for col in X.columns:
                if X[col].isnull().any():
                    if X[col].dtype in ['int64', 'float64']:
                        X[col].fillna(X[col].mean(), inplace=True)
                    else:
                        X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown', inplace=True)
        
        elif strategy == 'knn':
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            categorical_cols = X.select_dtypes(include=['object']).columns
            
            # Handle numeric columns with KNN
            if len(numeric_cols) > 0:
                imputer = KNNImputer(n_neighbors=5)
                X[numeric_cols] = imputer.fit_transform(X[numeric_cols])
                self.imputers['knn_numeric'] = imputer
            
            # Handle categorical columns with mode
            for col in categorical_cols:
                if X[col].isnull().any():
                    X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown', inplace=True)
        
        self.preprocessing_steps.append(f"Missing values handled with {strategy} strategy")
        return X
    
    def _handle_categorical_variables(self, X: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Handle categorical variables."""
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) == 0:
            return X
        
        logger.info(f"Handling categorical variables with strategy: {strategy}")
        
        if strategy == 'auto':
            # Decide based on number of unique values
            strategy = 'mixed'
        
        for col in categorical_cols:
            unique_count = X[col].nunique()
            
            if strategy == 'label':
                encoder = LabelEncoder()
                X[col] = encoder.fit_transform(X[col].astype(str))
                self.encoders[f'label_{col}'] = encoder
            
            elif strategy == 'onehot':
                # One-hot encoding for low cardinality
                if unique_count <= 10:
                    X = pd.get_dummies(X, columns=[col], prefix=col)
                else:
                    # Fall back to label encoding for high cardinality
                    encoder = LabelEncoder()
                    X[col] = encoder.fit_transform(X[col].astype(str))
                    self.encoders[f'label_{col}'] = encoder
            
            elif strategy == 'mixed':
                if unique_count <= 5:
                    # One-hot for very low cardinality
                    X = pd.get_dummies(X, columns=[col], prefix=col)
                else:
                    # Label encoding for higher cardinality
                    encoder = LabelEncoder()
                    X[col] = encoder.fit_transform(X[col].astype(str))
                    self.encoders[f'label_{col}'] = encoder
        
        self.preprocessing_steps.append(f"Categorical variables handled with {strategy} strategy")
        return X
    
    def _scale_numerical_features(self, X: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Scale numerical features."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return X
        
        logger.info(f"Scaling numerical features with strategy: {strategy}")
        
        if strategy == 'standard':
            scaler = StandardScaler()
        elif strategy == 'minmax':
            scaler = MinMaxScaler()
        elif strategy == 'robust':
            scaler = RobustScaler()
        else:
            return X  # No scaling
        
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        self.scalers[strategy] = scaler
        
        self.preprocessing_steps.append(f"Numerical features scaled with {strategy} scaler")
        return X
    
    def _handle_target_variable(self, y: pd.Series) -> Tuple[pd.Series, Dict[str, Any]]:
        """Handle target variable encoding if needed."""
        target_info = {
            'original_type': str(y.dtype),
            'unique_values': y.nunique(),
            'is_classification': self._is_classification_task(y)
        }
        
        if y.dtype == 'object':
            encoder = LabelEncoder()
            y = pd.Series(encoder.fit_transform(y), index=y.index)
            self.encoders['target'] = encoder
            target_info['encoded'] = True
            target_info['classes'] = list(encoder.classes_)
        else:
            target_info['encoded'] = False
        
        return y, target_info
    
    def _is_classification_task(self, y: pd.Series) -> bool:
        """Determine if the task is classification or regression."""
        if y.dtype == 'object':
            return True
        
        unique_count = y.nunique()
        total_count = len(y)
        
        # If unique values are less than 5% of total or less than 20, likely classification
        return unique_count < 20 or unique_count / total_count < 0.05
    
    def preprocess_test_data(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess new test data using fitted transformers."""
        try:
            logger.info("Preprocessing new test data")
            
            X_test = test_df.copy()
            
            # Apply same preprocessing steps
            for step in self.preprocessing_steps:
                if 'Missing values' in step:
                    # Apply same imputation strategy
                    for name, imputer in self.imputers.items():
                        if 'knn' in name:
                            numeric_cols = X_test.select_dtypes(include=[np.number]).columns
                            X_test[numeric_cols] = imputer.transform(X_test[numeric_cols])
                
                elif 'Categorical variables' in step:
                    # Apply same encoding
                    for name, encoder in self.encoders.items():
                        if name.startswith('label_'):
                            col = name.replace('label_', '')
                            if col in X_test.columns:
                                X_test[col] = encoder.transform(X_test[col].astype(str))
                
                elif 'scaled' in step:
                    # Apply same scaling
                    for name, scaler in self.scalers.items():
                        numeric_cols = X_test.select_dtypes(include=[np.number]).columns
                        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
            
            logger.info("Test data preprocessing completed")
            return X_test
            
        except Exception as e:
            logger.error(f"Test data preprocessing failed: {str(e)}")
            raise
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get summary of preprocessing steps performed."""
        return {
            'steps_performed': self.preprocessing_steps,
            'scalers_fitted': list(self.scalers.keys()),
            'encoders_fitted': list(self.encoders.keys()),
            'imputers_fitted': list(self.imputers.keys())
        }
    
    def _minimal_preprocessing_fallback(
        self, 
        df: pd.DataFrame, 
        target_column: str, 
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Minimal preprocessing fallback when advanced preprocessing fails.
        Only handles essential steps needed for model training.
        """
        try:
            logger.info("Applying minimal preprocessing fallback...")
            
            # Separate features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Only handle critical preprocessing steps
            original_shape = X.shape
            
            # 1. Handle missing values with simple strategy
            if X.isnull().sum().sum() > 0:
                logger.info("Handling missing values with simple mean/mode imputation")
                for col in X.columns:
                    if X[col].isnull().any():
                        if X[col].dtype in ['int64', 'float64']:
                            X[col].fillna(X[col].mean(), inplace=True)
                        else:
                            X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown', inplace=True)
                self.preprocessing_steps.append("Missing values handled with simple mean/mode strategy")
            
            # 2. Handle categorical variables with label encoding only
            categorical_cols = X.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                logger.info("Encoding categorical variables with label encoding")
                for col in categorical_cols:
                    try:
                        encoder = LabelEncoder()
                        X[col] = encoder.fit_transform(X[col].astype(str))
                        self.encoders[f'minimal_label_{col}'] = encoder
                    except Exception as e:
                        logger.warning(f"Could not encode column {col}: {str(e)}")
                        # Drop problematic columns
                        X = X.drop(columns=[col])
                self.preprocessing_steps.append("Categorical variables handled with label encoding")
            
            # 3. Handle target variable if needed
            target_info = {'original_type': str(y.dtype), 'encoded': False}
            if y.dtype == 'object':
                try:
                    encoder = LabelEncoder()
                    y = pd.Series(encoder.fit_transform(y), index=y.index)
                    self.encoders['minimal_target'] = encoder
                    target_info['encoded'] = True
                    target_info['classes'] = list(encoder.classes_)
                except Exception as e:
                    logger.error(f"Could not encode target variable: {str(e)}")
                    raise
            
            # 4. Convert to numpy arrays
            X_np = X.values.astype(np.float32)
            y_np = y.values
            
            # 5. Split the data
            try:
                is_classification = self._is_classification_task(y)
                X_train, X_test, y_train, y_test = train_test_split(
                    X_np, y_np, 
                    test_size=test_size, 
                    random_state=config.get('app.random_state', 42),
                    stratify=y_np if is_classification else None
                )
            except Exception as e:
                logger.warning(f"Stratified split failed: {str(e)}. Using random split.")
                X_train, X_test, y_train, y_test = train_test_split(
                    X_np, y_np, 
                    test_size=test_size, 
                    random_state=config.get('app.random_state', 42),
                    stratify=None
                )
            
            target_info.update({
                'unique_values': len(np.unique(y_np)),
                'is_classification': is_classification
            })
            
            preprocessing_info = {
                'original_shape': df.shape,
                'final_shape': X_np.shape,
                'features_removed': set(df.columns) - set(X.columns) - {target_column},
                'preprocessing_steps': self.preprocessing_steps + ["Applied minimal preprocessing fallback"],
                'target_info': target_info,
                'test_size': test_size,
                'fallback_used': True
            }
            
            logger.info(f"Minimal preprocessing completed successfully. Final shape: {X_np.shape}")
            
            return X_train, X_test, y_train, y_test, preprocessing_info
            
        except Exception as e:
            logger.error(f"Even minimal preprocessing failed: {str(e)}")
            # Return raw data split as last resort
            return self._raw_data_fallback(df, target_column, test_size)
    
    def _raw_data_fallback(
        self, 
        df: pd.DataFrame, 
        target_column: str, 
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Last resort: return raw numerical data only.
        """
        try:
            logger.warning("Applying raw data fallback - only numerical columns will be used")
            
            # Get only numerical columns
            X = df.select_dtypes(include=[np.number]).drop(columns=[target_column], errors='ignore')
            y = df[target_column]
            
            if X.empty:
                raise ValueError("No numerical columns available for training")
            
            # Fill missing values with 0
            X = X.fillna(0)
            
            # Handle target
            if y.dtype == 'object':
                encoder = LabelEncoder()
                y = pd.Series(encoder.fit_transform(y))
                self.encoders['raw_target'] = encoder
            
            # Convert to numpy
            X_np = X.values.astype(np.float32)
            y_np = y.values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_np, y_np, 
                test_size=test_size, 
                random_state=config.get('app.random_state', 42),
                stratify=None
            )
            
            preprocessing_info = {
                'original_shape': df.shape,
                'final_shape': X_np.shape,
                'features_removed': set(df.columns) - set(X.columns) - {target_column},
                'preprocessing_steps': ["Raw data fallback - only numerical features used"],
                'target_info': {
                    'original_type': str(df[target_column].dtype),
                    'encoded': y.dtype != df[target_column].dtype,
                    'is_classification': self._is_classification_task(y)
                },
                'test_size': test_size,
                'fallback_used': True,
                'raw_fallback': True
            }
            
            logger.warning(f"Raw data fallback completed. Using {X.shape[1]} numerical features only.")
            
            return X_train, X_test, y_train, y_test, preprocessing_info
            
        except Exception as e:
            logger.error(f"Raw data fallback also failed: {str(e)}")
            raise ValueError("Unable to preprocess data even with fallback methods. Please check your dataset.")

# Global preprocessor instance
preprocessor = DataPreprocessor()