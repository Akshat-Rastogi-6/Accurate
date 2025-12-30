"""
Data validation utilities for the Accurate ML application.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import streamlit as st
from ..utils.logger import logger

class DataValidator:
    """Data validation and quality checking utilities."""
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive data validation.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary with validation results
        """
        try:
            results = {
                'is_valid': True,
                'issues': [],
                'warnings': [],
                'statistics': {}
            }
            
            # Basic checks
            if df.empty:
                results['is_valid'] = False
                results['issues'].append("Dataset is empty")
                return results
            
            # Shape information
            results['statistics']['shape'] = df.shape
            results['statistics']['memory_usage_mb'] = df.memory_usage(deep=True).sum() / 1024**2
            
            # Missing values analysis
            missing_stats = self._analyze_missing_values(df)
            results['statistics']['missing_values'] = missing_stats
            
            if missing_stats['total_missing_percentage'] > 50:
                results['warnings'].append(
                    f"High missing values: {missing_stats['total_missing_percentage']:.1f}% of data is missing"
                )
            
            # Data types analysis
            dtype_stats = self._analyze_data_types(df)
            results['statistics']['data_types'] = dtype_stats
            
            # Duplicate rows
            duplicate_count = df.duplicated().sum()
            results['statistics']['duplicate_rows'] = duplicate_count
            if duplicate_count > 0:
                results['warnings'].append(f"Found {duplicate_count} duplicate rows")
            
            # Column name validation
            column_issues = self._validate_column_names(df.columns)
            if column_issues:
                results['warnings'].extend(column_issues)
            
            # Outlier detection for numeric columns
            outlier_stats = self._detect_outliers(df)
            results['statistics']['outliers'] = outlier_stats
            
            logger.info(f"Data validation completed. Shape: {df.shape}")
            
            return results
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            return {
                'is_valid': False,
                'issues': [f"Validation failed: {str(e)}"],
                'warnings': [],
                'statistics': {}
            }
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing values in the dataframe."""
        missing_counts = df.isnull().sum()
        total_cells = np.prod(df.shape)
        total_missing = missing_counts.sum()
        
        missing_by_column = {}
        for col in df.columns:
            missing_count = missing_counts[col]
            if missing_count > 0:
                missing_by_column[col] = {
                    'count': int(missing_count),
                    'percentage': float(missing_count / len(df) * 100)
                }
        
        return {
            'total_missing': int(total_missing),
            'total_missing_percentage': float(total_missing / total_cells * 100),
            'columns_with_missing': missing_by_column,
            'completely_missing_columns': list(df.columns[missing_counts == len(df)])
        }
    
    def _analyze_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data types in the dataframe."""
        dtype_counts = df.dtypes.value_counts().to_dict()
        
        # Convert numpy dtypes to strings for JSON serialization
        dtype_counts = {str(k): int(v) for k, v in dtype_counts.items()}
        
        numeric_columns = list(df.select_dtypes(include=[np.number]).columns)
        categorical_columns = list(df.select_dtypes(include=['object']).columns)
        datetime_columns = list(df.select_dtypes(include=['datetime64']).columns)
        
        return {
            'type_distribution': dtype_counts,
            'numeric_columns': numeric_columns,
            'categorical_columns': categorical_columns,
            'datetime_columns': datetime_columns,
            'total_columns': len(df.columns)
        }
    
    def _validate_column_names(self, columns: pd.Index) -> List[str]:
        """Validate column names for potential issues."""
        issues = []
        
        # Check for unnamed columns
        unnamed_cols = [col for col in columns if str(col).startswith('Unnamed:')]
        if unnamed_cols:
            issues.append(f"Found unnamed columns: {unnamed_cols}")
        
        # Check for duplicate column names
        duplicate_cols = columns[columns.duplicated()].tolist()
        if duplicate_cols:
            issues.append(f"Found duplicate column names: {duplicate_cols}")
        
        # Check for columns with special characters that might cause issues
        problematic_cols = []
        for col in columns:
            if not str(col).replace('_', '').replace(' ', '').isalnum():
                if not str(col).startswith('Unnamed:'):
                    problematic_cols.append(col)
        
        if problematic_cols:
            issues.append(f"Columns with special characters: {problematic_cols}")
        
        return issues
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers in numeric columns using IQR method."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_stats = {}
        
        for col in numeric_cols:
            try:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                
                if len(outliers) > 0:
                    outlier_stats[col] = {
                        'count': len(outliers),
                        'percentage': len(outliers) / len(df) * 100,
                        'lower_bound': float(lower_bound),
                        'upper_bound': float(upper_bound)
                    }
            except Exception as e:
                logger.warning(f"Could not detect outliers for column {col}: {str(e)}")
        
        return outlier_stats
    
    def suggest_target_columns(self, df: pd.DataFrame) -> List[str]:
        """Suggest potential target columns based on data characteristics."""
        suggestions = []
        
        for col in df.columns:
            # Check if column looks like a classification target
            if df[col].dtype == 'object':
                unique_values = df[col].nunique()
                if 2 <= unique_values <= 10:  # Good for classification
                    suggestions.append(col)
            elif df[col].dtype in [np.int64, np.float64]:
                unique_values = df[col].nunique()
                if unique_values <= 20:  # Could be classification
                    suggestions.append(col)
        
        return suggestions
    
    def display_validation_report(self, validation_results: Dict[str, Any]):
        """Display validation results in Streamlit."""
        if not validation_results['is_valid']:
            st.error("❌ Data Validation Failed")
            for issue in validation_results['issues']:
                st.error(f"• {issue}")
            return
        
        st.success("✅ Data Validation Passed")
        
        # Display warnings if any
        if validation_results['warnings']:
            st.warning("⚠️ Warnings:")
            for warning in validation_results['warnings']:
                st.warning(f"• {warning}")
        
        # Display statistics
        stats = validation_results['statistics']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Rows", f"{stats['shape'][0]:,}")
            st.metric("Columns", f"{stats['shape'][1]:,}")
        
        with col2:
            st.metric("Memory Usage", f"{stats['memory_usage_mb']:.2f} MB")
            st.metric("Missing Values", f"{stats['missing_values']['total_missing_percentage']:.1f}%")
        
        with col3:
            st.metric("Duplicate Rows", f"{stats.get('duplicate_rows', 0):,}")
            st.metric("Numeric Columns", f"{len(stats['data_types']['numeric_columns'])}")

# Global validator instance
validator = DataValidator()