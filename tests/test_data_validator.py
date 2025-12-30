"""
Unit tests for the data validator module.
"""

import pytest
import pandas as pd
import numpy as np
from src.utils.data_validator import DataValidator

class TestDataValidator:
    """Test cases for DataValidator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DataValidator()
    
    def test_empty_dataframe_validation(self):
        """Test validation of empty dataframe."""
        df = pd.DataFrame()
        results = self.validator.validate_dataframe(df)
        
        assert not results['is_valid']
        assert "Dataset is empty" in results['issues']
    
    def test_valid_dataframe_validation(self):
        """Test validation of valid dataframe."""
        df = pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5],
            'categorical_col': ['A', 'B', 'A', 'C', 'B'],
            'target': [0, 1, 0, 1, 0]
        })
        
        results = self.validator.validate_dataframe(df)
        
        assert results['is_valid']
        assert results['statistics']['shape'] == (5, 3)
        assert results['statistics']['missing_values']['total_missing'] == 0
    
    def test_missing_values_detection(self):
        """Test missing values detection."""
        df = pd.DataFrame({
            'col1': [1, 2, None, 4, 5],
            'col2': ['A', None, 'C', None, 'E']
        })
        
        results = self.validator.validate_dataframe(df)
        
        assert results['statistics']['missing_values']['total_missing'] == 3
        assert 'col1' in results['statistics']['missing_values']['columns_with_missing']
        assert 'col2' in results['statistics']['missing_values']['columns_with_missing']
    
    def test_target_column_suggestions(self):
        """Test target column suggestions."""
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'binary_target': np.random.choice([0, 1], 100),
            'multi_class': np.random.choice(['A', 'B', 'C'], 100),
            'continuous': np.random.randn(100) * 100
        })
        
        suggestions = self.validator.suggest_target_columns(df)
        
        assert 'binary_target' in suggestions
        assert 'multi_class' in suggestions
        assert 'continuous' not in suggestions

if __name__ == "__main__":
    pytest.main([__file__])