"""
Main Streamlit application for Accurate ML Platform.
A modular, robust, and production-ready machine learning application.
Restructured with tab-based workflow.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import traceback
from pathlib import Path
import sys

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import config
from src.utils.logger import logger
from src.utils.data_validator import validator
from src.preprocessing.data_preprocessor import preprocessor
from src.models.model_factory import model_registry, model_manager
from src.evaluation.model_evaluator import evaluator
from src.visualization.data_visualizer import visualizer

# Page configuration
st.set_page_config(
    page_title=config.get('app.title', 'Accurate'),
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AccurateApp:
    """Main application class for the Accurate ML Platform."""
    
    def __init__(self):
        self.data = None
        self.validation_results = None
        self.preprocessing_info = None
        self.model_results = {}
        self.current_model = None
        
        # Initialize session state
        self._initialize_session_state()
        
        # Load data from session state if available
        if 'data' in st.session_state:
            self.data = st.session_state.data
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        default_states = {
            'data_uploaded': False,
            'data_validated': False,
            'data_preprocessed': False,
            'model_trained': False,
            'target_column': None,
            'selected_models': [],
            'test_size': 0.2,
            'cross_validate': True,
            'save_models': True
        }
        
        for key, default_value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    def run(self):
        """Main application runner."""
        try:
            self._render_header()
            self._render_sidebar()
            self._render_main_content()
            
        except Exception as e:
            logger.exception("Application error occurred")
            st.error(f"An unexpected error occurred: {str(e)}")
            if config.get('app.debug', False):
                st.code(traceback.format_exc())
    
    def _render_header(self):
        """Render application header."""
        st.title(config.get('app.title', 'Accurate'))
        st.markdown("""
        ### Smart Machine Learning Made Simple
        
        Welcome to **Accurate** - your friendly AI assistant for machine learning!
        Just upload your data and we'll help you build and compare different AI models.
        """)
    
    def _render_sidebar(self):
        """Render application sidebar."""
        with st.sidebar:
            st.header("🎛️ Settings")
            
            # App information
            st.info(f"**Version:** {config.get('app.version', '2.0.0')}")
            
            # Help section
            with st.expander("ℹ️ Need Help?"):
                st.markdown("""
                **How to use Accurate:**
                1. 📤 Upload your data file
                2. 📊 Explore your data with visualizations
                3. 🔧 Preprocess and engineer features
                4. 🤖 Select and train ML models
                5. 📈 Evaluate and compare results
                
                **Supported file formats:**
                - CSV files (.csv)
                - Excel files (.xlsx)
                - JSON files (.json)
                """)
            
            # Reset button
            if st.button("🔄 Start New Analysis"):
                self._reset_session()
    
    def _render_main_content(self):
        """Render main application content with tabs."""
        # Check if data is uploaded
        if not st.session_state.data_uploaded:
            self._render_data_upload()
            return
        
        # Create tabs for different stages
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📤 Data Ingestion", 
            "📊 Data Visualization/EDA",
            "🔧 Data Preprocessing",
            "🔍 Preprocessed Data Viz",
            "🤖 Model Training",
            "📈 Model Evaluation"
        ])
        
        with tab1:
            self._render_data_ingestion_tab()
        
        with tab2:
            self._render_data_visualization_tab()
        
        with tab3:
            self._render_data_preprocessing_tab()
        
        with tab4:
            self._render_preprocessed_viz_tab()
        
        with tab5:
            self._render_model_training_tab()
        
        with tab6:
            self._render_model_evaluation_tab()
    
    def _render_data_upload(self):
        """Render data upload interface."""
        st.header("📤 Upload Your Data")
        
        st.markdown("""
        Let's start by uploading your dataset! We support CSV, Excel, and JSON files.
        """)
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose your dataset file",
            type=config.get('app.supported_formats', ['csv', 'xlsx', 'json']),
            help="Upload your data file - we'll analyze it automatically!"
        )
        
        if uploaded_file is not None:
            try:
                # Load data based on file type
                if uploaded_file.name.endswith('.csv'):
                    self.data = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    self.data = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    self.data = pd.read_json(uploaded_file)
                
                # Store data in session state
                st.session_state.data = self.data
                st.session_state.data_filename = uploaded_file.name
                st.session_state.data_uploaded = True
                
                # Display basic info
                st.success(f"🎉 Great! We loaded your dataset: {uploaded_file.name}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Rows", f"{self.data.shape[0]:,}")
                with col2:
                    st.metric("📋 Columns", f"{self.data.shape[1]:,}")
                with col3:
                    st.metric("💾 Size", f"{uploaded_file.size / 1024:.1f} KB")
                
                st.info("✅ Data uploaded! Now explore the tabs above to continue.")
                
                logger.info(f"Dataset loaded successfully: {self.data.shape}")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error loading dataset: {str(e)}")
                logger.error(f"Data loading error: {str(e)}")
    
    def _render_data_ingestion_tab(self):
        """Render data ingestion tab content."""
        st.header("📤 Data Ingestion")
        
        # Get data from session state
        if 'data' in st.session_state:
            self.data = st.session_state.data
            
            st.success(f"✅ Dataset: {st.session_state.get('data_filename', 'Loaded')}")
            
            # Display basic metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Total Rows", f"{self.data.shape[0]:,}")
            with col2:
                st.metric("📋 Total Columns", f"{self.data.shape[1]:,}")
            with col3:
                st.metric("💾 Memory", f"{self.data.memory_usage(deep=True).sum() / 1024:.1f} KB")
            with col4:
                st.metric("🔢 Numeric Cols", len(self.data.select_dtypes(include=[np.number]).columns))
            
            # Show data preview
            st.subheader("👀 Data Preview (First 20 Rows)")
            st.dataframe(self.data.head(20), use_container_width=True)
            
            # Show data info
            st.subheader("📋 Dataset Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Column Information:**")
                dtype_df = pd.DataFrame({
                    'Column': self.data.columns,
                    'Type': self.data.dtypes.astype(str),
                    'Non-Null': self.data.count().values,
                    'Null Count': self.data.isnull().sum().values
                })
                st.dataframe(dtype_df, use_container_width=True)
            
            with col2:
                st.markdown("**Statistical Summary:**")
                st.dataframe(self.data.describe(), use_container_width=True)
            
            # Data validation
            st.subheader("✅ Data Quality Check")
            with st.spinner("Validating data..."):
                self.validation_results = validator.validate_dataframe(self.data)
            validator.display_validation_report(self.validation_results)
            
            # Target column selection
            st.subheader("🎯 Select Target Column")
            st.markdown("Choose the column you want to predict:")
            
            suggested_targets = validator.suggest_target_columns(self.data)
            if suggested_targets:
                st.info(f"💡 **Suggested columns:** {', '.join(suggested_targets)}")
            
            target_column = st.selectbox(
                "Target column (what to predict)",
                options=[''] + self.data.columns.tolist(),
                index=0,
                help="This is what your AI model will learn to predict"
            )
            
            if target_column and target_column != '':
                st.session_state.target_column = target_column
                st.session_state.data_validated = True
                st.success(f"✅ Target column set to: **{target_column}**")
                
                # Show target distribution
                st.subheader("📈 Target Distribution")
                target_dist_plot = visualizer.create_target_distribution(
                    self.data[target_column], target_column
                )
                if target_dist_plot:
                    visualizer.display_image_from_base64(
                        target_dist_plot, 
                        f"Distribution of {target_column}"
                    )
        else:
            st.info("👆 Please upload a dataset first.")
    
    def _render_data_visualization_tab(self):
        """Render data visualization/EDA tab."""
        st.header("📊 Data Visualization & Exploratory Data Analysis")
        
        if 'data' in st.session_state:
            self.data = st.session_state.data
            
            # Overview visualizations
            st.subheader("📈 Data Overview")
            with st.spinner("Creating visualizations..."):
                overview_plots = visualizer.create_data_overview(self.data)
                
                if 'correlation_matrix' in overview_plots:
                    st.subheader("🔗 Correlation Matrix")
                    st.markdown("Shows relationships between numerical columns:")
                    visualizer.display_image_from_base64(
                        overview_plots['correlation_matrix'],
                        "Correlation matrix heatmap"
                    )
                
                col1, col2 = st.columns(2)
                
                if 'dtype_distribution' in overview_plots:
                    with col1:
                        st.subheader("📊 Data Types Distribution")
                        visualizer.display_image_from_base64(
                            overview_plots['dtype_distribution'],
                            "Distribution of column types"
                        )
                
                if 'missing_values_heatmap' in overview_plots:
                    with col2:
                        st.subheader("❓ Missing Values Heatmap")
                        visualizer.display_image_from_base64(
                            overview_plots['missing_values_heatmap'],
                            "Missing values visualization"
                        )
            
            # Column-wise analysis
            st.subheader("🔍 Column-wise Analysis")
            selected_column = st.selectbox(
                "Select a column to analyze",
                options=self.data.columns.tolist()
            )
            
            if selected_column:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Statistics for {selected_column}:**")
                    if self.data[selected_column].dtype in ['int64', 'float64']:
                        st.write(self.data[selected_column].describe())
                    else:
                        st.write(f"**Unique values:** {self.data[selected_column].nunique()}")
                        st.write(f"**Most common:** {self.data[selected_column].mode()[0] if len(self.data[selected_column].mode()) > 0 else 'N/A'}")
                        st.write(f"**Missing:** {self.data[selected_column].isnull().sum()}")
                
                with col2:
                    # Plot distribution
                    fig_dist = visualizer.create_target_distribution(
                        self.data[selected_column], selected_column
                    )
                    if fig_dist:
                        visualizer.display_image_from_base64(fig_dist, f"{selected_column} distribution")
        else:
            st.info("👆 Please upload data in the Data Ingestion tab first.")
    
    def _render_data_preprocessing_tab(self):
        """Render advanced data preprocessing tab."""
        st.header("🔧 Advanced Data Preprocessing & Engineering")
        
        if not st.session_state.get('data_validated', False):
            st.warning("⚠️ Please select a target column in the Data Ingestion tab first!")
            return
        
        # Get data from session state
        if 'data' in st.session_state:
            self.data = st.session_state.data
            
            st.markdown("""
            Configure comprehensive data preprocessing and feature engineering for optimal model performance.
            Choose between **Quick Mode** (automated) or **Custom Mode** (column-by-column control).
            """)
            
            # Preprocessing mode selection
            preprocessing_mode = st.radio(
                "Select Preprocessing Mode:",
                ["🚀 Quick Mode (Auto-Optimize)", "🎛️ Custom Mode (Advanced Control)"],
                help="Quick mode applies intelligent defaults. Custom mode gives you full control over each column."
            )
            
            if preprocessing_mode == "🚀 Quick Mode (Auto-Optimize)":
                self._render_quick_preprocessing_mode()
            else:
                self._render_custom_preprocessing_mode()
            
            # Show current status
            if st.session_state.get('data_preprocessed', False):
                st.success("✅ Data is preprocessed and ready for model training!")
        else:
            st.info("👆 Please upload data first.")
    
    def _render_quick_preprocessing_mode(self):
        """Render quick preprocessing mode with auto-optimization."""
        st.subheader("🚀 Quick Mode - Automated Optimization")
        
        st.info("💡 We'll analyze your data and apply the best preprocessing techniques automatically.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            missing_strategy = st.selectbox(
                "Missing values",
                ["auto", "drop", "mean", "median", "mode", "knn", "interpolate"],
                help="Auto: Smart detection based on data type and distribution"
            )
        
        with col2:
            scaling_strategy = st.selectbox(
                "Feature scaling",
                ["standard", "minmax", "robust", "maxabs", "none"],
                help="Standard: Z-score, MinMax: 0-1, Robust: resistant to outliers"
            )
        
        with col3:
            test_size = st.slider(
                "Test set size",
                0.1, 0.5, st.session_state.get('test_size', 0.2), 0.05
            )
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                handle_outliers = st.checkbox("Remove outliers", value=False, help="Remove extreme values using IQR method")
                apply_pca = st.checkbox("Apply PCA", value=False, help="Reduce dimensions while preserving variance")
                if apply_pca:
                    pca_variance = st.slider("PCA variance to keep", 0.8, 0.99, 0.95, 0.01)
            
            with col2:
                balance_classes = st.checkbox("Balance classes", value=False, help="For classification: handle imbalanced datasets")
                feature_selection = st.checkbox("Auto feature selection", value=False, help="Keep only important features")
                if feature_selection:
                    n_features = st.slider("Max features to keep", 5, 50, 20)
        
        # Run preprocessing
        if st.button("🚀 Auto-Preprocess Data", type="primary", key="quick_preprocess"):
            self._execute_quick_preprocessing(
                missing_strategy, scaling_strategy, test_size,
                handle_outliers, apply_pca, balance_classes, feature_selection,
                pca_variance if apply_pca else 0.95,
                n_features if feature_selection else 20
            )
    
    def _render_custom_preprocessing_mode(self):
        """Render custom preprocessing mode with column-by-column control."""
        st.subheader("🎛️ Custom Mode - Column-by-Column Control")
        
        # Analyze data and provide suggestions
        target_col = st.session_state.target_column
        features = [col for col in self.data.columns if col != target_col]
        
        st.markdown("### 📊 Column Analysis & Recommendations")
        
        # Initialize preprocessing config in session state
        if 'custom_preprocessing_config' not in st.session_state:
            st.session_state.custom_preprocessing_config = {}
        
        # Analyze each column and suggest preprocessing
        column_configs = {}
        
        for col in features:
            with st.expander(f"🔧 {col} - {self.data[col].dtype}", expanded=False):
                col_data = self.data[col]
                col_type = col_data.dtype
                
                # Display column statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Non-null", f"{col_data.count()}/{len(col_data)}")
                with col2:
                    st.metric("Unique", col_data.nunique())
                with col3:
                    missing_pct = (col_data.isnull().sum() / len(col_data) * 100)
                    st.metric("Missing", f"{missing_pct:.1f}%")
                
                # Suggest preprocessing based on data type
                suggestion = self._suggest_preprocessing(col_data, col)
                st.info(f"💡 **Suggestion:** {suggestion}")
                
                # Configuration for this column
                config = {}
                
                # Handle missing values
                if col_data.isnull().sum() > 0:
                    st.markdown("**Missing Values:**")
                    if col_type in ['int64', 'float64']:
                        config['missing'] = st.selectbox(
                            "Handle missing",
                            ["mean", "median", "mode", "forward_fill", "backward_fill", "interpolate", "constant", "drop"],
                            key=f"missing_{col}"
                        )
                        if config['missing'] == 'constant':
                            config['fill_value'] = st.number_input(f"Fill value for {col}", value=0.0, key=f"fill_{col}")
                    else:
                        config['missing'] = st.selectbox(
                            "Handle missing",
                            ["mode", "constant", "drop"],
                            key=f"missing_{col}"
                        )
                        if config['missing'] == 'constant':
                            config['fill_value'] = st.text_input(f"Fill value for {col}", value="Unknown", key=f"fill_{col}")
                
                # Type-specific preprocessing
                if col_type in ['int64', 'float64']:
                    st.markdown("**Numerical Preprocessing:**")
                    
                    # Check for potential binning (age, price ranges, etc.)
                    if self._is_binnable_column(col, col_data):
                        apply_binning = st.checkbox(f"Apply binning/grouping", value=False, key=f"bin_{col}")
                        if apply_binning:
                            config['binning'] = st.selectbox(
                                "Binning method",
                                ["equal_width", "equal_frequency", "custom"],
                                key=f"bin_method_{col}"
                            )
                            config['n_bins'] = st.slider("Number of bins", 3, 10, 5, key=f"n_bins_{col}")
                            
                            if config['binning'] == 'custom':
                                bins_str = st.text_input(
                                    "Custom bin edges (comma-separated)",
                                    value="0,25,50,75,100",
                                    key=f"custom_bins_{col}"
                                )
                                config['custom_bins'] = [float(x.strip()) for x in bins_str.split(',')]
                    
                    # Scaling
                    config['scaling'] = st.selectbox(
                        "Scaling method",
                        ["none", "standard", "minmax", "robust", "maxabs", "log", "sqrt", "power"],
                        key=f"scale_{col}",
                        help="Standard: Z-score, MinMax: 0-1, Robust: IQR-based, Log: log(x+1)"
                    )
                    
                    if config.get('scaling') == 'power':
                        config['power_lambda'] = st.slider("Power lambda", -2.0, 2.0, 1.0, 0.1, key=f"power_{col}")
                    
                    # Outlier handling
                    config['outliers'] = st.selectbox(
                        "Outlier handling",
                        ["none", "clip", "remove", "winsorize"],
                        key=f"outlier_{col}",
                        help="Clip: Cap at percentiles, Remove: Drop rows, Winsorize: Replace extremes"
                    )
                    
                    if config['outliers'] != 'none':
                        config['outlier_threshold'] = st.slider(
                            "Outlier threshold (IQR multiplier)",
                            1.0, 3.0, 1.5, 0.1,
                            key=f"outlier_thresh_{col}"
                        )
                
                elif col_type == 'object' or col_type.name == 'category':
                    st.markdown("**Categorical Preprocessing:**")
                    
                    unique_count = col_data.nunique()
                    
                    # Encoding method
                    if unique_count == 2:
                        config['encoding'] = st.selectbox(
                            "Encoding method",
                            ["binary", "label", "onehot"],
                            index=0,
                            key=f"encode_{col}",
                            help="Binary encoding recommended for binary categories"
                        )
                    elif unique_count > 10:
                        config['encoding'] = st.selectbox(
                            "Encoding method",
                            ["label", "frequency", "target", "hash", "embeddings"],
                            index=0,
                            key=f"encode_{col}",
                            help="Label/Frequency recommended for high cardinality"
                        )
                    else:
                        config['encoding'] = st.selectbox(
                            "Encoding method",
                            ["onehot", "label", "ordinal", "binary", "target"],
                            index=0,
                            key=f"encode_{col}",
                            help="OneHot recommended for low cardinality"
                        )
                    
                    # Handle rare categories
                    if unique_count > 5:
                        config['rare_categories'] = st.selectbox(
                            "Rare categories",
                            ["keep", "group_as_other", "drop"],
                            key=f"rare_{col}",
                            help="Group infrequent categories"
                        )
                        if config['rare_categories'] == 'group_as_other':
                            config['rare_threshold'] = st.slider(
                                "Frequency threshold (%)",
                                0.5, 10.0, 5.0, 0.5,
                                key=f"rare_thresh_{col}"
                            )
                    
                    # Text preprocessing for string columns
                    if col_type == 'object':
                        apply_text_processing = st.checkbox("Apply text preprocessing", value=False, key=f"text_{col}")
                        if apply_text_processing:
                            config['text_processing'] = st.multiselect(
                                "Text operations",
                                ["lowercase", "remove_special_chars", "remove_numbers", "strip_whitespace"],
                                default=["lowercase", "strip_whitespace"],
                                key=f"text_ops_{col}"
                            )
                
                # Feature engineering options
                st.markdown("**Feature Engineering:**")
                config['feature_engineering'] = st.multiselect(
                    "Create new features",
                    ["polynomial", "interaction", "log", "sqrt", "reciprocal"],
                    key=f"feat_eng_{col}",
                    help="Generate additional features from this column"
                )
                
                column_configs[col] = config
        
        # Global settings
        st.markdown("### 🌐 Global Settings")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05, key="custom_test_size")
        with col2:
            remove_multicollinearity = st.checkbox("Remove multicollinear features", value=False)
            if remove_multicollinearity:
                corr_threshold = st.slider("Correlation threshold", 0.7, 0.99, 0.95, 0.01)
        with col3:
            normalize_data = st.checkbox("Normalize entire dataset", value=False, help="Apply normalization after all transformations")
        
        # Run custom preprocessing
        if st.button("🔧 Apply Custom Preprocessing", type="primary", key="custom_preprocess"):
            self._execute_custom_preprocessing(
                column_configs, test_size,
                remove_multicollinearity if remove_multicollinearity else False,
                corr_threshold if remove_multicollinearity else 0.95,
                normalize_data
            )
    
    def _suggest_preprocessing(self, col_data, col_name):
        """Suggest preprocessing based on column characteristics."""
        col_type = col_data.dtype
        unique_count = col_data.nunique()
        missing_pct = col_data.isnull().sum() / len(col_data) * 100
        
        suggestions = []
        
        # Missing value suggestion
        if missing_pct > 0:
            if missing_pct > 50:
                suggestions.append("High missing data - consider dropping")
            elif col_type in ['int64', 'float64']:
                if col_data.skew() > 1:
                    suggestions.append("Use median for missing (skewed data)")
                else:
                    suggestions.append("Use mean for missing (normal distribution)")
            else:
                suggestions.append("Use mode for missing categorical values")
        
        # Type-specific suggestions
        if col_type in ['int64', 'float64']:
            # Check if it looks like age, year, etc.
            if any(keyword in col_name.lower() for keyword in ['age', 'year', 'price', 'salary', 'income']):
                suggestions.append("Consider binning into ranges")
            
            # Check distribution
            if abs(col_data.skew()) > 1:
                suggestions.append("Apply log/sqrt transformation (skewed)")
            
            # Check for outliers
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((col_data < (Q1 - 1.5 * IQR)) | (col_data > (Q3 + 1.5 * IQR))).sum()
            if outliers > 0:
                suggestions.append(f"Has {outliers} outliers - consider clipping")
            
            suggestions.append("Apply standard/robust scaling")
        
        elif col_type == 'object':
            if unique_count == 2:
                suggestions.append("Binary encoding (2 unique values)")
            elif unique_count > 10:
                suggestions.append(f"High cardinality ({unique_count}) - use label/frequency encoding")
            else:
                suggestions.append("OneHot encoding (low cardinality)")
            
            # Check for rare categories
            value_counts = col_data.value_counts(normalize=True)
            rare_categories = (value_counts < 0.05).sum()
            if rare_categories > 0:
                suggestions.append(f"Group {rare_categories} rare categories")
        
        return " | ".join(suggestions) if suggestions else "No specific recommendations"
    
    def _is_binnable_column(self, col_name, col_data):
        """Check if column is suitable for binning."""
        keywords = ['age', 'year', 'price', 'salary', 'income', 'score', 'rating', 'amount', 'distance', 'duration']
        return any(keyword in col_name.lower() for keyword in keywords) or (col_data.nunique() > 20 and col_data.dtype in ['int64', 'float64'])
    
    def _execute_quick_preprocessing(self, missing_strategy, scaling_strategy, test_size,
                                     handle_outliers, apply_pca, balance_classes, feature_selection,
                                     pca_variance, n_features):
        """Execute quick preprocessing with auto-optimization."""
        try:
            with st.spinner("🚀 Auto-preprocessing your data..."):
                preprocessing_options = {
                    'missing_strategy': missing_strategy,
                    'scaling_strategy': scaling_strategy,
                    'handle_outliers': handle_outliers,
                    'apply_pca': apply_pca,
                    'pca_variance': pca_variance,
                    'balance_classes': balance_classes,
                    'feature_selection': feature_selection,
                    'n_features': n_features
                }
                
                X_train, X_test, y_train, y_test, preprocessing_info = preprocessor.preprocess_data(
                    self.data,
                    st.session_state.target_column,
                    test_size=test_size,
                    preprocessing_options=preprocessing_options
                )
                
                # Store in session state
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.preprocessing_info = preprocessing_info
                st.session_state.data_preprocessed = True
                
                st.success("✅ Preprocessing completed successfully!")
                
                # Display summary
                self._display_preprocessing_summary(preprocessing_info, test_size)
                
        except Exception as e:
            st.error(f"❌ Preprocessing failed: {str(e)}")
            logger.error(f"Quick preprocessing error: {str(e)}")
            st.info("💡 Try Custom Mode for more control over preprocessing steps.")
    
    def _execute_custom_preprocessing(self, column_configs, test_size,
                                      remove_multicollinearity, corr_threshold, normalize_data):
        """Execute custom preprocessing with column-specific configurations."""
        try:
            with st.spinner("🔧 Applying custom preprocessing..."):
                # Create a copy of the data
                processed_data = self.data.copy()
                target_col = st.session_state.target_column
                
                preprocessing_steps = []
                
                # Separate target
                y = processed_data[target_col]
                X = processed_data.drop(columns=[target_col])
                
                # Apply column-specific preprocessing
                for col, config in column_configs.items():
                    if col not in X.columns:
                        continue
                    
                    # Handle missing values
                    if 'missing' in config and X[col].isnull().sum() > 0:
                        if config['missing'] == 'mean':
                            X[col].fillna(X[col].mean(), inplace=True)
                            preprocessing_steps.append(f"{col}: Filled missing with mean")
                        elif config['missing'] == 'median':
                            X[col].fillna(X[col].median(), inplace=True)
                            preprocessing_steps.append(f"{col}: Filled missing with median")
                        elif config['missing'] == 'mode':
                            X[col].fillna(X[col].mode()[0], inplace=True)
                            preprocessing_steps.append(f"{col}: Filled missing with mode")
                        elif config['missing'] == 'constant':
                            X[col].fillna(config.get('fill_value', 0), inplace=True)
                            preprocessing_steps.append(f"{col}: Filled missing with constant")
                        elif config['missing'] == 'interpolate':
                            X[col].interpolate(inplace=True)
                            preprocessing_steps.append(f"{col}: Interpolated missing values")
                        elif config['missing'] == 'forward_fill':
                            X[col].fillna(method='ffill', inplace=True)
                            preprocessing_steps.append(f"{col}: Forward filled missing")
                        elif config['missing'] == 'backward_fill':
                            X[col].fillna(method='bfill', inplace=True)
                            preprocessing_steps.append(f"{col}: Backward filled missing")
                    
                    # Numerical preprocessing
                    if X[col].dtype in ['int64', 'float64']:
                        # Outlier handling
                        if config.get('outliers', 'none') != 'none':
                            Q1 = X[col].quantile(0.25)
                            Q3 = X[col].quantile(0.75)
                            IQR = Q3 - Q1
                            threshold = config.get('outlier_threshold', 1.5)
                            lower_bound = Q1 - threshold * IQR
                            upper_bound = Q3 + threshold * IQR
                            
                            if config['outliers'] == 'clip':
                                X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)
                                preprocessing_steps.append(f"{col}: Clipped outliers")
                            elif config['outliers'] == 'remove':
                                mask = (X[col] >= lower_bound) & (X[col] <= upper_bound)
                                X = X[mask]
                                y = y[mask]
                                preprocessing_steps.append(f"{col}: Removed outlier rows")
                        
                        # Binning
                        if 'binning' in config:
                            if config['binning'] == 'equal_width':
                                X[f"{col}_binned"] = pd.cut(X[col], bins=config['n_bins'], labels=False)
                                preprocessing_steps.append(f"{col}: Created equal-width bins")
                            elif config['binning'] == 'equal_frequency':
                                X[f"{col}_binned"] = pd.qcut(X[col], q=config['n_bins'], labels=False, duplicates='drop')
                                preprocessing_steps.append(f"{col}: Created equal-frequency bins")
                            elif config['binning'] == 'custom':
                                X[f"{col}_binned"] = pd.cut(X[col], bins=config['custom_bins'], labels=False)
                                preprocessing_steps.append(f"{col}: Created custom bins")
                        
                        # Scaling/Transformation
                        if config.get('scaling', 'none') != 'none':
                            from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
                            
                            if config['scaling'] == 'log':
                                X[col] = np.log1p(X[col])
                                preprocessing_steps.append(f"{col}: Applied log transformation")
                            elif config['scaling'] == 'sqrt':
                                X[col] = np.sqrt(X[col])
                                preprocessing_steps.append(f"{col}: Applied sqrt transformation")
                            elif config['scaling'] == 'power':
                                from sklearn.preprocessing import PowerTransformer
                                pt = PowerTransformer(method='yeo-johnson')
                                X[col] = pt.fit_transform(X[[col]])
                                preprocessing_steps.append(f"{col}: Applied power transformation")
                    
                    # Categorical preprocessing
                    elif X[col].dtype == 'object':
                        # Text preprocessing
                        if config.get('text_processing'):
                            for op in config['text_processing']:
                                if op == 'lowercase':
                                    X[col] = X[col].str.lower()
                                elif op == 'remove_special_chars':
                                    X[col] = X[col].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
                                elif op == 'remove_numbers':
                                    X[col] = X[col].str.replace(r'\d+', '', regex=True)
                                elif op == 'strip_whitespace':
                                    X[col] = X[col].str.strip()
                            preprocessing_steps.append(f"{col}: Applied text preprocessing")
                        
                        # Handle rare categories
                        if config.get('rare_categories') == 'group_as_other':
                            threshold = config.get('rare_threshold', 5) / 100
                            value_counts = X[col].value_counts(normalize=True)
                            rare_cats = value_counts[value_counts < threshold].index
                            X[col] = X[col].replace(rare_cats, 'Other')
                            preprocessing_steps.append(f"{col}: Grouped rare categories")
                        
                        # Encoding
                        encoding = config.get('encoding', 'label')
                        if encoding == 'label':
                            from sklearn.preprocessing import LabelEncoder
                            le = LabelEncoder()
                            X[col] = le.fit_transform(X[col].astype(str))
                            preprocessing_steps.append(f"{col}: Label encoded")
                        elif encoding == 'onehot':
                            dummies = pd.get_dummies(X[col], prefix=col)
                            X = pd.concat([X, dummies], axis=1)
                            X.drop(columns=[col], inplace=True)
                            preprocessing_steps.append(f"{col}: OneHot encoded")
                        elif encoding == 'frequency':
                            freq = X[col].value_counts(normalize=True)
                            X[col] = X[col].map(freq)
                            preprocessing_steps.append(f"{col}: Frequency encoded")
                
                # Convert to numpy arrays
                X_array = X.select_dtypes(include=[np.number]).fillna(0).values
                y_array = y.values
                
                # Handle target encoding if needed
                if y.dtype == 'object':
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    y_array = le.fit_transform(y_array)
                
                # Train-test split
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_array, y_array, test_size=test_size, random_state=42
                )
                
                # Apply scaling to entire dataset if requested
                if normalize_data:
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                    preprocessing_steps.append("Applied standard scaling to entire dataset")
                
                # Store results
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.preprocessing_info = {
                    'original_shape': self.data.shape,
                    'final_shape': X_array.shape,
                    'preprocessing_steps': preprocessing_steps,
                    'target_info': {'is_classification': len(np.unique(y_array)) < 20},
                    'test_size': test_size
                }
                st.session_state.data_preprocessed = True
                
                st.success("✅ Custom preprocessing completed!")
                self._display_preprocessing_summary(st.session_state.preprocessing_info, test_size)
                
        except Exception as e:
            st.error(f"❌ Custom preprocessing failed: {str(e)}")
            logger.error(f"Custom preprocessing error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    def _display_preprocessing_summary(self, preprocessing_info, test_size):
        """Display preprocessing summary."""
        st.subheader("📋 Preprocessing Summary")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Shape", f"{preprocessing_info['original_shape'][0]}×{preprocessing_info['original_shape'][1]}")
        with col2:
            st.metric("Final Shape", f"{preprocessing_info['final_shape'][0]}×{preprocessing_info['final_shape'][1]}")
        with col3:
            st.metric("Test Size", f"{test_size:.1%}")
        
        # Show steps
        st.subheader("🔄 Applied Transformations")
        for i, step in enumerate(preprocessing_info['preprocessing_steps'], 1):
            st.write(f"{i}. {step}")
        
        # Task type
        is_classification = preprocessing_info['target_info']['is_classification']
        task_type = "Classification" if is_classification else "Regression"
        st.info(f"🎯 **Task Type:** {task_type}")
    
    def _render_preprocessed_viz_tab(self):
        """Render preprocessed data visualization tab."""
        st.header("🔍 Preprocessed Data Visualization")
        
        if not st.session_state.get('data_preprocessed', False):
            st.warning("⚠️ Please preprocess data in the Preprocessing tab first!")
            return
        
        # Get preprocessed data
        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test
        preprocessing_info = st.session_state.preprocessing_info
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Training Samples", f"{len(X_train):,}")
        with col2:
            st.metric("Test Samples", f"{len(X_test):,}")
        with col3:
            st.metric("Features", f"{X_train.shape[1]}")
        with col4:
            task = "Classification" if preprocessing_info['target_info']['is_classification'] else "Regression"
            st.metric("Task Type", task)
        
        # Target distribution after preprocessing
        st.subheader("📊 Target Distribution (After Preprocessing)")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Training Set:**")
            y_train_series = pd.Series(y_train, name=st.session_state.target_column)
            train_dist = visualizer.create_target_distribution(y_train_series, "Training Target")
            if train_dist:
                visualizer.display_image_from_base64(train_dist, "Training target distribution")
        
        with col2:
            st.markdown("**Test Set:**")
            y_test_series = pd.Series(y_test, name=st.session_state.target_column)
            test_dist = visualizer.create_target_distribution(y_test_series, "Test Target")
            if test_dist:
                visualizer.display_image_from_base64(test_dist, "Test target distribution")
        
        # Feature statistics
        st.subheader("📈 Feature Statistics")
        X_train_df = pd.DataFrame(X_train, columns=[f"Feature_{i}" for i in range(X_train.shape[1])])
        st.dataframe(X_train_df.describe(), use_container_width=True)
        
        # Sample of preprocessed data
        st.subheader("👀 Preprocessed Data Sample")
        sample_size = min(20, len(X_train))
        st.dataframe(X_train_df.head(sample_size), use_container_width=True)
    
    def _render_model_training_tab(self):
        """Render model training tab with model selection and hyperparameter tuning."""
        st.header("🤖 Model Training & Hyperparameter Tuning")
        
        if not st.session_state.get('data_preprocessed', False):
            st.warning("⚠️ Please preprocess data first!")
            return
        
        # Dataset info
        st.subheader("📊 Dataset Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Samples", f"{len(st.session_state.X_train):,}")
        with col2:
            st.metric("Test Samples", f"{len(st.session_state.X_test):,}")
        with col3:
            task = "Classification" if st.session_state.preprocessing_info['target_info']['is_classification'] else "Regression"
            st.metric("Task Type", task)
        
        st.divider()
        
        # Model Selection
        st.subheader("🤖 Select Models to Train")
        
        # Model category filter
        categories = ["All"] + list(model_registry.model_categories.keys())
        selected_category = st.selectbox(
            "Filter by Model Category",
            categories,
            help="Choose a category to filter models or select 'All' to see everything"
        )
        
        # Get available models
        if selected_category == "All":
            available_models = model_registry.get_all_models()
        else:
            available_models = model_registry.get_models_by_category(selected_category)
        
        # Model selection with descriptions
        st.markdown("**Available Models:**")
        selected_models = st.multiselect(
            "Choose one or more models to train and compare",
            available_models,
            default=[],
            help="Select multiple models to compare their performance"
        )
        
        if not selected_models:
            st.info("👆 Please select at least one model to proceed with training.")
            return
        
        st.success(f"✅ Selected {len(selected_models)} model(s): {', '.join(selected_models)}")
        
        st.divider()
        
        # Training Configuration
        st.subheader("⚙️ Training Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**General Settings:**")
            cross_validate = st.checkbox(
                "Enable Cross-Validation",
                value=True,
                help="Use k-fold cross-validation for more reliable performance estimates"
            )
            
            if cross_validate:
                cv_folds = st.slider(
                    "Number of CV folds",
                    3, 10, 5,
                    help="More folds = more reliable but slower"
                )
            else:
                cv_folds = 5
            
            save_models = st.checkbox(
                "Save trained models",
                value=True,
                help="Save models to disk for later use"
            )
            
            random_state = st.number_input(
                "Random seed",
                value=42,
                help="Set seed for reproducible results"
            )
        
        with col2:
            st.markdown("**Hyperparameter Tuning:**")
            enable_tuning = st.checkbox(
                "Enable hyperparameter optimization",
                value=False,
                help="Automatically search for best hyperparameters (takes longer)"
            )
            
            if enable_tuning:
                tuning_method = st.radio(
                    "Optimization method",
                    ["Grid Search", "Random Search", "Bayesian Optimization"],
                    horizontal=True,
                    help="Grid: exhaustive, Random: faster, Bayesian: most efficient"
                )
                
                n_iterations = st.slider(
                    "Number of iterations",
                    10, 100, 20,
                    help="More iterations = better results but slower"
                )
                
                st.warning("⚠️ Hyperparameter tuning will take significantly longer!")
        
        # Model-specific parameters
        st.divider()
        st.subheader("🎛️ Model-Specific Parameters")
        
        model_params = {}
        
        for model_name in selected_models:
            with st.expander(f"⚙️ {model_name} Parameters", expanded=False):
                model_params[model_name] = self._get_model_parameters(model_name)
        
        # Store settings in session state
        st.session_state.cross_validate = cross_validate
        st.session_state.cv_folds = cv_folds
        st.session_state.save_models = save_models
        st.session_state.random_state = random_state
        st.session_state.model_params = model_params
        
        st.divider()
        
        # Start training
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("🚀 Start Training", type="primary", use_container_width=True):
                self._train_models(selected_models, enable_tuning, model_params)
        
        # Show training status
        if st.session_state.get('model_trained', False):
            st.success("✅ Models trained successfully! Check the Model Evaluation tab.")
    
    def _get_model_parameters(self, model_name):
        """Get model-specific parameters interface."""
        params = {}
        
        # Common parameters for different model types
        if "Random Forest" in model_name or "Extra Trees" in model_name:
            col1, col2 = st.columns(2)
            with col1:
                params['n_estimators'] = st.slider(
                    "Number of trees",
                    10, 500, 100, 10,
                    key=f"{model_name}_n_estimators"
                )
                params['max_depth'] = st.slider(
                    "Max depth",
                    1, 50, 10,
                    key=f"{model_name}_max_depth"
                )
            with col2:
                params['min_samples_split'] = st.slider(
                    "Min samples split",
                    2, 20, 2,
                    key=f"{model_name}_min_samples_split"
                )
                params['min_samples_leaf'] = st.slider(
                    "Min samples leaf",
                    1, 20, 1,
                    key=f"{model_name}_min_samples_leaf"
                )
        
        elif "XGBoost" in model_name:
            col1, col2 = st.columns(2)
            with col1:
                params['n_estimators'] = st.slider(
                    "Number of estimators",
                    10, 500, 100, 10,
                    key=f"{model_name}_n_estimators"
                )
                params['max_depth'] = st.slider(
                    "Max depth",
                    1, 15, 6,
                    key=f"{model_name}_max_depth"
                )
                params['learning_rate'] = st.slider(
                    "Learning rate",
                    0.01, 1.0, 0.1, 0.01,
                    key=f"{model_name}_learning_rate"
                )
            with col2:
                params['subsample'] = st.slider(
                    "Subsample ratio",
                    0.5, 1.0, 0.8, 0.1,
                    key=f"{model_name}_subsample"
                )
                params['colsample_bytree'] = st.slider(
                    "Column sample ratio",
                    0.5, 1.0, 0.8, 0.1,
                    key=f"{model_name}_colsample"
                )
        
        elif "LightGBM" in model_name or "LGBM" in model_name:
            col1, col2 = st.columns(2)
            with col1:
                params['n_estimators'] = st.slider(
                    "Number of estimators",
                    10, 500, 100, 10,
                    key=f"{model_name}_n_estimators"
                )
                params['num_leaves'] = st.slider(
                    "Number of leaves",
                    10, 200, 31,
                    key=f"{model_name}_num_leaves"
                )
                params['learning_rate'] = st.slider(
                    "Learning rate",
                    0.01, 1.0, 0.1, 0.01,
                    key=f"{model_name}_learning_rate"
                )
            with col2:
                params['max_depth'] = st.slider(
                    "Max depth",
                    -1, 50, -1,
                    key=f"{model_name}_max_depth",
                    help="-1 means no limit"
                )
                params['min_child_samples'] = st.slider(
                    "Min child samples",
                    5, 100, 20,
                    key=f"{model_name}_min_child_samples"
                )
        
        elif "CatBoost" in model_name:
            col1, col2 = st.columns(2)
            with col1:
                params['iterations'] = st.slider(
                    "Number of iterations",
                    10, 500, 100, 10,
                    key=f"{model_name}_iterations"
                )
                params['depth'] = st.slider(
                    "Depth",
                    1, 16, 6,
                    key=f"{model_name}_depth"
                )
                params['learning_rate'] = st.slider(
                    "Learning rate",
                    0.01, 1.0, 0.1, 0.01,
                    key=f"{model_name}_learning_rate"
                )
            with col2:
                params['l2_leaf_reg'] = st.slider(
                    "L2 regularization",
                    1, 10, 3,
                    key=f"{model_name}_l2_reg"
                )
        
        elif "SVM" in model_name or "Support Vector" in model_name:
            col1, col2 = st.columns(2)
            with col1:
                params['C'] = st.slider(
                    "Regularization (C)",
                    0.01, 10.0, 1.0, 0.1,
                    key=f"{model_name}_C"
                )
                params['kernel'] = st.selectbox(
                    "Kernel",
                    ['rbf', 'linear', 'poly', 'sigmoid'],
                    key=f"{model_name}_kernel"
                )
            with col2:
                if params['kernel'] == 'rbf':
                    params['gamma'] = st.selectbox(
                        "Gamma",
                        ['scale', 'auto'],
                        key=f"{model_name}_gamma"
                    )
        
        elif "Logistic Regression" in model_name or "Linear Regression" in model_name:
            col1, col2 = st.columns(2)
            with col1:
                if "Logistic" in model_name:
                    params['C'] = st.slider(
                        "Regularization (C)",
                        0.01, 10.0, 1.0, 0.1,
                        key=f"{model_name}_C"
                    )
                    params['penalty'] = st.selectbox(
                        "Penalty",
                        ['l2', 'l1', 'elasticnet', 'none'],
                        key=f"{model_name}_penalty"
                    )
            with col2:
                params['max_iter'] = st.slider(
                    "Max iterations",
                    100, 5000, 1000, 100,
                    key=f"{model_name}_max_iter"
                )
        
        elif "KNN" in model_name or "K-Nearest" in model_name:
            col1, col2 = st.columns(2)
            with col1:
                params['n_neighbors'] = st.slider(
                    "Number of neighbors",
                    1, 50, 5,
                    key=f"{model_name}_n_neighbors"
                )
            with col2:
                params['weights'] = st.selectbox(
                    "Weights",
                    ['uniform', 'distance'],
                    key=f"{model_name}_weights"
                )
                params['metric'] = st.selectbox(
                    "Distance metric",
                    ['euclidean', 'manhattan', 'minkowski'],
                    key=f"{model_name}_metric"
                )
        
        elif "Decision Tree" in model_name:
            col1, col2 = st.columns(2)
            with col1:
                params['max_depth'] = st.slider(
                    "Max depth",
                    1, 50, 10,
                    key=f"{model_name}_max_depth"
                )
                params['min_samples_split'] = st.slider(
                    "Min samples split",
                    2, 20, 2,
                    key=f"{model_name}_min_samples_split"
                )
            with col2:
                params['min_samples_leaf'] = st.slider(
                    "Min samples leaf",
                    1, 20, 1,
                    key=f"{model_name}_min_samples_leaf"
                )
                params['criterion'] = st.selectbox(
                    "Criterion",
                    ['gini', 'entropy'] if "Classifier" in model_name else ['mse', 'mae'],
                    key=f"{model_name}_criterion"
                )
        
        elif "Neural Network" in model_name or "MLP" in model_name:
            col1, col2 = st.columns(2)
            with col1:
                hidden_layers = st.text_input(
                    "Hidden layers (comma-separated)",
                    value="100,50",
                    key=f"{model_name}_hidden_layers",
                    help="E.g., 100,50 means two layers with 100 and 50 neurons"
                )
                params['hidden_layer_sizes'] = tuple([int(x.strip()) for x in hidden_layers.split(',')])
                params['activation'] = st.selectbox(
                    "Activation function",
                    ['relu', 'tanh', 'logistic'],
                    key=f"{model_name}_activation"
                )
            with col2:
                params['learning_rate'] = st.selectbox(
                    "Learning rate",
                    ['constant', 'invscaling', 'adaptive'],
                    key=f"{model_name}_learning_rate"
                )
                params['max_iter'] = st.slider(
                    "Max iterations",
                    100, 5000, 200, 100,
                    key=f"{model_name}_max_iter"
                )
        
        else:
            st.info(f"Using default parameters for {model_name}")
        
        return params
    
    def _train_models(self, selected_models, enable_tuning=False, model_params=None):
        """Train selected models with custom parameters."""
        try:
            X_train = st.session_state.X_train
            X_test = st.session_state.X_test
            y_train = st.session_state.y_train
            y_test = st.session_state.y_test
            
            is_classification = st.session_state.preprocessing_info['target_info']['is_classification']
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_container = st.container()
            
            model_results = []
            
            for i, model_name in enumerate(selected_models):
                try:
                    status_text.text(f"🤖 Training {model_name}... ({i+1}/{len(selected_models)})")
                    
                    # Get custom parameters for this model
                    custom_params = model_params.get(model_name, {}) if model_params else {}
                    
                    # Train model with custom parameters
                    model = model_manager.train_model(model_name, X_train, y_train, **custom_params)
                    
                    # Evaluate model
                    if is_classification:
                        eval_results = evaluator.evaluate_classification_model(
                            model, X_test, y_test, model_name,
                            cross_validate=st.session_state.get('cross_validate', True)
                        )
                    else:
                        eval_results = evaluator.evaluate_regression_model(
                            model, X_test, y_test, model_name,
                            cross_validate=st.session_state.get('cross_validate', True)
                        )
                    
                    model_results.append(eval_results)
                    
                    # Save model if requested
                    if st.session_state.get('save_models', True):
                        model_manager.save_model(model_name)
                    
                    # Display quick results
                    with results_container:
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.success(f"✅ {model_name}")
                        with col2:
                            if is_classification:
                                st.write(f"Accuracy: {eval_results.get('accuracy', 0):.4f}")
                            else:
                                st.write(f"R² Score: {eval_results.get('r2_score', 0):.4f}")
                
                except Exception as e:
                    st.error(f"❌ Failed to train {model_name}: {str(e)}")
                    logger.error(f"Model training error for {model_name}: {str(e)}")
                
                # Update progress
                progress_bar.progress((i + 1) / len(selected_models))
            
            # Store results
            st.session_state.model_results = model_results
            st.session_state.model_trained = True
            
            status_text.text("🎉 Training complete!")
            st.success(f"🎉 Successfully trained {len([r for r in model_results if r])} models!")
            
            logger.info(f"Training completed for {len(model_results)} models")
            
        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            logger.error(f"Model training failed: {str(e)}")
    
    def _render_model_evaluation_tab(self):
        """Render model evaluation tab with classification/regression specific metrics."""
        st.header("📈 Model Evaluation")
        
        if not st.session_state.get('model_trained', False):
            st.warning("⚠️ Please train models first!")
            return
        
        model_results = st.session_state.model_results
        
        if not model_results:
            st.error("No model results available!")
            return
        
        is_classification = st.session_state.preprocessing_info['target_info']['is_classification']
        
        # Model comparison table
        st.subheader("🏆 Model Comparison")
        comparison_df = evaluator.compare_models(model_results)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Best model
        best_model = evaluator.get_best_model(model_results)
        if best_model:
            st.success(f"🏆 **Best Model:** {best_model}")
        
        # Model comparison visualization
        if len(model_results) > 1:
            st.subheader("📊 Performance Comparison")
            comparison_chart = visualizer.create_model_comparison_chart(comparison_df)
            if comparison_chart:
                visualizer.display_image_from_base64(
                    comparison_chart,
                    "Model performance comparison"
                )
        
        # Detailed model analysis
        st.subheader("🔍 Detailed Model Analysis")
        selected_model_name = st.selectbox(
            "Select model for detailed analysis",
            [result['model_name'] for result in model_results]
        )
        
        if selected_model_name:
            selected_result = next(
                (r for r in model_results if r['model_name'] == selected_model_name),
                None
            )
            
            if selected_result:
                # Display metrics
                evaluator.display_evaluation_results(selected_result)
                
                # Task-specific visualizations
                if is_classification:
                    st.subheader("📊 Classification Metrics & Visualizations")
                    self._render_classification_visualizations(selected_model_name, selected_result)
                else:
                    st.subheader("📈 Regression Metrics & Visualizations")
                    self._render_regression_visualizations(selected_model_name, selected_result)
        
        # Export options
        st.subheader("💾 Export Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Download Report"):
                self._export_report(model_results, comparison_df)
        
        with col2:
            if st.button("💾 Save Models"):
                self._download_models()
        
        with col3:
            if st.button("🔄 New Analysis"):
                self._reset_session()
    
    def _render_classification_visualizations(self, model_name, results):
        """Render classification-specific visualizations."""
        try:
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            
            model = model_manager.get_trained_model(model_name)
            y_pred = model.predict(X_test)
            
            col1, col2 = st.columns(2)
            
            # Confusion Matrix
            with col1:
                st.markdown("**Confusion Matrix:**")
                if 'confusion_matrix' in results:
                    cm_plot = visualizer.create_confusion_matrix(
                        y_test, y_pred, model_name=model_name
                    )
                    if cm_plot:
                        visualizer.display_image_from_base64(cm_plot, "Confusion Matrix")
            
            # ROC Curves
            with col2:
                st.markdown("**ROC Curve:**")
                if hasattr(model, 'predict_proba'):
                    try:
                        y_proba = model.predict_proba(X_test)
                        roc_plot = visualizer.create_roc_curves(
                            y_test, y_proba, model_name=model_name
                        )
                        if roc_plot:
                            visualizer.display_image_from_base64(roc_plot, "ROC Curves")
                    except Exception as e:
                        st.info("ROC curve not available for this model")
            
            # Precision-Recall Curve (for binary classification)
            if hasattr(model, 'predict_proba'):
                try:
                    y_proba = model.predict_proba(X_test)
                    if y_proba.shape[1] == 2:
                        st.markdown("**Precision-Recall Curve:**")
                        pr_plot = visualizer.create_precision_recall_curve(
                            y_test, y_proba, model_name=model_name
                        )
                        if pr_plot:
                            visualizer.display_image_from_base64(pr_plot, "Precision-Recall Curve")
                except Exception as e:
                    logger.warning(f"Could not generate PR curve: {str(e)}")
            
            # Classification metrics
            st.markdown("**Classification Metrics:**")
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            with metrics_col1:
                st.metric("Accuracy", f"{results.get('accuracy', 0):.4f}")
            with metrics_col2:
                st.metric("Precision", f"{results.get('precision', 0):.4f}")
            with metrics_col3:
                st.metric("Recall", f"{results.get('recall', 0):.4f}")
            with metrics_col4:
                st.metric("F1 Score", f"{results.get('f1_score', 0):.4f}")
            
        except Exception as e:
            logger.error(f"Error generating classification visualizations: {str(e)}")
            st.error("Could not generate some visualizations")
    
    def _render_regression_visualizations(self, model_name, results):
        """Render regression-specific visualizations."""
        try:
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            
            model = model_manager.get_trained_model(model_name)
            y_pred = model.predict(X_test)
            
            # Residuals plot
            st.markdown("**Residuals Analysis:**")
            residuals_plot = visualizer.create_residuals_plot(
                y_test, y_pred, model_name=model_name
            )
            if residuals_plot:
                visualizer.display_image_from_base64(residuals_plot, "Residuals Plot")
            
            # Regression metrics
            st.markdown("**Regression Metrics:**")
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            with metrics_col1:
                st.metric("R² Score", f"{results.get('r2_score', 0):.4f}")
            with metrics_col2:
                st.metric("MAE", f"{results.get('mae', 0):.4f}")
            with metrics_col3:
                st.metric("MSE", f"{results.get('mse', 0):.4f}")
            with metrics_col4:
                st.metric("RMSE", f"{results.get('rmse', 0):.4f}")
            
            # Prediction vs Actual scatter plot
            st.markdown("**Predictions vs Actual:**")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Sample Predictions:")
                pred_df = pd.DataFrame({
                    'Actual': y_test[:10],
                    'Predicted': y_pred[:10],
                    'Error': y_test[:10] - y_pred[:10]
                })
                st.dataframe(pred_df, use_container_width=True)
            
            with col2:
                st.write("Error Distribution:")
                errors = y_test - y_pred
                error_series = pd.Series(errors, name='Prediction Errors')
                error_dist = visualizer.create_target_distribution(error_series, "Errors")
                if error_dist:
                    visualizer.display_image_from_base64(error_dist, "Error distribution")
            
        except Exception as e:
            logger.error(f"Error generating regression visualizations: {str(e)}")
            st.error("Could not generate some visualizations")
    
    def _export_report(self, model_results, comparison_df):
        """Export analysis report."""
        try:
            # Create CSV report
            csv_buffer = io.StringIO()
            comparison_df.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="📄 Download Report (CSV)",
                data=csv_buffer.getvalue(),
                file_name=f"ml_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            st.success("✅ Report ready for download!")
            
        except Exception as e:
            st.error(f"Failed to create report: {str(e)}")
            logger.error(f"Report export error: {str(e)}")
    
    def _download_models(self):
        """Provide model download information."""
        try:
            saved_models = model_manager.list_saved_models()
            
            if saved_models:
                st.success(f"✅ {len(saved_models)} models saved!")
                st.info("📁 Models are saved in the 'saved_models' folder")
                
                st.markdown("**Saved models:**")
                for model_file in saved_models:
                    st.write(f"• {model_file}")
            else:
                st.warning("No models were saved")
        
        except Exception as e:
            st.error(f"Error accessing saved models: {str(e)}")
            logger.error(f"Model download error: {str(e)}")
    
    def _reset_session(self):
        """Reset session state for new analysis."""
        keys_to_reset = [
            'data_uploaded', 'data_validated', 'data_preprocessed', 'model_trained',
            'target_column', 'X_train', 'X_test', 'y_train', 'y_test',
            'preprocessing_info', 'model_results', 'selected_models', 'data'
        ]
        
        for key in keys_to_reset:
            if key in st.session_state:
                del st.session_state[key]
        
        st.rerun()

def main():
    """Main application entry point."""
    try:
        logger.info("Starting Accurate ML Application")
        app = AccurateApp()
        app.run()
        
    except Exception as e:
        logger.exception("Critical application error")
        st.error("A critical error occurred. Please refresh the page.")
        if config.get('app.debug', False):
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
