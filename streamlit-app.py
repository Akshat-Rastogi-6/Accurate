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
            
            # Model selection (if data is available)
            if st.session_state.data_uploaded:
                st.subheader("🤖 Choose AI Models")
                
                # Model category filter
                categories = ["All"] + list(model_registry.model_categories.keys())
                selected_category = st.selectbox("Model Type", categories)
                
                # Get available models
                if selected_category == "All":
                    available_models = model_registry.get_all_models()
                else:
                    available_models = model_registry.get_models_by_category(selected_category)
                
                # Model selection
                selected_models = st.multiselect(
                    "Select Models to Try",
                    available_models,
                    default=available_models[:3] if len(available_models) >= 3 else available_models,
                    help="Pick multiple models to compare their performance"
                )
                
                st.session_state.selected_models = selected_models
                
                # Advanced options
                with st.expander("⚙️ More Options"):
                    st.session_state.test_size = st.slider(
                        "Test data size", 0.1, 0.5, 
                        config.get('models.default_test_size', 0.2), 0.05
                    )
                    
                    st.session_state.cross_validate = st.checkbox(
                        "Use Cross-Validation", True
                    )
                    
                    st.session_state.save_models = st.checkbox(
                        "Save Trained Models", 
                        config.get('models.save_models', True)
                    )
            
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
        """Render data preprocessing tab."""
        st.header("🔧 Data Preprocessing")
        
        if not st.session_state.get('data_validated', False):
            st.warning("⚠️ Please select a target column in the Data Ingestion tab first!")
            return
        
        # Get data from session state
        if 'data' in st.session_state:
            self.data = st.session_state.data
            
            st.markdown("""
            Configure how to prepare your data for machine learning models.
            """)
            
            # Preprocessing configuration
            st.subheader("⚙️ Preprocessing Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                missing_strategy = st.selectbox(
                    "Missing values strategy",
                    ["auto", "drop", "mean_mode", "knn"],
                    help="How to handle missing data"
                )
                
                encoding_strategy = st.selectbox(
                    "Categorical encoding",
                    ["auto", "label", "onehot", "mixed"],
                    help="How to convert text to numbers"
                )
            
            with col2:
                scaling_strategy = st.selectbox(
                    "Numerical scaling",
                    ["standard", "minmax", "robust", "none"],
                    help="How to scale numerical features"
                )
                
                test_size = st.slider(
                    "Test set size",
                    0.1, 0.5, st.session_state.get('test_size', 0.2), 0.05
                )
            
            # Run preprocessing
            if st.button("🔧 Preprocess Data", type="primary"):
                try:
                    with st.spinner("Preprocessing data..."):
                        preprocessing_options = {
                            'missing_strategy': missing_strategy,
                            'encoding_strategy': encoding_strategy,
                            'scaling_strategy': scaling_strategy
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
                        
                        st.success("✅ Preprocessing completed!")
                        
                        # Display summary
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
                        
                except Exception as e:
                    st.error(f"❌ Preprocessing failed: {str(e)}")
                    logger.error(f"Preprocessing error: {str(e)}")
            
            # Show current status
            if st.session_state.get('data_preprocessed', False):
                st.success("✅ Data is preprocessed and ready for model training!")
        else:
            st.info("👆 Please upload data first.")
    
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
        """Render model training tab with hyperparameter tuning."""
        st.header("🤖 Model Training & Hyperparameter Tuning")
        
        if not st.session_state.get('data_preprocessed', False):
            st.warning("⚠️ Please preprocess data first!")
            return
        
        selected_models = st.session_state.get('selected_models', [])
        
        if not selected_models:
            st.warning("🎯 Please select at least one model in the sidebar!")
            return
        
        # Display configuration
        st.subheader("🎯 Training Configuration")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Models Selected", len(selected_models))
        with col2:
            st.metric("Training Samples", f"{len(st.session_state.X_train):,}")
        with col3:
            st.metric("Cross-Validation", "Yes" if st.session_state.cross_validate else "No")
        
        # Hyperparameter tuning options
        st.subheader("⚙️ Hyperparameter Tuning")
        enable_tuning = st.checkbox(
            "Enable hyperparameter tuning",
            value=False,
            help="This will take longer but may improve model performance"
        )
        
        if enable_tuning:
            tuning_method = st.radio(
                "Tuning method",
                ["Grid Search", "Random Search"],
                horizontal=True
            )
            st.info("💡 Hyperparameter tuning is enabled. This may take significantly longer.")
        
        # Start training
        if st.button("🚀 Train Models", type="primary"):
            self._train_models(selected_models, enable_tuning)
        
        # Show training status
        if st.session_state.get('model_trained', False):
            st.success("✅ Models trained successfully! Check the Model Evaluation tab.")
    
    def _train_models(self, selected_models, enable_tuning=False):
        """Train selected models."""
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
                    
                    # Train model
                    model = model_manager.train_model(model_name, X_train, y_train)
                    
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
