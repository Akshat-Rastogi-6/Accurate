"""
Main Streamlit application for Accurate ML Platform.
A modular, robust, and production-ready machine learning application.
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
            'current_step': 'upload'
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
        Just upload your data and we'll help you build and compare different AI models
        to solve your problem. No coding required - we'll guide you through each step!
        """)
        
        # Progress indicator
        progress_steps = ["📤 Upload Data", "✅ Check Data", "🔧 Prepare Data", "🤖 Train Models", "📊 See Results"]
        current_step_idx = {
            'upload': 0, 'validate': 1, 'preprocess': 2, 'train': 3, 'evaluate': 4
        }.get(st.session_state.current_step, 0)
        
        cols = st.columns(len(progress_steps))
        for i, (col, step) in enumerate(zip(cols, progress_steps)):
            with col:
                if i <= current_step_idx:
                    st.success(step)
                else:
                    st.info(step)
    
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
                        "How much data to use for testing?", 0.1, 0.5, 
                        config.get('models.default_test_size', 0.2), 0.05,
                        help="Higher values = more data for testing, less for training"
                    )
                    
                    st.session_state.cross_validate = st.checkbox(
                        "Use Cross-Validation for Better Results", True,
                        help="This makes results more reliable but takes longer"
                    )
                    
                    st.session_state.save_models = st.checkbox(
                        "Save Trained Models", 
                        config.get('models.save_models', True),
                        help="Save your models so you can use them later"
                    )
            
            # Help section
            with st.expander("ℹ️ Need Help?"):
                st.markdown("""
                **How to use Accurate:**
                1. 📤 Upload your data file (CSV, Excel, or JSON)
                2. 🎯 Choose what you want to predict
                3. 🔧 Let us prepare your data
                4. 🤖 Pick AI models to try
                5. 📊 Compare results and find the best one
                
                **What files can I upload?**
                - CSV files (.csv) - most common
                - Excel files (.xlsx) 
                - JSON files (.json)
                
                **Tips for best results:**
                - Make sure your data has a clear target column
                - More data usually means better results
                - Try multiple models to find the best one
                - Check data quality before training
                """)
    
    def _render_main_content(self):
        """Render main application content."""
        # Step 1: Data Upload
        if st.session_state.current_step == 'upload':
            self._render_data_upload()
        
        # Step 2: Data Validation
        elif st.session_state.current_step == 'validate':
            self._render_data_validation()
        
        # Step 3: Data Preprocessing
        elif st.session_state.current_step == 'preprocess':
            self._render_data_preprocessing()
        
        # Step 4: Model Training
        elif st.session_state.current_step == 'train':
            self._render_model_training()
        
        # Step 5: Model Evaluation
        elif st.session_state.current_step == 'evaluate':
            self._render_model_evaluation()
    
    def _render_data_upload(self):
        """Render data upload interface."""
        st.header("📤 Upload Your Data")
        
        st.markdown("""
        Let's start by uploading your dataset! We support CSV, Excel, and JSON files.
        Don't worry - we'll check your data and help you every step of the way.
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
                
                # Display basic info
                st.success(f"🎉 Great! We loaded your dataset: {uploaded_file.name}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Rows of Data", f"{self.data.shape[0]:,}")
                with col2:
                    st.metric("📋 Columns", f"{self.data.shape[1]:,}")
                with col3:
                    st.metric("💾 File Size", f"{uploaded_file.size / 1024:.1f} KB")
                
                # Show data preview
                st.subheader("� Let's Look at Your Data")
                st.dataframe(self.data.head(10), use_container_width=True)
                
                # Data overview visualizations
                st.subheader("� Data Insights")
                with st.spinner("Creating visualizations..."):
                    overview_plots = visualizer.create_data_overview(self.data)
                    
                    if 'correlation_matrix' in overview_plots:
                        st.subheader("🔗 How Your Data is Connected")
                        visualizer.display_image_from_base64(
                            overview_plots['correlation_matrix'],
                            "Shows relationships between different columns"
                        )
                    
                    if 'dtype_distribution' in overview_plots:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("📊 Types of Data")
                            visualizer.display_image_from_base64(
                                overview_plots['dtype_distribution'],
                                "Different types of data in your file"
                            )
                        
                        if 'missing_values_heatmap' in overview_plots:
                            with col2:
                                st.subheader("❓ Missing Information")
                                visualizer.display_image_from_base64(
                                    overview_plots['missing_values_heatmap'],
                                    "Shows where data is missing"
                                )
                
                # Proceed to validation
                if st.button("➡️ Next: Check My Data", type="primary"):
                    st.session_state.data_uploaded = True
                    st.session_state.current_step = 'validate'
                    st.rerun()
                
                logger.info(f"Dataset loaded successfully: {self.data.shape}")
                
            except Exception as e:
                st.error(f"❌ Error loading dataset: {str(e)}")
                logger.error(f"Data loading error: {str(e)}")
    
    def _render_data_validation(self):
        """Render data validation interface."""
        st.header("✅ Let's Check Your Data")
        
        st.markdown("""
        Now we'll check your data quality and help you choose what you want to predict.
        Don't worry if there are some issues - we can fix most problems automatically!
        """)
        
        # Get data from session state
        if 'data' in st.session_state:
            self.data = st.session_state.data
        
        if self.data is not None:
            with st.spinner("Checking your data quality..."):
                self.validation_results = validator.validate_dataframe(self.data)
            
            # Display validation results
            validator.display_validation_report(self.validation_results)
            
            # Target column suggestions
            if self.validation_results['is_valid']:
                st.subheader("🎯 What Do You Want to Predict?")
                st.markdown("Choose the column that contains what you want the AI to learn to predict:")
                
                suggested_targets = validator.suggest_target_columns(self.data)
                if suggested_targets:
                    st.info(f"💡 **We suggest these columns:** {', '.join(suggested_targets)}")
                
                target_column = st.selectbox(
                    "Select your target column (what to predict)",
                    options=self.data.columns.tolist(),
                    index=0 if not suggested_targets else self.data.columns.tolist().index(suggested_targets[0]),
                    help="This is what your AI model will learn to predict"
                )
                
                # Show target distribution
                if target_column:
                    st.subheader("📈 Let's Look at What You Want to Predict")
                    target_dist_plot = visualizer.create_target_distribution(
                        self.data[target_column], target_column
                    )
                    if target_dist_plot:
                        visualizer.display_image_from_base64(
                            target_dist_plot, 
                            f"Distribution of {target_column}"
                        )
                
                # Proceed to preprocessing
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("⬅️ Back to Upload"):
                        st.session_state.current_step = 'upload'
                        st.rerun()
                
                with col2:
                    if st.button("➡️ Next: Prepare Data", type="primary"):
                        if target_column:
                            st.session_state.target_column = target_column
                            st.session_state.data_validated = True
                            st.session_state.current_step = 'preprocess'
                            st.rerun()
                        else:
                            st.error("Please select what you want to predict")
        else:
            st.error("We need your data first. Let's go back and upload it!")
            if st.button("⬅️ Back to Upload"):
                st.session_state.current_step = 'upload'
                st.rerun()
    
    def _render_data_preprocessing(self):
        """Render data preprocessing interface."""
        st.header("🔧 Let's Prepare Your Data")
        
        st.markdown("""
        Now we'll prepare your data for the AI models. We can handle missing values,
        convert text to numbers, and scale your data. Don't worry - we'll explain what we're doing!
        """)
        
        # Get data from session state
        if 'data' in st.session_state:
            self.data = st.session_state.data
        
        if self.data is not None and hasattr(st.session_state, 'target_column'):
            # Preprocessing configuration
            st.subheader("⚙️ How Should We Prepare Your Data?")
            
            # Option to skip preprocessing
            skip_preprocessing = st.checkbox(
                "🚀 **Quick Start**: Use simple data preparation", 
                help="Skip advanced preparation for faster setup. Good for clean datasets."
            )
            
            if not skip_preprocessing:
                col1, col2 = st.columns(2)
                
                with col1:
                    missing_strategy = st.selectbox(
                        "How to handle missing data?",
                        ["auto", "drop", "mean_mode", "knn"],
                        help="Auto will choose the best method for your data"
                    )
                    
                    encoding_strategy = st.selectbox(
                        "How to handle text data?",
                        ["auto", "label", "onehot", "mixed"],
                        help="Auto will convert text to numbers automatically"
                    )
                
                with col2:
                    scaling_strategy = st.selectbox(
                        "How to scale numbers?",
                        ["standard", "minmax", "robust", "none"],
                        help="This helps AI models work better with your numbers"
                    )
                    
                    test_size = st.session_state.get('test_size', 0.2)
                    st.write(f"**Data split:** {int((1-test_size)*100)}% for training, {int(test_size*100)}% for testing")
            else:
                st.info("🚀 **Quick Start Mode**: We'll use simple data preparation with basic encoding and fill missing values.")
                missing_strategy = "mean_mode"
                encoding_strategy = "label"
                scaling_strategy = "none"
                test_size = st.session_state.get('test_size', 0.2)
            
            # Run preprocessing
            button_text = "🚀 Quick Preparation" if skip_preprocessing else "🔧 Prepare My Data"
            if st.button(button_text, type="primary"):
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
                        
                        # Check if fallback was used
                        if preprocessing_info.get('fallback_used', False):
                            if preprocessing_info.get('raw_fallback', False):
                                st.warning("⚠️ Advanced preprocessing failed. Used raw numerical data only.")
                            else:
                                st.warning("⚠️ Advanced preprocessing failed. Used minimal preprocessing fallback.")
                            st.info("💡 **Tip:** Your data may have quality issues, but model training can still proceed.")
                        else:
                            st.success("✅ Preprocessing completed successfully!")
                        
                        # Display preprocessing summary
                        st.subheader("📋 Preprocessing Summary")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Original Shape", f"{preprocessing_info['original_shape'][0]}×{preprocessing_info['original_shape'][1]}")
                        with col2:
                            st.metric("Final Shape", f"{preprocessing_info['final_shape'][0]}×{preprocessing_info['final_shape'][1]}")
                        with col3:
                            st.metric("Test Size", f"{test_size:.1%}")
                        
                        # Show preprocessing steps
                        st.subheader("🔄 Applied Transformations")
                        for i, step in enumerate(preprocessing_info['preprocessing_steps'], 1):
                            if preprocessing_info.get('fallback_used', False):
                                st.write(f"{i}. {step} {'⚠️' if 'fallback' in step.lower() else ''}")
                            else:
                                st.write(f"{i}. {step}")
                        
                        # Task type detection
                        is_classification = preprocessing_info['target_info']['is_classification']
                        task_type = "Classification" if is_classification else "Regression"
                        st.info(f"🎯 **Detected Task Type:** {task_type}")
                        
                        st.session_state.data_preprocessed = True
                        
                except Exception as e:
                    st.error(f"❌ All preprocessing methods failed: {str(e)}")
                    st.error("🚫 Cannot proceed to model training without basic data preparation.")
                    logger.error(f"Preprocessing error: {str(e)}")
                    
                    # Offer to skip preprocessing (for advanced users)
                    if st.checkbox("🔧 **Advanced**: Skip preprocessing and use raw data"):
                        st.warning("⚠️ This will use only numerical columns and may cause model training issues.")
                        if st.button("⚠️ Proceed with Raw Data", type="secondary"):
                            try:
                                # Basic data split without preprocessing
                                X = self.data.select_dtypes(include=[np.number]).drop(columns=[st.session_state.target_column], errors='ignore')
                                y = self.data[st.session_state.target_column]
                                
                                if X.empty:
                                    st.error("No numerical columns available. Cannot proceed.")
                                else:
                                    X = X.fillna(0)
                                    if y.dtype == 'object':
                                        from sklearn.preprocessing import LabelEncoder
                                        encoder = LabelEncoder()
                                        y = pd.Series(encoder.fit_transform(y))
                                    
                                    from sklearn.model_selection import train_test_split
                                    X_train, X_test, y_train, y_test = train_test_split(
                                        X.values, y.values, test_size=test_size, random_state=42
                                    )
                                    
                                    # Store in session state
                                    st.session_state.X_train = X_train
                                    st.session_state.X_test = X_test
                                    st.session_state.y_train = y_train
                                    st.session_state.y_test = y_test
                                    st.session_state.preprocessing_info = {
                                        'original_shape': self.data.shape,
                                        'final_shape': X.shape,
                                        'preprocessing_steps': ['No preprocessing - raw numerical data only'],
                                        'target_info': {'is_classification': len(np.unique(y)) < 20},
                                        'test_size': test_size,
                                        'raw_mode': True
                                    }
                                    
                                    st.session_state.data_preprocessed = True
                                    st.success("✅ Raw data prepared for training!")
                                    
                            except Exception as raw_e:
                                st.error(f"❌ Even raw data preparation failed: {str(raw_e)}")
            
            # Navigation buttons
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("⬅️ Back to Validation"):
                    st.session_state.current_step = 'validate'
                    st.rerun()
            
            with col2:
                if st.session_state.get('data_preprocessed', False):
                    if st.button("➡️ Proceed to Model Training", type="primary"):
                        st.session_state.current_step = 'train'
                        st.rerun()
        
        else:
            st.error("Data preprocessing requires a validated dataset with selected target column.")
            if st.button("⬅️ Back to Validation"):
                st.session_state.current_step = 'validate'
                st.rerun()
    
    def _render_model_training(self):
        """Render model training interface."""
        st.header("🤖 Let's Train Your AI Models")
        
        st.markdown("""
        Now comes the exciting part! We'll train different AI models on your data
        and see which one works best for your problem. This might take a few minutes.
        """)
        
        if st.session_state.get('data_preprocessed', False):
            selected_models = st.session_state.get('selected_models', [])
            
            if not selected_models:
                st.warning("🎯 Please choose at least one AI model in the sidebar first!")
                return
            
            # Display training configuration
            st.subheader("🎯 Ready to Train!")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🤖 AI Models Selected", len(selected_models))
            with col2:
                st.metric("📚 Training Examples", f"{len(st.session_state.X_train):,}")
            with col3:
                st.metric("🧪 Testing Examples", f"{len(st.session_state.X_test):,}")
            
            # Start training
            if st.button("🚀 Start Training AI Models", type="primary"):
                self._train_models(selected_models)
            
            # Navigation
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("⬅️ Back to Data Prep"):
                    st.session_state.current_step = 'preprocess'
                    st.rerun()
            
            with col2:
                if st.session_state.get('model_trained', False):
                    if st.button("➡️ See My Results!", type="primary"):
                        st.session_state.current_step = 'evaluate'
                        st.rerun()
        
        else:
            st.error("We need to prepare your data first!")
            if st.button("⬅️ Back to Data Prep"):
                st.session_state.current_step = 'preprocess'
                st.rerun()
    
    def _train_models(self, selected_models):
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
            
            status_text.text("🎉 All done! Your AI models are ready!")
            st.success(f"🎉 Amazing! We successfully trained {len([r for r in model_results if r])} AI models for you!")
            
            logger.info(f"Training completed for {len(model_results)} models")
            
        except Exception as e:
            st.error(f"❌ Oops! Training had an issue: {str(e)}")
            logger.error(f"Model training failed: {str(e)}")
    
    def _render_model_evaluation(self):
        """Render model evaluation and results."""
        st.header("📊 Your AI Results Are Here!")
        
        st.markdown("""
        🎉 Fantastic! Your AI models are trained and ready. Let's see which one works best
        for your data and explore the results together.
        """)
        
        if st.session_state.get('model_trained', False):
            model_results = st.session_state.model_results
            
            if not model_results:
                st.error("Hmm, we don't see any results. Let's try training again!")
                return
            
            # Model comparison table
            st.subheader("🏆 Which AI Model Won?")
            comparison_df = evaluator.compare_models(model_results)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Best model identification
            best_model = evaluator.get_best_model(model_results)
            if best_model:
                st.success(f"🏆 **Winner:** {best_model} performed the best on your data!")
            
            # Model comparison visualization
            if len(model_results) > 1:
                st.subheader("📊 Visual Comparison")
                comparison_chart = visualizer.create_model_comparison_chart(comparison_df)
                if comparison_chart:
                    visualizer.display_image_from_base64(
                        comparison_chart,
                        "Compare how different AI models performed"
                    )
            
            # Detailed model results
            st.subheader("🔍 Dive Deeper Into Results")
            selected_model_for_details = st.selectbox(
                "Pick a model to explore in detail",
                [result['model_name'] for result in model_results],
                help="See detailed analysis for any model you trained"
            )
            
            if selected_model_for_details:
                selected_result = next(
                    (r for r in model_results if r['model_name'] == selected_model_for_details),
                    None
                )
                
                if selected_result:
                    # Display evaluation metrics
                    evaluator.display_evaluation_results(selected_result)
                    
                    # Generate visualizations
                    self._render_model_visualizations(selected_model_for_details, selected_result)
            
            # Export options
            st.subheader("💾 Save Your Work")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📄 Download Report"):
                    self._export_report(model_results, comparison_df)
            
            with col2:
                if st.button("💾 Save AI Models"):
                    self._download_models()
            
            with col3:
                if st.button("🔄 Try New Data"):
                    self._reset_session()
            
            # Navigation
            if st.button("⬅️ Back to Training"):
                st.session_state.current_step = 'train'
                st.rerun()
        
        else:
            st.error("We need to train some models first!")
            if st.button("⬅️ Back to Training"):
                st.session_state.current_step = 'train'
                st.rerun()
    
    def _render_model_visualizations(self, model_name, results):
        """Render visualizations for a specific model."""
        try:
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            
            model = model_manager.get_trained_model(model_name)
            y_pred = model.predict(X_test)
            
            is_classification = st.session_state.preprocessing_info['target_info']['is_classification']
            
            if is_classification:
                # Confusion Matrix
                if 'confusion_matrix' in results:
                    st.subheader("🎯 Confusion Matrix")
                    cm_plot = visualizer.create_confusion_matrix(
                        y_test, y_pred, model_name=model_name
                    )
                    if cm_plot:
                        visualizer.display_image_from_base64(cm_plot)
                
                # ROC Curves (if probabilities available)
                if hasattr(model, 'predict_proba'):
                    try:
                        y_proba = model.predict_proba(X_test)
                        
                        st.subheader("📈 ROC Curves")
                        roc_plot = visualizer.create_roc_curves(
                            y_test, y_proba, model_name=model_name
                        )
                        if roc_plot:
                            visualizer.display_image_from_base64(roc_plot)
                        
                        # Precision-Recall Curve (for binary classification)
                        if y_proba.shape[1] == 2:
                            st.subheader("📊 Precision-Recall Curve")
                            pr_plot = visualizer.create_precision_recall_curve(
                                y_test, y_proba, model_name=model_name
                            )
                            if pr_plot:
                                visualizer.display_image_from_base64(pr_plot)
                    
                    except Exception as e:
                        logger.warning(f"Could not generate probability-based plots: {str(e)}")
            
            else:
                # Regression visualizations
                st.subheader("📈 Residuals Analysis")
                residuals_plot = visualizer.create_residuals_plot(
                    y_test, y_pred, model_name=model_name
                )
                if residuals_plot:
                    visualizer.display_image_from_base64(residuals_plot)
            
            # Feature importance (if available)
            if 'feature_importance' in results and results['feature_importance']:
                st.subheader("🔍 What Matters Most")
                st.markdown("These features had the biggest impact on predictions:")
                # Create dummy feature names (in production, you'd store actual feature names)
                feature_names = [f'Feature_{i}' for i in range(len(results['feature_importance']))]
                
                importance_plot = visualizer.create_feature_importance_plot(
                    feature_names, results['feature_importance'], model_name=model_name
                )
                if importance_plot:
                    visualizer.display_image_from_base64(importance_plot)
        
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            st.error(f"We couldn't create some charts, but your results are still valid!")
    
    def _export_report(self, model_results, comparison_df):
        """Export analysis report."""
        try:
            # Create a comprehensive report
            report_data = {
                'timestamp': pd.Timestamp.now(),
                'dataset_info': {
                    'original_shape': st.session_state.preprocessing_info['original_shape'],
                    'final_shape': st.session_state.preprocessing_info['final_shape'],
                    'target_column': st.session_state.target_column,
                    'task_type': 'Classification' if st.session_state.preprocessing_info['target_info']['is_classification'] else 'Regression'
                },
                'preprocessing_steps': st.session_state.preprocessing_info['preprocessing_steps'],
                'model_comparison': comparison_df.to_dict('records'),
                'detailed_results': model_results
            }
            
            # Convert to CSV for download
            csv_buffer = io.StringIO()
            comparison_df.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="📄 Download Your Report",
                data=csv_buffer.getvalue(),
                file_name=f"accurate_ai_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Download a detailed report of your AI model results"
            )
            
            st.success("✅ Your report is ready to download!")
            
        except Exception as e:
            st.error(f"Couldn't create report: {str(e)}")
            logger.error(f"Report export error: {str(e)}")
    
    def _download_models(self):
        """Provide model download functionality."""
        try:
            saved_models = model_manager.list_saved_models()
            
            if saved_models:
                st.success(f"✅ Great! We saved {len(saved_models)} AI models for you!")
                st.info("📁 Your trained models are saved in the 'saved_models' folder.")
                
                # Display saved models
                st.markdown("**Your saved models:**")
                for model_file in saved_models:
                    st.write(f"• {model_file}")
            else:
                st.warning("No models were saved. You can enable 'Save AI Models' in the sidebar next time!")
        
        except Exception as e:
            st.error(f"Couldn't access saved models: {str(e)}")
            logger.error(f"Model download error: {str(e)}")
    
    def _reset_session(self):
        """Reset session state for new analysis."""
        keys_to_reset = [
            'data_uploaded', 'data_validated', 'data_preprocessed', 'model_trained',
            'target_column', 'X_train', 'X_test', 'y_train', 'y_test',
            'preprocessing_info', 'model_results', 'selected_models'
        ]
        
        for key in keys_to_reset:
            if key in st.session_state:
                del st.session_state[key]
        
        st.session_state.current_step = 'upload'
        st.rerun()

def main():
    """Main application entry point."""
    try:
        logger.info("Starting Accurate ML Application")
        app = AccurateApp()
        app.run()
        
    except Exception as e:
        logger.exception("Critical application error")
        st.error("A critical error occurred. Please refresh the page and try again.")
        if config.get('app.debug', False):
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()