import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, classification_report
from sklearn.inspection import permutation_importance
import xgboost as xgb

# -----------------------------------------------------------------------------
# CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Rogers' Diffusion Analytics",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #334155;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
    }
    .insight-box {
        background-color: #eff6ff;
        border-left: 5px solid #2563eb;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
        color: #1e293b;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SESSION STATE MANAGEMENT
# -----------------------------------------------------------------------------
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_meta' not in st.session_state:
    st.session_state.model_meta = {}

# -----------------------------------------------------------------------------
# UTILITY FUNCTIONS
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(file):
    """Loads CSV or Excel data with caching."""
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def determine_task_type(df, target_col):
    """
    Heuristic: 
    - If target has < 15 unique values -> Classification (Categories/Ratings)
    - If target has > 15 unique values -> Regression (Continuous Score/Time)
    """
    col_data = df[target_col]
    if pd.api.types.is_numeric_dtype(col_data):
        if col_data.nunique() < 15:
            return "Classification"
        return "Regression"
    return "Classification"

# -----------------------------------------------------------------------------
# MAIN LAYOUT
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">Engineering Adoption: Rogers\' Framework</div>', unsafe_allow_html=True)
st.markdown("*A Computational Framework for Sociometry and Data Engineering*")

with st.sidebar:
    st.header("Navigation")
    mode = st.radio("Select Module:", 
        ["1. Data Ingestion & EDA", 
         "2. Model Training Engine", 
         "3. Prediction Playground"]
    )
    st.divider()
    st.info("""
    **Attributes (Rogers, 1962):**
    1. Relative Advantage
    2. Compatibility
    3. Complexity
    4. Trialability
    5. Observability
    """)

# -----------------------------------------------------------------------------
# MODULE 1: DATA INGESTION & EDA
# -----------------------------------------------------------------------------
if mode == "1. Data Ingestion & EDA":
    st.markdown('<div class="sub-header">1. Data Ingestion</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Dataset (.csv or .xlsx)", type=['csv', 'xlsx'])
    
    # 1.1 Load Data
    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None:
            st.session_state.df = df
            st.success(f"Data Loaded: {df.shape[0]} respondents, {df.shape[1]} attributes.")
            
    # 1.2 Visualization
    if st.session_state.df is not None:
        df = st.session_state.df
        
        with st.expander("View Raw Data", expanded=False):
            st.dataframe(df, use_container_width=True)
            
        st.markdown('<div class="sub-header">2. Attribute Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Distribution Inspector**")
            viz_target = st.selectbox("Select Attribute", df.columns)
            plot_type = st.radio("Chart Type", ["Histogram", "Box Plot"])
            
        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            if plot_type == "Histogram":
                sns.histplot(df[viz_target], kde=True, ax=ax, color='#2563eb')
                ax.set_title(f"Distribution of {viz_target}")
            elif plot_type == "Box Plot":
                sns.boxplot(x=df[viz_target], ax=ax, color='#60a5fa')
                ax.set_title(f"Spread of {viz_target}")
            st.pyplot(fig)
            
        # Correlation Matrix
        st.markdown('<div class="sub-header">3. Correlation Analysis (Rogers\' Drivers)</div>', unsafe_allow_html=True)
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            fig_corr, ax_corr = plt.subplots(figsize=(10, 5))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
            st.pyplot(fig_corr)

# -----------------------------------------------------------------------------
# MODULE 2: MODEL TRAINING ENGINE
# -----------------------------------------------------------------------------
elif mode == "2. Model Training Engine":
    st.markdown('<div class="sub-header">Model Configuration</div>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("Please upload data in Module 1 first.")
        st.stop()
        
    df = st.session_state.df.copy() # Work on a copy
    
    # 2.1 Feature Selection
    c1, c2 = st.columns(2)
    with c1:
        # Default to last column as target
        target_col = st.selectbox("Target Variable (Adoption Outcome)", df.columns, index=len(df.columns)-1)
    with c2:
        # Default to all other columns
        candidates = [c for c in df.columns if c != target_col and c != 'Respondent_ID']
        feature_cols = st.multiselect("Predictor Attributes (Features)", candidates, default=candidates)
        
    if not feature_cols:
        st.stop()
        
    # 2.2 Algorithm & Task Logic
    task_type = determine_task_type(df, target_col)
    
    st.markdown(f"""
    <div class='insight-box'>
        <b>System Intelligence:</b> Based on the cardinality of <code>{target_col}</code>, 
        the system has selected a <b>{task_type}</b> strategy.
    </div>
    """, unsafe_allow_html=True)
    
    col_algo, col_hyper = st.columns(2)
    with col_algo:
        algorithm = st.selectbox("Machine Learning Algorithm", ["Random Forest (Bagging)", "XGBoost (Boosting)"])
    with col_hyper:
        n_estimators = st.slider("Model Complexity (Trees)", 50, 500, 100)

    # 2.3 Training Pipeline
    if st.button("Train Model", type="primary"):
        with st.spinner("Training Ensemble Model..."):
            try:
                # A. Prepare X and y
                X = df[feature_cols]
                y = df[target_col]
                
                # B. Categorical Encoding (One-Hot)
                X_encoded = pd.get_dummies(X, drop_first=True)
                
                # C. Label Encoding (for Classification targets)
                le = None
                if task_type == "Classification":
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                
                # D. Split Data
                X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
                
                # E. Initialize Model
                if task_type == "Classification":
                    if "Random Forest" in algorithm:
                        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                    else:
                        model = xgb.XGBClassifier(n_estimators=n_estimators, random_state=42)
                else: # Regression
                    if "Random Forest" in algorithm:
                        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                    else:
                        model = xgb.XGBRegressor(n_estimators=n_estimators, random_state=42)
                        
                # F. Fit Model
                model.fit(X_train, y_train)
                
                # G. Save to Session State (Critical for Prediction Playground)
                st.session_state.model = model
                st.session_state.model_meta = {
                    'target_col': target_col,
                    'feature_cols': feature_cols,
                    'task_type': task_type,
                    'label_encoder': le,
                    'train_columns': X_encoded.columns.tolist(), # SAVED FOR SCHEMA ALIGNMENT
                    'algorithm': algorithm
                }
                
                st.success(f"Successfully trained {algorithm} on {len(X_train)} samples.")
                
                # 2.4 Evaluation Metrics
                st.markdown('<div class="sub-header">Performance Metrics</div>', unsafe_allow_html=True)
                y_pred = model.predict(X_test)
                
                m1, m2 = st.columns(2)
                if task_type == "Classification":
                    acc = accuracy_score(y_test, y_pred)
                    m1.metric("Model Accuracy", f"{acc:.1%}")
                    with st.expander("Detailed Classification Report"):
                        target_names = [str(c) for c in le.classes_] if le else None
                        st.code(classification_report(y_test, y_pred, target_names=target_names))
                else:
                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    m1.metric("R² Score (Variance Explained)", f"{r2:.4f}")
                    m2.metric("Mean Squared Error", f"{mse:.4f}")

                # 2.5 Feature Importance (Permutation)
                st.markdown('<div class="sub-header">Attribute Importance (Permutation)</div>', unsafe_allow_html=True)
                st.caption("Which Rogers' attributes are driving the adoption decision?")
                
                result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
                sorted_idx = result.importances_mean.argsort()
                
                fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
                ax_imp.barh(X_test.columns[sorted_idx], result.importances_mean[sorted_idx], color='#1E3A8A')
                ax_imp.set_xlabel("Impact on Model Prediction")
                st.pyplot(fig_imp)
                
            except Exception as e:
                st.error(f"Training Failed: {e}")

# -----------------------------------------------------------------------------
# MODULE 3: PREDICTION PLAYGROUND
# -----------------------------------------------------------------------------
elif mode == "3. Prediction Playground":
    st.markdown('<div class="sub-header">Simulation Interface</div>', unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.warning("No active model. Please train a model in Module 2 first.")
        st.stop()
        
    model = st.session_state.model
    meta = st.session_state.model_meta
    df = st.session_state.df
    
    st.info(f"Using **{meta['algorithm']}** to predict **{meta['target_col']}**")
    
    # 3.1 Dynamic Input Form
    with st.form("simulation_form"):
        st.markdown("**Configure Innovation Attributes**")
        input_data = {}
        
        # Grid layout for inputs
        cols = st.columns(2)
        
        for idx, col_name in enumerate(meta['feature_cols']):
            active_col = cols[idx % 2]
            
            # Check data type of the original column to decide widget
            if pd.api.types.is_numeric_dtype(df[col_name]):
                min_val = float(df[col_name].min())
                max_val = float(df[col_name].max())
                step = 1.0 if df[col_name].dtype == 'int64' else 0.1
                default_val = float(df[col_name].median())
                
                input_data[col_name] = active_col.slider(
                    f"{col_name}", min_val, max_val, default_val, step=step
                )
            else:
                # Categorical
                options = df[col_name].unique().tolist()
                input_data[col_name] = active_col.selectbox(f"{col_name}", options)
        
        submitted = st.form_submit_button("Simulate Adoption Outcome")
        
    # 3.2 Prediction Logic
    if submitted:
        # Create DataFrame from input
        input_df = pd.DataFrame([input_data])
        
        # One-Hot Encoding (same as training)
        input_encoded = pd.get_dummies(input_df)
        
        # SCHEMA ALIGNMENT (CRITICAL STEP)
        # Force input to have exact same columns as training data, filling missing with 0
        final_input = input_encoded.reindex(columns=meta['train_columns'], fill_value=0)
        
        # Make Prediction
        prediction = model.predict(final_input)
        
        st.divider()
        st.markdown("### Simulation Results")
        
        c_res1, c_res2 = st.columns([1, 2])
        
        with c_res1:
            if meta['task_type'] == "Classification":
                # Decode Label if necessary
                le = meta['label_encoder']
                res_label = le.inverse_transform([prediction[0]])[0] if le else prediction[0]
                st.metric("Predicted Status", f"{res_label}")
            else:
                st.metric("Predicted Score", f"{prediction[0]:.2f}")

        with c_res2:
             if meta['task_type'] == "Classification" and hasattr(model, "predict_proba"):
                 # Show probabilities for classes
                 probs = model.predict_proba(final_input)
                 prob_df = pd.DataFrame(probs, columns=le.classes_ if le else model.classes_)
                 st.bar_chart(prob_df.T)
                 st.caption("Probability distribution across adopter categories.")
