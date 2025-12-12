import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, classification_report
from sklearn.inspection import permutation_importance
import xgboost as xgb

# ---------------------------------------------------------------------
# DEFAULT DATASET (UPDATED FILE NAME)
# ---------------------------------------------------------------------
DEFAULT_DATASET_FILENAME = "sparkonomy_synthetic_survey_30.xlsx"
DEFAULT_DATASET_PATH = os.path.join("data", DEFAULT_DATASET_FILENAME)

# ---------------------------------------------------------------------
# CONFIGURATION & STYLING
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Rogers' Diffusion Analytics",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# ---------------------------------------------------------------------
# SESSION STATE MANAGEMENT
# ---------------------------------------------------------------------
st.session_state.setdefault("df", None)
st.session_state.setdefault("model", None)
st.session_state.setdefault("model_meta", {})

# ---------------------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------------------
@st.cache_data
def load_data(file):
    """Loads CSV or Excel data with caching."""
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    if file.name.endswith((".xls", ".xlsx")):
        return pd.read_excel(file)
    raise ValueError("Unsupported file format. Please upload .csv or .xlsx")


def load_default_data_if_present():
    """Loads the default dataset from /data if it exists."""
    if os.path.exists(DEFAULT_DATASET_PATH):
        return pd.read_excel(DEFAULT_DATASET_PATH)
    return None


def determine_task_type(df, target_col):
    """
    Heuristic:
    - If numeric and >15 unique values -> Regression
    - Otherwise -> Classification
    """
    col_data = df[target_col]
    if pd.api.types.is_numeric_dtype(col_data):
        return "Regression" if col_data.nunique() > 15 else "Classification"
    return "Classification"

# ---------------------------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------------------------
st.markdown('<div class="main-header">Engineering Adoption: Rogers\' Framework</div>', unsafe_allow_html=True)
st.markdown("*A Computational Framework for Sociometry and Data Engineering*")

with st.sidebar:
    st.header("Navigation")
    mode = st.radio(
        "Select Module:",
        ["1. Data Ingestion & EDA", "2. Model Training Engine", "3. Prediction Playground"]
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

# ---------------------------------------------------------------------
# MODULE 1: DATA INGESTION & EDA
# ---------------------------------------------------------------------
if mode == "1. Data Ingestion & EDA":
    st.markdown('<div class="sub-header">1. Data Ingestion</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Dataset (.csv or .xlsx)", type=["csv", "xlsx"])

    # Load uploaded OR default dataset
    if uploaded_file:
        try:
            df = load_data(uploaded_file)
            st.session_state.df = df
            st.success(f"Data Loaded: {df.shape[0]} respondents, {df.shape[1]} attributes.")
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")
            st.stop()
    else:
        if st.session_state.df is None:
            df_default = load_default_data_if_present()
            if df_default is not None:
                st.session_state.df = df_default
                st.info(f"Loaded default dataset: `{DEFAULT_DATASET_PATH}`")
                st.success(f"Data Loaded: {df_default.shape[0]} respondents, {df_default.shape[1]} attributes.")
            else:
                st.warning(
                    f"No uploaded file detected and default dataset not found at `{DEFAULT_DATASET_PATH}`.\n\n"
                    f"Add it to your repo at: `data/{DEFAULT_DATASET_FILENAME}`"
                )
                st.stop()

    # Visualization
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
            sns.histplot(df[viz_target], kde=True, ax=ax)
            ax.set_title(f"Distribution of {viz_target}")
        else:
            sns.boxplot(x=df[viz_target], ax=ax)
            ax.set_title(f"Spread of {viz_target}")
        st.pyplot(fig)

    st.markdown('<div class="sub-header">3. Correlation Analysis</div>', unsafe_allow_html=True)
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        fig_corr, ax_corr = plt.subplots(figsize=(10, 5))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
        st.pyplot(fig_corr)
    else:
        st.info("No numeric columns detected for correlation analysis.")

# ---------------------------------------------------------------------
# MODULE 2: MODEL TRAINING ENGINE
# ---------------------------------------------------------------------
elif mode == "2. Model Training Engine":
    st.markdown('<div class="sub-header">Model Configuration</div>', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("Please upload data in Module 1 first.")
        st.stop()

    df = st.session_state.df.copy()

    c1, c2 = st.columns(2)
    with c1:
        target_col = st.selectbox("Target Variable (Adoption Outcome)", df.columns, index=len(df.columns) - 1)
    with c2:
        candidates = [c for c in df.columns if c != target_col and c != "Respondent_ID"]
        feature_cols = st.multiselect("Predictor Attributes (Features)", candidates, default=candidates)

    if not feature_cols:
        st.warning("Please select at least one feature column.")
        st.stop()

    task_type = determine_task_type(df, target_col)

    st.markdown(f"""
    <div class='insight-box'>
        <b>System Intelligence:</b> Based on the structure of <code>{target_col}</code>, 
        the system has selected a <b>{task_type}</b> strategy.
    </div>
    """, unsafe_allow_html=True)

    col_algo, col_hyper = st.columns(2)
    with col_algo:
        algorithm = st.selectbox("Machine Learning Algorithm", ["Random Forest (Bagging)", "XGBoost (Boosting)"])
    with col_hyper:
        n_estimators = st.slider("Model Complexity (Trees)", 50, 500, 100)

    if st.button("Train Model", type="primary"):
        with st.spinner("Training Ensemble Model..."):
            try:
                X = df[feature_cols]
                y = df[target_col]

                X_encoded = pd.get_dummies(X, drop_first=True)

                le = None
                if task_type == "Classification":
                    le = LabelEncoder()
                    y = le.fit_transform(y)

                X_train, X_test, y_train, y_test = train_test_split(
                    X_encoded, y, test_size=0.2, random_state=42
                )

                if task_type == "Classification":
                    if "Random Forest" in algorithm:
                        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                    else:
                        model = xgb.XGBClassifier(n_estimators=n_estimators, random_state=42)
                else:
                    if "Random Forest" in algorithm:
                        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                    else:
                        model = xgb.XGBRegressor(n_estimators=n_estimators, random_state=42)

                model.fit(X_train, y_train)

                st.session_state.model = model
                st.session_state.model_meta = {
                    "target_col": target_col,
                    "feature_cols": feature_cols,
                    "task_type": task_type,
                    "label_encoder": le,
                    "train_columns": X_encoded.columns.tolist(),
                    "algorithm": algorithm
                }

                st.success(f"Successfully trained {algorithm} on {len(X_train)} samples.")

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

                st.markdown('<div class="sub-header">Attribute Importance (Permutation)</div>', unsafe_allow_html=True)
                st.caption("Which attributes are driving the adoption decision?")

                result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
                sorted_idx = result.importances_mean.argsort()

                fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
                ax_imp.barh(X_test.columns[sorted_idx], result.importances_mean[sorted_idx])
                ax_imp.set_xlabel("Impact on Model Prediction")
                st.pyplot(fig_imp)

            except Exception as e:
                st.error(f"Training Failed: {e}")

# ---------------------------------------------------------------------
# MODULE 3: PREDICTION PLAYGROUND
# ---------------------------------------------------------------------
elif mode == "3. Prediction Playground":
    st.markdown('<div class="sub-header">Simulation Interface</div>', unsafe_allow_html=True)

    if st.session_state.model is None:
        st.warning("No active model. Please train a model in Module 2 first.")
        st.stop()

    model = st.session_state.model
    meta = st.session_state.model_meta
    df = st.session_state.df

    st.info(f"Using **{meta['algorithm']}** to predict **{meta['target_col']}**")

    with st.form("simulation_form"):
        st.markdown("**Configure Innovation Attributes**")
        input_data = {}

        cols = st.columns(2)
        for idx, col_name in enumerate(meta["feature_cols"]):
            active_col = cols[idx % 2]

            if pd.api.types.is_numeric_dtype(df[col_name]):
                min_val = float(df[col_name].min())
                max_val = float(df[col_name].max())
                default_val = float(df[col_name].median())
                step = 1.0 if pd.api.types.is_integer_dtype(df[col_name]) else 0.1

                input_data[col_name] = active_col.slider(
                    f"{col_name}", min_val, max_val, default_val, step=step
                )
            else:
                options = df[col_name].dropna().unique().tolist()
                input_data[col_name] = active_col.selectbox(f"{col_name}", options)

        submitted = st.form_submit_button("Simulate Adoption Outcome")

    if submitted:
        input_df = pd.DataFrame([input_data])
        input_encoded = pd.get_dummies(input_df)
        final_input = input_encoded.reindex(columns=meta["train_columns"], fill_value=0)

        prediction = model.predict(final_input)

        st.divider()
        st.markdown("### Simulation Results")

        c_res1, c_res2 = st.columns([1, 2])

        with c_res1:
            if meta["task_type"] == "Classification":
                le = meta["label_encoder"]
                res_label = le.inverse_transform([prediction[0]])[0] if le else prediction[0]
                st.metric("Predicted Status", f"{res_label}")
            else:
                st.metric("Predicted Score", f"{prediction[0]:.2f}")

        with c_res2:
            if meta["task_type"] == "Classification" and hasattr(model, "predict_proba"):
                probs = model.predict_proba(final_input)
                le = meta["label_encoder"]
                class_names = le.classes_ if le else model.classes_
                prob_df = pd.DataFrame(probs, columns=class_names)
                st.bar_chart(prob_df.T)
                st.caption("Probability distribution across adopter categories.")
