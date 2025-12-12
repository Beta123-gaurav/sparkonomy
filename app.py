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

# -----------------------------------------------------------------------------
# DEFAULT DATASET (ROOT-LEVEL FILE — MATCHES YOUR REPO)
# -----------------------------------------------------------------------------
DEFAULT_DATASET_FILENAME = "sparkonomy_synthetic_survey_30.xlsx"

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Rogers' Diffusion Analytics",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CUSTOM STYLING
# -----------------------------------------------------------------------------
st.markdown("""
<style>
.main-header {font-size:2.2rem;font-weight:700;color:#1E3A8A;}
.sub-header {font-size:1.3rem;font-weight:600;color:#334155;margin-top:1.5rem;border-bottom:2px solid #e2e8f0;}
.insight-box {background:#eff6ff;border-left:5px solid #2563eb;padding:1rem;border-radius:4px;}
.stButton>button {width:100%;}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------------------------
st.session_state.setdefault("df", None)
st.session_state.setdefault("model", None)
st.session_state.setdefault("model_meta", {})

# -----------------------------------------------------------------------------
# UTILITY FUNCTIONS
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

def load_default_data():
    if os.path.exists(DEFAULT_DATASET_FILENAME):
        return pd.read_excel(DEFAULT_DATASET_FILENAME)
    return None

def determine_task_type(df, target_col):
    col = df[target_col]
    if pd.api.types.is_numeric_dtype(col) and col.nunique() > 15:
        return "Regression"
    return "Classification"

# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">Engineering Adoption: Rogers\' Framework</div>', unsafe_allow_html=True)
st.caption("A Computational Framework for Sociometry and Data Engineering")

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Navigation")
    mode = st.radio(
        "Select Module",
        ["1. Data Ingestion & EDA", "2. Model Training Engine", "3. Prediction Playground"]
    )
    st.divider()
    st.info("""
Rogers' Attributes:
• Relative Advantage
• Compatibility
• Complexity
• Trialability
• Observability
""")

# -----------------------------------------------------------------------------
# MODULE 1 — DATA INGESTION & EDA
# -----------------------------------------------------------------------------
if mode == "1. Data Ingestion & EDA":
    st.markdown('<div class="sub-header">Data Ingestion</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

    if uploaded_file:
        st.session_state.df = load_data(uploaded_file)
        st.success("Dataset uploaded successfully")
    elif st.session_state.df is None:
        df_default = load_default_data()
        if df_default is not None:
            st.session_state.df = df_default
            st.info("Loaded default dataset from repository")
        else:
            st.error("No dataset found. Upload a file or commit sparkonomy_synthetic_survey_30.xlsx")
            st.stop()

    df = st.session_state.df
    st.dataframe(df, use_container_width=True)

    st.markdown('<div class="sub-header">Exploratory Analysis</div>', unsafe_allow_html=True)
    col = st.selectbox("Select Attribute", df.columns)

    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax)
    st.pyplot(fig)

# -----------------------------------------------------------------------------
# MODULE 2 — MODEL TRAINING
# -----------------------------------------------------------------------------
elif mode == "2. Model Training Engine":
    if st.session_state.df is None:
        st.warning("Upload data first")
        st.stop()

    df = st.session_state.df.copy()

    target = st.selectbox("Target Variable", df.columns, index=len(df.columns) - 1)
    features = st.multiselect(
        "Feature Columns",
        [c for c in df.columns if c != target],
        default=[c for c in df.columns if c != target]
    )

    task = determine_task_type(df, target)
    algo = st.selectbox("Algorithm", ["Random Forest", "XGBoost"])
    n_estimators = st.slider("Trees", 50, 500, 100)

    if st.button("Train Model"):
        X = pd.get_dummies(df[features], drop_first=True)
        y = df[target]

        le = None
        if task == "Classification":
            le = LabelEncoder()
            y = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        if task == "Classification":
            model = (
                RandomForestClassifier(n_estimators=n_estimators)
                if algo == "Random Forest"
                else xgb.XGBClassifier(n_estimators=n_estimators)
            )
        else:
            model = (
                RandomForestRegressor(n_estimators=n_estimators)
                if algo == "Random Forest"
                else xgb.XGBRegressor(n_estimators=n_estimators)
            )

        model.fit(X_train, y_train)

        st.session_state.model = model
        st.session_state.model_meta = {
            "task": task,
            "features": features,
            "columns": X.columns.tolist(),
            "label_encoder": le
        }

        st.success("Model trained successfully")

# -----------------------------------------------------------------------------
# MODULE 3 — PREDICTION PLAYGROUND
# -----------------------------------------------------------------------------
elif mode == "3. Prediction Playground":
    if st.session_state.model is None:
        st.warning("Train a model first")
        st.stop()

    model = st.session_state.model
    meta = st.session_state.model_meta
    df = st.session_state.df

    st.subheader("Simulate Adoption Scenario")
    inputs = {}

    for col in meta["features"]:
        if pd.api.types.is_numeric_dtype(df[col]):
            inputs[col] = st.slider(
                col,
                float(df[col].min()),
                float(df[col].max()),
                float(df[col].median())
            )
        else:
            inputs[col] = st.selectbox(col, df[col].unique())

    if st.button("Predict"):
        input_df = pd.DataFrame([inputs])
        input_enc = pd.get_dummies(input_df)
        final = input_enc.reindex(columns=meta["columns"], fill_value=0)
        pred = model.predict(final)

        if meta["task"] == "Classification":
            le = meta["label_encoder"]
            result = le.inverse_transform(pred)[0] if le else pred[0]
            st.metric("Predicted Class", result)
        else:
            st.metric("Predicted Value", f"{pred[0]:.2f}")
