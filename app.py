# app.py
# Engineering Adoption: Rogers' Diffusion Analytics Platform
# Production-grade Streamlit dashboard with descriptive statistics + explainable EDA + ML training + simulation

import os
import math
import warnings

import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------
# Optional imports (Zero-crash)
# ---------------------------
# If dependencies aren't installed correctly on cloud, the app will show a clear message instead of crashing.
PLOTTING_OK = True
ML_OK = True

try:
    import matplotlib.pyplot as plt
except Exception:
    PLOTTING_OK = False
    plt = None

try:
    import seaborn as sns
except Exception:
    PLOTTING_OK = False
    sns = None

try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, classification_report
    from sklearn.inspection import permutation_importance
except Exception:
    ML_OK = False
    train_test_split = None
    RandomForestClassifier = RandomForestRegressor = None
    LabelEncoder = None
    accuracy_score = r2_score = mean_squared_error = classification_report = None
    permutation_importance = None

try:
    import xgboost as xgb
except Exception:
    ML_OK = False
    xgb = None

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
DEFAULT_DATASET_FILENAME = "sparkonomy_synthetic_survey_30.xlsx"  # repo root (matches your screenshot)
APP_VERSION = "1.1.0"

st.set_page_config(
    page_title="Rogers' Diffusion Analytics",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# STYLING
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .main-header {
            font-size: 2.2rem;
            color: #1E3A8A;
            font-weight: 800;
            margin-bottom: 0.3rem;
        }
        .sub-header {
            font-size: 1.25rem;
            color: #0f172a;
            font-weight: 700;
            margin-top: 1.25rem;
            margin-bottom: 0.75rem;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 0.4rem;
        }
        .pill {
            display:inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            color: #1e3a8a;
            font-size: 0.85rem;
            font-weight: 600;
            margin-right: 0.35rem;
        }
        .insight-box {
            background: #f8fafc;
            border-left: 5px solid #2563eb;
            padding: 0.9rem 1rem;
            border-radius: 10px;
            color: #0f172a;
            margin: 0.4rem 0 0.8rem 0;
        }
        .warn-box {
            background: #fff7ed;
            border-left: 5px solid #fb923c;
            padding: 0.9rem 1rem;
            border-radius: 10px;
            color: #7c2d12;
            margin: 0.4rem 0 0.8rem 0;
        }
        .stButton>button { width: 100%; }
        code { font-size: 0.95em; }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------------------------
st.session_state.setdefault("df", None)
st.session_state.setdefault("model", None)
st.session_state.setdefault("model_meta", {})
st.session_state.setdefault("last_train_report", "")

# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------
def _safe_stop_with_message(msg: str):
    st.markdown(f"<div class='warn-box'><b>Action needed:</b> {msg}</div>", unsafe_allow_html=True)
    st.stop()


@st.cache_data(show_spinner=False)
def load_uploaded(file) -> pd.DataFrame:
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    if file.name.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(file)
    raise ValueError("Unsupported file type. Please upload .csv or .xlsx")


@st.cache_data(show_spinner=False)
def load_default_from_repo() -> pd.DataFrame | None:
    if os.path.exists(DEFAULT_DATASET_FILENAME):
        return pd.read_excel(DEFAULT_DATASET_FILENAME)
    return None


def determine_task_type(df: pd.DataFrame, target_col: str) -> str:
    """
    Heuristic:
    - If numeric and many unique values -> Regression
    - Else -> Classification
    """
    col = df[target_col]
    if pd.api.types.is_numeric_dtype(col):
        return "Regression" if col.nunique(dropna=True) > 15 else "Classification"
    return "Classification"


def describe_numeric(numeric_df: pd.DataFrame) -> pd.DataFrame:
    desc = numeric_df.describe().T
    desc["median"] = numeric_df.median(numeric_only=True)
    desc["skewness"] = numeric_df.skew(numeric_only=True)
    desc["kurtosis"] = numeric_df.kurtosis(numeric_only=True)
    desc["missing_%"] = numeric_df.isna().mean() * 100
    # reorder
    preferred = ["count", "mean", "std", "min", "25%", "50%", "75%", "max", "median", "skewness", "kurtosis", "missing_%"]
    cols = [c for c in preferred if c in desc.columns] + [c for c in desc.columns if c not in preferred]
    return desc[cols]


def corr_strength_label(abs_r: float) -> str:
    if abs_r >= 0.70:
        return "Strong"
    if abs_r >= 0.40:
        return "Moderate"
    return "Weak"


def iqr_outlier_share(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return 0.0
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return float(((s < low) | (s > high)).mean() * 100.0)


def interpret_distribution(series: pd.Series) -> str:
    s = series.dropna()
    if s.empty:
        return "No data available for interpretation."
    mean = float(s.mean())
    med = float(s.median())
    std = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    skew = float(s.skew()) if len(s) > 2 else 0.0
    out_share = iqr_outlier_share(s)

    skew_txt = (
        "approximately symmetric"
        if -0.5 <= skew <= 0.5
        else "right-skewed (more low values, few high values)"
        if skew > 0.5
        else "left-skewed (more high values, few low values)"
    )

    variability_txt = (
        "low variation (responses are consistent)"
        if std <= 0.75
        else "moderate variation (some disagreement across respondents)"
        if std <= 1.5
        else "high variation (responses are widely dispersed)"
    )

    return (
        f"Mean **{mean:.2f}** vs Median **{med:.2f}** suggests the distribution is **{skew_txt}**. "
        f"Standard deviation **{std:.2f}** indicates **{variability_txt}**. "
        f"Estimated outliers (IQR rule): **{out_share:.1f}%** of observations."
    )


def top_correlations(corr: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    # Flatten abs correlations, remove self-pairs and duplicates
    c = corr.copy()
    np.fill_diagonal(c.values, np.nan)
    long = (
        c.abs()
        .stack()
        .reset_index()
        .rename(columns={"level_0": "var_1", "level_1": "var_2", 0: "abs_corr"})
    )
    # Remove duplicates by sorting pair names
    long["pair"] = long.apply(lambda r: " | ".join(sorted([str(r["var_1"]), str(r["var_2"])])), axis=1)
    long = long.drop_duplicates("pair").drop(columns=["pair"])
    long["strength"] = long["abs_corr"].apply(corr_strength_label)
    long = long.sort_values("abs_corr", ascending=False).head(top_n)
    return long


def ensure_dependencies():
    if not PLOTTING_OK:
        _safe_stop_with_message(
            "Plotting libraries are missing. Ensure your `requirements.txt` is valid and includes "
            "`matplotlib` and `seaborn`, then **reboot** the Streamlit app."
        )
    if not ML_OK:
        st.markdown(
            "<div class='warn-box'><b>Note:</b> ML libraries are not available. "
            "EDA will work, but Model Training/Prediction will be disabled until dependencies install.</div>",
            unsafe_allow_html=True
        )


# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">Engineering Adoption: Rogers\' Diffusion Analytics</div>', unsafe_allow_html=True)
st.write(
    f"<span class='pill'>v{APP_VERSION}</span>"
    "<span class='pill'>Descriptive Analytics</span>"
    "<span class='pill'>Ensemble ML</span>"
    "<span class='pill'>Simulation Playground</span>",
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Navigation")
    mode = st.radio(
        "Select Module",
        ["1. Data Ingestion & EDA", "2. Model Training Engine", "3. Prediction Playground"],
        index=0
    )
    st.divider()
    st.caption("Rogers' Innovation Attributes")
    st.write("1) Relative Advantage\n\n2) Compatibility\n\n3) Complexity\n\n4) Trialability\n\n5) Observability")
    st.divider()
    st.caption("Default dataset (repo root)")
    st.code(DEFAULT_DATASET_FILENAME)

# -----------------------------------------------------------------------------
# MODULE 1: DATA INGESTION & EDA (UPGRADED)
# -----------------------------------------------------------------------------
if mode == "1. Data Ingestion & EDA":
    ensure_dependencies()

    st.markdown('<div class="sub-header">1) Data Ingestion</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Dataset (.csv or .xlsx)", type=["csv", "xlsx"])

    if uploaded is not None:
        try:
            with st.spinner("Loading uploaded dataset..."):
                st.session_state.df = load_uploaded(uploaded)
            st.success("Dataset uploaded and loaded successfully.")
        except Exception as e:
            _safe_stop_with_message(f"Failed to load uploaded file. Error: {e}")
    elif st.session_state.df is None:
        # auto-load default if present
        df_default = load_default_from_repo()
        if df_default is not None:
            st.session_state.df = df_default
            st.info(f"Loaded default dataset from repo: `{DEFAULT_DATASET_FILENAME}`")
        else:
            _safe_stop_with_message(
                f"No dataset loaded. Upload a file OR commit `{DEFAULT_DATASET_FILENAME}` to the repo root."
            )

    df = st.session_state.df
    st.success(f"Data loaded: **{df.shape[0]}** rows × **{df.shape[1]}** columns")

    with st.expander("View raw data", expanded=False):
        st.dataframe(df, use_container_width=True)

    # Data quality panel
    st.markdown('<div class="sub-header">2) Data Quality Snapshot</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isna().sum().sum())
    missing_pct = (missing_cells / total_cells * 100) if total_cells else 0

    dup_rows = int(df.duplicated().sum())
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    c1.metric("Rows", f"{df.shape[0]}")
    c2.metric("Columns", f"{df.shape[1]}")
    c3.metric("Missing cells", f"{missing_cells} ({missing_pct:.1f}%)")
    c4.metric("Duplicate rows", f"{dup_rows}")

    with st.expander("Column types & missingness", expanded=False):
        types_df = pd.DataFrame({
            "column": df.columns,
            "dtype": [str(df[c].dtype) for c in df.columns],
            "missing_%": (df.isna().mean() * 100).round(2).values,
            "unique": [df[c].nunique(dropna=True) for c in df.columns]
        }).sort_values("missing_%", ascending=False)
        st.dataframe(types_df, use_container_width=True)

    # Descriptive stats
    st.markdown('<div class="sub-header">3) Descriptive Statistics (Numeric)</div>', unsafe_allow_html=True)

    if not numeric_cols:
        _safe_stop_with_message("No numeric columns detected. Add numeric attributes (e.g., 1–10 scales) and retry.")

    numeric_df = df[numeric_cols].copy()
    desc_df = describe_numeric(numeric_df).round(3)
    st.dataframe(desc_df, use_container_width=True)

    st.markdown(
        "<div class='insight-box'>"
        "<b>How to read this:</b> "
        "Use <code>mean/median</code> to understand central tendency, "
        "<code>std</code> for disagreement/dispersion, "
        "<code>skewness</code> for bias toward low/high responses, "
        "and <code>missing_%</code> to assess data quality risk."
        "</div>",
        unsafe_allow_html=True
    )

    # Attribute deep-dive
    st.markdown('<div class="sub-header">4) Attribute Deep-Dive</div>', unsafe_allow_html=True)

    attr = st.selectbox("Select a numeric attribute", numeric_cols)
    left, right = st.columns([2.2, 1])

    with left:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(numeric_df[attr], kde=True, ax=ax)
        ax.set_title(f"Distribution: {attr}")
        ax.set_xlabel(attr)
        ax.set_ylabel("Count")
        st.pyplot(fig, use_container_width=True)

        fig2, ax2 = plt.subplots(figsize=(8, 1.6))
        sns.boxplot(x=numeric_df[attr], ax=ax2)
        ax2.set_title(f"Spread / Outliers: {attr}")
        st.pyplot(fig2, use_container_width=True)

    with right:
        s = numeric_df[attr]
        st.metric("Mean", f"{s.mean():.2f}")
        st.metric("Median", f"{s.median():.2f}")
        st.metric("Std Dev", f"{s.std(ddof=1):.2f}" if s.count() > 1 else "0.00")
        st.metric("Skewness", f"{s.skew():.2f}" if s.count() > 2 else "0.00")
        st.metric("Outliers (IQR %)", f"{iqr_outlier_share(s):.1f}%")

    st.markdown(f"<div class='insight-box'>{interpret_distribution(numeric_df[attr])}</div>", unsafe_allow_html=True)

    # Correlation analysis with interpretation
    st.markdown('<div class="sub-header">5) Correlation & Relationship Strength</div>', unsafe_allow_html=True)

    corr = numeric_df.corr(numeric_only=True)
    figc, axc = plt.subplots(figsize=(10, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=axc)
    axc.set_title("Correlation Matrix (Numeric Attributes)")
    st.pyplot(figc, use_container_width=True)

    st.markdown("**Top relationships (absolute correlation)**")
    top_corr = top_correlations(corr, top_n=10)
    st.dataframe(top_corr, use_container_width=True)

    st.markdown(
        "<div class='insight-box'>"
        "<b>Interpretation guide:</b> "
        "<code>Weak</code> (&lt;0.40) suggests limited linear relationship, "
        "<code>Moderate</code> (0.40–0.69) suggests meaningful association, "
        "<code>Strong</code> (≥0.70) suggests high redundancy or tight coupling."
        "</div>",
        unsafe_allow_html=True
    )

    # Outcome-based summaries (if user chooses an outcome/target)
    st.markdown('<div class="sub-header">6) Outcome-Based Descriptives (Optional)</div>', unsafe_allow_html=True)

    outcome_col = st.selectbox(
        "Select an outcome/target column to compare groups (optional)",
        ["(Skip)"] + df.columns.tolist(),
        index=0
    )

    if outcome_col != "(Skip)":
        # If outcome is numeric, allow binning; if categorical, group directly
        if pd.api.types.is_numeric_dtype(df[outcome_col]):
            bins = st.slider("Bin outcome into quantiles", 2, 6, 3)
            try:
                grp = pd.qcut(df[outcome_col], q=bins, duplicates="drop")
                grouped = df.assign(_outcome_bin=grp).groupby("_outcome_bin")[numeric_cols].mean(numeric_only=True).round(3)
                st.write("**Mean attribute scores by outcome quantile**")
                st.dataframe(grouped, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not bin outcome column: {e}")
        else:
            # categorical outcome
            grp_counts = df[outcome_col].value_counts(dropna=False).reset_index()
            grp_counts.columns = [outcome_col, "count"]
            st.write("**Outcome distribution**")
            st.dataframe(grp_counts, use_container_width=True)

            st.write("**Mean attribute scores by outcome class**")
            grouped = df.groupby(outcome_col)[numeric_cols].mean(numeric_only=True).round(3)
            st.dataframe(grouped, use_container_width=True)

            # Optional: boxplot by class for a chosen attribute
            by_attr = st.selectbox("Boxplot attribute by outcome class", numeric_cols)
            figb, axb = plt.subplots(figsize=(10, 4))
            sns.boxplot(data=df, x=outcome_col, y=by_attr, ax=axb)
            axb.set_title(f"{by_attr} by {outcome_col}")
            axb.set_xlabel(outcome_col)
            axb.set_ylabel(by_attr)
            st.pyplot(figb, use_container_width=True)

# -----------------------------------------------------------------------------
# MODULE 2: MODEL TRAINING ENGINE
# -----------------------------------------------------------------------------
elif mode == "2. Model Training Engine":
    if st.session_state.df is None:
        _safe_stop_with_message("Upload or load data first in Module 1.")

    if not ML_OK:
        _safe_stop_with_message(
            "ML dependencies are not installed. Fix `requirements.txt` then reboot the app. "
            "Needed: scikit-learn, xgboost."
        )

    df = st.session_state.df.copy()

    st.markdown('<div class="sub-header">1) Model Setup</div>', unsafe_allow_html=True)

    # target selection
    c1, c2 = st.columns(2)
    with c1:
        target_col = st.selectbox("Target Variable (Outcome)", df.columns, index=len(df.columns) - 1)
    with c2:
        # exclude common ID fields automatically if present
        excluded = {"Respondent_ID", "respondent_id", "id", "ID"}
        candidates = [c for c in df.columns if c != target_col and c not in excluded]
        feature_cols = st.multiselect("Predictor Attributes (Features)", candidates, default=candidates)

    if not feature_cols:
        _safe_stop_with_message("Select at least one feature column to train a model.")

    # determine task
    task_type = determine_task_type(df, target_col)

    st.markdown(
        f"<div class='insight-box'><b>System choice:</b> Target <code>{target_col}</code> "
        f"→ <b>{task_type}</b> modeling strategy.</div>",
        unsafe_allow_html=True
    )

    c3, c4 = st.columns(2)
    with c3:
        algorithm = st.selectbox("Algorithm", ["Random Forest", "XGBoost"])
    with c4:
        n_estimators = st.slider("Trees (Model Complexity)", 50, 500, 200, step=10)

    test_size = st.slider("Test split (%)", 10, 40, 20, step=5) / 100.0

    st.markdown('<div class="sub-header">2) Train</div>', unsafe_allow_html=True)

    if st.button("Train Model", type="primary"):
        with st.spinner("Training model..."):
            try:
                X = df[feature_cols]
                y = df[target_col]

                # one-hot encode X
                X_encoded = pd.get_dummies(X, drop_first=True)

                # label encode y if classification
                le = None
                if task_type == "Classification":
                    le = LabelEncoder()
                    y = le.fit_transform(y.astype(str))

                X_train, X_test, y_train, y_test = train_test_split(
                    X_encoded, y, test_size=test_size, random_state=42
                )

                if task_type == "Classification":
                    if algorithm == "Random Forest":
                        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                    else:
                        model = xgb.XGBClassifier(
                            n_estimators=n_estimators,
                            random_state=42,
                            eval_metric="logloss"
                        )
                else:
                    if algorithm == "Random Forest":
                        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                    else:
                        model = xgb.XGBRegressor(n_estimators=n_estimators, random_state=42)

                model.fit(X_train, y_train)

                # save
                st.session_state.model = model
                st.session_state.model_meta = {
                    "target_col": target_col,
                    "feature_cols": feature_cols,
                    "task_type": task_type,
                    "label_encoder": le,
                    "train_columns": X_encoded.columns.tolist(),
                    "algorithm": algorithm,
                    "test_size": test_size
                }

                st.success(f"Trained **{algorithm}** on **{len(X_train)}** samples (test size: {int(test_size*100)}%).")

                # evaluation
                st.markdown('<div class="sub-header">3) Performance</div>', unsafe_allow_html=True)

                y_pred = model.predict(X_test)

                a, b = st.columns(2)
                if task_type == "Classification":
                    acc = accuracy_score(y_test, y_pred)
                    a.metric("Accuracy", f"{acc:.1%}")

                    with st.expander("Classification report"):
                        target_names = le.classes_.tolist() if le else None
                        rep = classification_report(y_test, y_pred, target_names=target_names)
                        st.session_state.last_train_report = rep
                        st.code(rep)
                else:
                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    a.metric("R² (variance explained)", f"{r2:.4f}")
                    b.metric("MSE", f"{mse:.4f}")

                # permutation importance (explainability)
                st.markdown('<div class="sub-header">4) Explainability: Permutation Importance</div>', unsafe_allow_html=True)
                st.caption("Identifies which attributes most influence the model’s predictions.")

                try:
                    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
                    sorted_idx = np.argsort(result.importances_mean)

                    fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
                    ax_imp.barh(X_test.columns[sorted_idx], result.importances_mean[sorted_idx])
                    ax_imp.set_xlabel("Mean importance (drop in performance when permuted)")
                    ax_imp.set_title("Permutation Feature Importance")
                    st.pyplot(fig_imp, use_container_width=True)

                    # Top 8 table
                    imp_df = pd.DataFrame({
                        "feature": X_test.columns,
                        "importance_mean": result.importances_mean,
                        "importance_std": result.importances_std
                    }).sort_values("importance_mean", ascending=False).head(8)
                    st.dataframe(imp_df.round(6), use_container_width=True)

                except Exception as e:
                    st.warning(f"Could not compute permutation importance for this model/data: {e}")

            except Exception as e:
                _safe_stop_with_message(f"Training failed. Error: {e}")

# -----------------------------------------------------------------------------
# MODULE 3: PREDICTION PLAYGROUND
# -----------------------------------------------------------------------------
elif mode == "3. Prediction Playground":
    if st.session_state.df is None:
        _safe_stop_with_message("Load data first in Module 1.")

    if st.session_state.model is None:
        _safe_stop_with_message("Train a model first in Module 2.")

    if not ML_OK:
        _safe_stop_with_message("ML dependencies are missing. Fix requirements and reboot.")

    df = st.session_state.df
    model = st.session_state.model
    meta = st.session_state.model_meta

    st.markdown('<div class="sub-header">Prediction Playground</div>', unsafe_allow_html=True)
    st.markdown(
        f"<div class='insight-box'>"
        f"<b>Active model:</b> {meta.get('algorithm','(unknown)')} "
        f"→ Predicting <code>{meta.get('target_col','(target)')}</code> "
        f"as <b>{meta.get('task_type','(task)')}</b>."
        f"</div>",
        unsafe_allow_html=True
    )

    # Dynamic input form
    with st.form("simulation_form"):
        st.write("Set attribute values to simulate an adoption outcome.")
        inputs = {}

        cols = st.columns(2)
        for i, col_name in enumerate(meta["feature_cols"]):
            ui = cols[i % 2]
            if pd.api.types.is_numeric_dtype(df[col_name]):
                s = df[col_name].dropna()
                min_val = float(s.min()) if not s.empty else 0.0
                max_val = float(s.max()) if not s.empty else 10.0
                default_val = float(s.median()) if not s.empty else (min_val + max_val) / 2
                step = 1.0 if pd.api.types.is_integer_dtype(df[col_name]) else 0.1
                inputs[col_name] = ui.slider(col_name, min_val, max_val, default_val, step=step)
            else:
                options = df[col_name].dropna().astype(str).unique().tolist()
                if not options:
                    options = ["(no categories)"]
                inputs[col_name] = ui.selectbox(col_name, options)

        run = st.form_submit_button("Simulate Outcome")

    if run:
        try:
            input_df = pd.DataFrame([inputs])

            # Align encoding to training schema
            input_enc = pd.get_dummies(input_df, drop_first=True)
            final = input_enc.reindex(columns=meta["train_columns"], fill_value=0)

            pred = model.predict(final)

            st.divider()
            st.markdown("### Results")

            c1, c2 = st.columns([1, 2])

            with c1:
                if meta["task_type"] == "Classification":
                    le = meta.get("label_encoder")
                    label = le.inverse_transform([pred[0]])[0] if le else pred[0]
                    st.metric("Predicted Class", str(label))
                else:
                    st.metric("Predicted Value", f"{float(pred[0]):.2f}")

            with c2:
                if meta["task_type"] == "Classification" and hasattr(model, "predict_proba"):
                    probs = model.predict_proba(final)
                    le = meta.get("label_encoder")
                    class_names = le.classes_.tolist() if le else getattr(model, "classes_", [])
                    if len(class_names) == probs.shape[1]:
                        prob_df = pd.DataFrame(probs, columns=class_names).T
                        st.bar_chart(prob_df)
                        st.caption("Probability distribution across outcome classes.")
                    else:
                        st.info("Probability chart unavailable (class labels mismatch).")

            # Explain what changed
            with st.expander("Show inputs used for this simulation"):
                st.dataframe(input_df, use_container_width=True)

        except Exception as e:
            _safe_stop_with_message(f"Prediction failed. Error: {e}")
