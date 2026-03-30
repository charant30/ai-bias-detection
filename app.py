import streamlit as st
import pandas as pd
import streamlit_antd_components as sac
import ml_utils

st.set_page_config(page_title="Bias Detection Dashboard", page_icon="⚖️", layout="wide")


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def load_data(source):
    try:
        return pd.read_csv(source)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None


def sidebar_hyperparams(model_type):
    """Render dynamic hyperparameter sliders for the selected model and return a dict."""
    hp = {}
    st.sidebar.markdown("**Hyperparameters**")
    if model_type == "Logistic Regression":
        hp["C"] = st.sidebar.select_slider(
            "Regularization C", options=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], value=1.0,
            help="Higher C = less regularization (more complex model)")
    elif model_type == "Decision Tree":
        hp["max_depth"] = st.sidebar.slider("Max Depth", 1, 30, 8)
    elif model_type == "Random Forest":
        hp["n_estimators"] = st.sidebar.slider("Trees", 10, 500, 100, step=10)
        hp["max_depth"] = st.sidebar.slider("Max Depth (0 = None)", 0, 30, 0)
        if hp["max_depth"] == 0: hp["max_depth"] = None
    elif model_type in ("Gradient Boosting", "XGBoost", "LightGBM"):
        hp["n_estimators"] = st.sidebar.slider("Estimators", 10, 500, 100, step=10)
        hp["learning_rate"] = st.sidebar.select_slider(
            "Learning Rate", options=[0.001, 0.01, 0.05, 0.1, 0.2, 0.5], value=0.1)
        hp["max_depth"] = st.sidebar.slider("Max Depth", 1, 15, 3)
        if model_type == "LightGBM":
            hp["num_leaves"] = st.sidebar.slider("Num Leaves", 8, 256, 31)
    elif model_type == "SVM":
        hp["C"] = st.sidebar.select_slider(
            "Regularization C", options=[0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)
        hp["kernel"] = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly"])
    elif model_type == "KNN":
        hp["n_neighbors"] = st.sidebar.slider("Neighbours (k)", 1, 50, 5)
        hp["weights"] = st.sidebar.selectbox("Weights", ["uniform", "distance"])
    return hp


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR ABOUT SECTION
# ════════════════════════════════════════════════════════════════════════════

def render_about():
    with st.sidebar.expander("ℹ️ About this Dashboard", expanded=False):
        st.markdown("""
### What is this?
This dashboard helps you **detect and visualize bias** in ML models and apply **mitigation strategies**.

---
### How to use — Step by Step
1. **Choose a dataset** — demo or your own CSV
2. **Select Target** — what the model predicts (e.g. `Loan_Approved`)
3. **Select Sensitive Attribute** — group to analyse (e.g. `Gender`)
4. **Pick an algorithm + tune hyperparameters**
5. **Set training options** — cross-validation, imbalance handling
6. **Optionally enable Bias Mitigation** and choose a strategy
7. **Choose Run Mode** — Single Model or Compare Multiple
8. **Click 🚀 Run**

---
### Metrics Glossary

| Metric | Meaning |
|--------|---------|
| **Accuracy** | % correct predictions |
| **Precision** | Correct positives / all predicted positives |
| **Recall** | Correct positives / all actual positives |
| **F1-Score** | Harmonic mean of Precision & Recall |
| **FPR** | False Positive Rate — wrong positive alerts |
| **FNR** | False Negative Rate — missed positives |
| **AUC** | Area under ROC — 1.0 perfect, 0.5 random |

---
### Bias Mitigation Strategies

| Strategy | How it works |
|----------|-------------|
| **Reweighting** | Up-weights underrepresented groups in training |
| **SMOTE** | Oversamples minority class via synthetic examples |
| **Threshold Adjustment** | Sets group-specific decision thresholds to equalise FPR |
| **FairLearn – Demographic Parity** | Constrains model so groups get equal positive rates |
| **FairLearn – Equalized Odds** | Constrains model to equalise TPR and FPR across groups |

---
### Tips
- Bias is flagged when any metric gap between groups **exceeds 10%**
- Use *Compare Mode* to find the **fairest algorithm** on your data
- Check Before vs After when mitigation is enabled
""")
    st.sidebar.divider()


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    st.title("⚖️ Bias Detection Dashboard")
    st.markdown("Train ML models and evaluate potential **bias** or unfairness across demographic groups.")
    st.divider()

    render_about()

    # ── 1. Data Source ───────────────────────────────────────────────────
    st.sidebar.title("1. Data Source")
    datasource = st.sidebar.radio("Choose Input:", ["Demo Dataset", "Upload CSV"])
    df = None
    if datasource == "Demo Dataset":
        df = load_data("demo_dataset.csv")
        st.info("ℹ️ Using the synthetic demo dataset with intentional gender bias in loan approvals.")
    else:
        uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            df = load_data(uploaded)

    if df is None:
        st.info("Upload a CSV or select the demo dataset from the sidebar.")
        return

    # ── Data Preview ─────────────────────────────────────────────────────
    st.subheader("Data Preview")
    st.dataframe(df.head(), use_container_width=True)
    st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    st.divider()

    # ── 2. Target & Sensitive ────────────────────────────────────────────
    st.sidebar.title("2. Column Selection")
    columns = df.columns.tolist()
    target_col = st.sidebar.selectbox("Target Column", columns, index=len(columns)-1)
    sens_opts = [c for c in columns if c != target_col]
    def_idx = sens_opts.index("Gender") if "Gender" in sens_opts else 0
    sensitive_col = st.sidebar.selectbox("Sensitive Attribute", sens_opts, index=def_idx)

    # ── 3. Algorithm & Hyperparams ───────────────────────────────────────
    st.sidebar.title("3. Algorithm")
    all_models = ml_utils.available_models()
    model_type = st.sidebar.selectbox("ML Algorithm", all_models)
    hp = sidebar_hyperparams(model_type)

    # ── 4. Training Options ──────────────────────────────────────────────
    st.sidebar.title("4. Training Options")
    cv_choice = st.sidebar.selectbox("Cross-Validation", ["None", "3-Fold", "5-Fold", "10-Fold"])
    cv_folds = {"None": None, "3-Fold": 3, "5-Fold": 5, "10-Fold": 10}[cv_choice]

    imbalance_opts = ["None", "SMOTE", "Class Weighting"]
    imbalance_strategy = st.sidebar.selectbox("Imbalance Handling", imbalance_opts)
    if imbalance_strategy == "SMOTE" and not ml_utils.SMOTE_OK:
        st.sidebar.warning("imbalanced-learn not installed. SMOTE unavailable.")
        imbalance_strategy = "None"

    # ── 5. Bias Mitigation ───────────────────────────────────────────────
    st.sidebar.title("5. Bias Mitigation")
    enable_mitigation = st.sidebar.toggle("Enable Bias Mitigation")
    mitigation_strategy = None
    if enable_mitigation:
        mit_opts = [
            "Pre-processing: Reweighting",
            "Post-processing: Threshold Adjustment",
            "In-processing: Demographic Parity (FairLearn)",
            "In-processing: Equalized Odds (FairLearn)",
        ]
        mitigation_strategy = st.sidebar.selectbox("Mitigation Strategy", mit_opts)
        if "FairLearn" in (mitigation_strategy or "") and not ml_utils.FAIRLEARN_OK:
            st.sidebar.error("fairlearn not installed. Choose another strategy.")
            mitigation_strategy = None

    show_before_after = enable_mitigation and mitigation_strategy is not None

    # ── 6. Run Mode ──────────────────────────────────────────────────────
    st.sidebar.title("6. Run Mode")
    run_mode = st.sidebar.radio("Mode", ["Single Model", "Compare Models"])
    compare_selection = None
    if run_mode == "Compare Models":
        compare_selection = st.sidebar.multiselect(
            "Models to compare", all_models, default=all_models[:3])

    run_btn = st.sidebar.button("🚀 Run", type="primary", use_container_width=True)
    if not run_btn:
        return

    # ════════════════════════════════════════════════════════════════════
    # COMPARE MODE
    # ════════════════════════════════════════════════════════════════════
    if run_mode == "Compare Models":
        if not compare_selection:
            st.warning("Please select at least one model to compare.")
            return
        with st.spinner(f"Training {len(compare_selection)} models..."):
            try:
                results, is_binary = ml_utils.compare_all_models(
                    df, target_col, sensitive_col, compare_selection)
            except Exception as e:
                st.error(f"Error: {e}"); return
        st.header("Model Comparison")
        ml_utils.plot_model_comparison(results, is_binary)
        return

    # ════════════════════════════════════════════════════════════════════
    # SINGLE MODEL MODE
    # ════════════════════════════════════════════════════════════════════
    spinner_msg = f"Training {model_type}" + (f" with {mitigation_strategy}" if mitigation_strategy else "") + "..."

    if show_before_after:
        with st.spinner("Running Before vs After comparison..."):
            try:
                before, after = ml_utils.compare_before_after(
                    df, target_col, sensitive_col, model_type, hp,
                    cv_folds, imbalance_strategy, mitigation_strategy)
                result = after
            except Exception as e:
                st.error(f"Error: {e}"); return
    else:
        with st.spinner(spinner_msg):
            try:
                result = ml_utils.train_and_evaluate(
                    df, target_col, sensitive_col, model_type, hp,
                    cv_folds, imbalance_strategy, mitigation_strategy)
            except Exception as e:
                st.error(f"Error: {e}"); return

    gen = result["gen_metrics"]
    group_df = result["group_df"]

    # ── Overall metrics ──────────────────────────────────────────────────
    st.header("Evaluation Results")
    algo_label = model_type + (f" + {mitigation_strategy}" if mitigation_strategy else "")
    st.caption(f"Algorithm: **{algo_label}** | CV: **{cv_choice}** | Imbalance: **{imbalance_strategy}**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{gen['Accuracy']:.2%}")
    c2.metric("Precision", f"{gen['Precision (Weighted)']:.2%}")
    c3.metric("Recall",    f"{gen['Recall (Weighted)']:.2%}")
    c4.metric("F1-Score",  f"{gen['F1-Score (Weighted)']:.2%}")

    # ── Tabs ─────────────────────────────────────────────────────────────
    st.divider()
    tab_labels = [
        "📊 Fairness Metrics", "🗓️ Confusion Matrices",
        "📈 ROC Curve", "🎯 Precision-Recall",
        "🔍 Model Insights", "📋 Group Table", "💡 Summary"
    ]
    if result.get("cv_results"):
        tab_labels.insert(0, "📐 Cross-Validation")
    if show_before_after:
        tab_labels.append("⚖️ Before vs After")

    tabs = st.tabs(tab_labels)
    idx = 0

    if result.get("cv_results"):
        with tabs[idx]: ml_utils.show_cv_results(result["cv_results"])
        idx += 1

    with tabs[idx]:   ml_utils.plot_bias_metrics(group_df);                                  idx += 1
    with tabs[idx]:   ml_utils.plot_confusion_matrices(result["y_test"], result["y_pred"], result["sens_test"]); idx += 1
    with tabs[idx]:   ml_utils.plot_roc_curve(result["y_test"], result["y_prob"], result["sens_test"]); idx += 1
    with tabs[idx]:   ml_utils.plot_precision_recall_curve(result["y_test"], result["y_prob"], result["sens_test"]); idx += 1
    with tabs[idx]:   ml_utils.plot_model_specific(result["model"], result["feature_names"], model_type); idx += 1
    with tabs[idx]:   ml_utils.show_group_metrics_table(group_df);                           idx += 1
    with tabs[idx]:   ml_utils.generate_bias_insights(group_df);                              idx += 1

    if show_before_after:
        with tabs[idx]:
            ml_utils.plot_before_after(before, after, mitigation_strategy)


if __name__ == "__main__":
    main()
