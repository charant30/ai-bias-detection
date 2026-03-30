"""
ml_utils.py — Complete ML backend for the Bias Detection Dashboard.
Supports: 8 algorithms, hyperparameter tuning, cross-validation,
          SMOTE / class-weighting, and 3 bias mitigation strategies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, make_scorer
)

# ── Optional dependencies ────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    XGBOOST_OK = True
except ImportError:
    XGBOOST_OK = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_OK = True
except ImportError:
    LIGHTGBM_OK = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_OK = True
except ImportError:
    SMOTE_OK = False

try:
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
    FAIRLEARN_OK = True
except ImportError:
    FAIRLEARN_OK = False

# ════════════════════════════════════════════════════════════════════════════
# MODEL FACTORY
# ════════════════════════════════════════════════════════════════════════════

def available_models():
    names = [
        "Logistic Regression", "Decision Tree", "Random Forest",
        "Gradient Boosting", "SVM",  "KNN"
    ]
    if XGBOOST_OK:
        names.append("XGBoost")
    if LIGHTGBM_OK:
        names.append("LightGBM")
    return names


def get_model(model_type, hp: dict):
    """Instantiate the chosen model with given hyperparameters."""
    if model_type == "Logistic Regression":
        return LogisticRegression(C=hp.get("C", 1.0), max_iter=1000)
    elif model_type == "Decision Tree":
        return DecisionTreeClassifier(max_depth=hp.get("max_depth", 8), random_state=42)
    elif model_type == "Random Forest":
        return RandomForestClassifier(
            n_estimators=hp.get("n_estimators", 100),
            max_depth=hp.get("max_depth", None),
            random_state=42)
    elif model_type == "Gradient Boosting":
        return GradientBoostingClassifier(
            n_estimators=hp.get("n_estimators", 100),
            learning_rate=hp.get("learning_rate", 0.1),
            max_depth=hp.get("max_depth", 3),
            random_state=42)
    elif model_type == "SVM":
        return SVC(C=hp.get("C", 1.0), kernel=hp.get("kernel", "rbf"), probability=True)
    elif model_type == "KNN":
        return KNeighborsClassifier(
            n_neighbors=hp.get("n_neighbors", 5),
            weights=hp.get("weights", "uniform"))
    elif model_type == "XGBoost" and XGBOOST_OK:
        return XGBClassifier(
            n_estimators=hp.get("n_estimators", 100),
            learning_rate=hp.get("learning_rate", 0.1),
            max_depth=hp.get("max_depth", 3),
            use_label_encoder=False, eval_metric="logloss",
            verbosity=0, random_state=42)
    elif model_type == "LightGBM" and LIGHTGBM_OK:
        return LGBMClassifier(
            n_estimators=hp.get("n_estimators", 100),
            learning_rate=hp.get("learning_rate", 0.1),
            num_leaves=hp.get("num_leaves", 31),
            verbosity=-1, random_state=42)
    else:
        return LogisticRegression(max_iter=1000)


# ════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ════════════════════════════════════════════════════════════════════════════

def prepare_data(df, target_col, sensitive_col, test_size=0.3):
    df_clean = df.dropna(subset=[target_col, sensitive_col]).copy()
    if df_clean.empty:
        raise ValueError("Dataset is empty after dropping rows with missing values.")

    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    sensitive = df_clean[sensitive_col]

    X_enc = pd.get_dummies(X, drop_first=True)
    feature_names = X_enc.columns.tolist()

    le = None
    if y.dtype == "object" or y.dtype.name == "category":
        le = LabelEncoder()
        y = le.fit_transform(y)

    is_binary = len(np.unique(y)) == 2

    X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
        X_enc, y, sensitive, test_size=test_size, random_state=42, stratify=y
    )
    return X_tr, X_te, y_tr, y_te, s_tr, s_te, feature_names, is_binary


# ════════════════════════════════════════════════════════════════════════════
# IMBALANCE STRATEGIES
# ════════════════════════════════════════════════════════════════════════════

def apply_imbalance(X_train, y_train, strategy):
    if strategy == "SMOTE":
        if not SMOTE_OK:
            st.warning("imbalanced-learn not installed. SMOTE skipped.")
            return X_train, y_train, None
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        return X_res, y_res, None   # no sample_weight needed
    return X_train, y_train, None


def compute_sample_weights(y_train, s_train, strategy):
    """Return sample_weight array or None."""
    if strategy == "Class Weighting":
        classes, counts = np.unique(y_train, return_counts=True)
        weight_map = {c: len(y_train) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
        return np.array([weight_map[yi] for yi in y_train])
    if strategy == "Pre-processing: Reweighting":
        groups, gcounts = np.unique(s_train, return_counts=True)
        g_weight_map = {g: len(s_train) / (len(groups) * cnt) for g, cnt in zip(groups, gcounts)}
        return np.array([g_weight_map[gi] for gi in s_train])
    return None


# ════════════════════════════════════════════════════════════════════════════
# METRIC HELPERS
# ════════════════════════════════════════════════════════════════════════════

def compute_general_metrics(y_test, y_pred):
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (Weighted)": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall (Weighted)": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1-Score (Weighted)": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }


def compute_group_metrics(y_test, y_pred, s_test, is_binary):
    rows = []
    y_test_arr, y_pred_arr, s_arr = np.array(y_test), np.array(y_pred), np.array(s_test)
    for g in np.unique(s_arr):
        mask = s_arr == g
        yt_g, yp_g = y_test_arr[mask], y_pred_arr[mask]
        fpr = fnr = 0.0
        if is_binary and len(np.unique(yt_g)) == 2:
            try:
                tn, fp, fn, tp = confusion_matrix(yt_g, yp_g, labels=[0, 1]).ravel()
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            except Exception:
                pass
        rows.append({
            "Group": str(g), "Size": int(mask.sum()),
            "Accuracy": accuracy_score(yt_g, yp_g),
            "F1-Score": f1_score(yt_g, yp_g, average="weighted", zero_division=0),
            "FPR": fpr, "FNR": fnr,
        })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# POST-PROCESSING: THRESHOLD ADJUSTMENT
# ════════════════════════════════════════════════════════════════════════════

def post_process_thresholds(model, X_test, y_test, s_test, is_binary):
    """Equalize FPR across groups by setting group-specific decision thresholds."""
    if not is_binary:
        return None, {}

    y_prob = model.predict_proba(X_test)[:, 1]
    y_test_arr, s_arr = np.array(y_test), np.array(s_test)

    # baseline overall FPR at 0.5
    y_def = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test_arr, y_def, labels=[0, 1]).ravel()
    target_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.3

    group_thresholds = {}
    for g in np.unique(s_arr):
        mask = s_arr == g
        yt_g, yp_g = y_test_arr[mask], y_prob[mask]
        if len(np.unique(yt_g)) < 2:
            group_thresholds[g] = 0.5
            continue
        fpr_arr, _, thr_arr = roc_curve(yt_g, yp_g)
        idx = int(np.argmin(np.abs(fpr_arr - target_fpr)))
        idx = min(idx, len(thr_arr) - 1)
        group_thresholds[g] = float(thr_arr[idx])

    # Build adjusted predictions
    y_adj = np.zeros(len(y_test_arr), dtype=int)
    for i in range(len(y_test_arr)):
        thr = group_thresholds.get(s_arr[i], 0.5)
        y_adj[i] = 1 if y_prob[i] >= thr else 0

    return y_adj, group_thresholds


# ════════════════════════════════════════════════════════════════════════════
# MAIN TRAIN & EVALUATE
# ════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(df, target_col, sensitive_col, model_type, hp,
                       cv_folds, imbalance_strategy, mitigation_strategy):
    """
    Full pipeline: prepare → imbalance → train → (mitigation) → evaluate.
    Returns a dict with all data needed for visualisation.
    """
    (X_tr, X_te, y_tr, y_te,
     s_tr, s_te, feature_names, is_binary) = prepare_data(df, target_col, sensitive_col)

    # ── Cross-validation (overall metrics only) ──────────────────────────
    cv_results = None
    if cv_folds and cv_folds > 1:
        base_model = get_model(model_type, hp)
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scoring = {"accuracy": "accuracy",
                   "precision": make_scorer(precision_score, average="weighted", zero_division=0),
                   "recall": make_scorer(recall_score, average="weighted", zero_division=0),
                   "f1": make_scorer(f1_score, average="weighted", zero_division=0)}
        X_all = pd.concat([X_tr, X_te])
        y_all = np.concatenate([y_tr, y_te])
        cv_raw = cross_validate(base_model, X_all, y_all, cv=skf, scoring=scoring)
        cv_results = {
            "Accuracy":   (cv_raw["test_accuracy"].mean(),   cv_raw["test_accuracy"].std()),
            "Precision":  (cv_raw["test_precision"].mean(),  cv_raw["test_precision"].std()),
            "Recall":     (cv_raw["test_recall"].mean(),     cv_raw["test_recall"].std()),
            "F1-Score":   (cv_raw["test_f1"].mean(),         cv_raw["test_f1"].std()),
        }

    # ── Imbalance strategy ───────────────────────────────────────────────
    X_tr_res, y_tr_res, _ = apply_imbalance(X_tr, y_tr, imbalance_strategy)
    sample_weight = compute_sample_weights(y_tr_res, s_tr[:len(y_tr_res)], imbalance_strategy)

    # Support class-weighting via sample_weight too
    cw_weight = compute_sample_weights(y_tr_res, s_tr[:len(y_tr_res)], "Class Weighting") \
        if imbalance_strategy == "Class Weighting" else None
    final_weight = cw_weight if cw_weight is not None else sample_weight

    # ── Model training ───────────────────────────────────────────────────
    use_fairlearn = (mitigation_strategy or "").startswith("In-processing") and FAIRLEARN_OK
    model = get_model(model_type, hp)

    if use_fairlearn:
        constraint = DemographicParity() if "Demographic" in mitigation_strategy else EqualizedOdds()
        eg = ExponentiatedGradient(estimator=model, constraints=constraint)
        eg.fit(X_tr_res, y_tr_res, sensitive_features=s_tr[:len(y_tr_res)])
        y_pred = eg.predict(X_te)
        y_prob = None   # EG does not expose predict_proba reliably
        trained_model = eg
    else:
        kw = {}
        if final_weight is not None:
            try:
                model.fit(X_tr_res, y_tr_res, sample_weight=final_weight)
                kw["used_sw"] = True
            except TypeError:
                model.fit(X_tr_res, y_tr_res)
        else:
            model.fit(X_tr_res, y_tr_res)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1] if is_binary else None
        trained_model = model

        # ── Post-processing threshold adjustment ─────────────────────────
        if mitigation_strategy == "Post-processing: Threshold Adjustment" and is_binary:
            y_pred, thresholds = post_process_thresholds(model, X_te, y_te, s_te, is_binary)
        else:
            thresholds = {}

    gen_metrics = compute_general_metrics(y_te, y_pred)
    group_df = compute_group_metrics(y_te, y_pred, s_te, is_binary)

    return {
        "gen_metrics": gen_metrics,
        "group_df": group_df,
        "cv_results": cv_results,
        "y_test": np.array(y_te),
        "y_pred": y_pred,
        "y_prob": y_prob,
        "sens_test": s_te,
        "model": trained_model,
        "feature_names": feature_names,
        "is_binary": is_binary,
        "thresholds": thresholds,
    }


# ════════════════════════════════════════════════════════════════════════════
# BEFORE vs AFTER COMPARISON
# ════════════════════════════════════════════════════════════════════════════

def compare_before_after(df, target_col, sensitive_col, model_type, hp,
                         cv_folds, imbalance_strategy, mitigation_strategy):
    """Returns (before_result, after_result) dicts."""
    before = train_and_evaluate(
        df, target_col, sensitive_col, model_type, hp, cv_folds,
        imbalance_strategy="None", mitigation_strategy=None)
    after = train_and_evaluate(
        df, target_col, sensitive_col, model_type, hp, cv_folds,
        imbalance_strategy=imbalance_strategy, mitigation_strategy=mitigation_strategy)
    return before, after


# ════════════════════════════════════════════════════════════════════════════
# COMPARE ALL MODELS
# ════════════════════════════════════════════════════════════════════════════

def compare_all_models(df, target_col, sensitive_col, model_names):
    hp_defaults = {}
    (X_tr, X_te, y_tr, y_te,
     s_tr, s_te, feature_names, is_binary) = prepare_data(df, target_col, sensitive_col)

    results = {}
    for name in model_names:
        model = get_model(name, hp_defaults)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1] if is_binary else None
        results[name] = {
            "y_test": np.array(y_te), "y_pred": y_pred, "y_prob": y_prob,
            "sens_test": s_te,
            "overall": compute_general_metrics(y_te, y_pred),
            "group_df": compute_group_metrics(y_te, y_pred, s_te, is_binary),
        }
    return results, is_binary


# ════════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS — FAIRNESS CHARTS
# ════════════════════════════════════════════════════════════════════════════

def plot_bias_metrics(group_df):
    st.markdown("### Fairness Metrics by Group")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    charts = [("Accuracy", "viridis"), ("F1-Score", "mako"),
              ("FPR", "magma"), ("FNR", "cubehelix")]
    for ax, (metric, pal) in zip(axes.flatten(), charts):
        sns.barplot(data=group_df, x="Group", y=metric, ax=ax, palette=pal)
        ax.set_title(f"{metric} by Group"); ax.set_ylim(0, 1.05)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)


def plot_confusion_matrices(y_test, y_pred, sens_test):
    st.markdown("### Confusion Matrices")
    groups = list(np.unique(np.array(sens_test)))
    n = 1 + len(groups); cols = min(n, 3); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.array(axes).flatten()

    def _cm(ax, yt, yp, title):
        sns.heatmap(confusion_matrix(yt, yp), annot=True, fmt="d",
                    cmap="Blues", ax=ax, cbar=False, linewidths=0.5)
        ax.set(title=title, xlabel="Predicted", ylabel="Actual")

    _cm(axes[0], y_test, y_pred, "Overall")
    for i, g in enumerate(groups):
        mask = np.array(sens_test) == g
        _cm(axes[i+1], y_test[mask], y_pred[mask], f"Group: {g}")
    for ax in axes[n:]: ax.set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)


def plot_roc_curve(y_test, y_prob, sens_test):
    if y_prob is None:
        st.info("ROC Curve requires binary classification. Skipping."); return
    st.markdown("### ROC Curve")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    pal = plt.cm.tab10.colors

    fpr_a, tpr_a, _ = roc_curve(y_test, y_prob)
    axes[0].plot(fpr_a, tpr_a, lw=2, color="steelblue",
                 label=f"AUC = {auc(fpr_a, tpr_a):.3f}")
    axes[0].plot([0,1],[0,1],"--",color="gray",lw=1)
    axes[0].set(title="Overall ROC", xlabel="FPR", ylabel="TPR", xlim=[0,1], ylim=[0,1.02])
    axes[0].legend(loc="lower right")

    for i, g in enumerate(np.unique(np.array(sens_test))):
        mask = np.array(sens_test) == g
        yt_g, yp_g = y_test[mask], y_prob[mask]
        if len(np.unique(yt_g)) < 2: continue
        fg, tg, _ = roc_curve(yt_g, yp_g)
        axes[1].plot(fg, tg, lw=2, color=pal[i%10], label=f"{g} (AUC={auc(fg,tg):.3f})")
    axes[1].plot([0,1],[0,1],"--",color="gray",lw=1)
    axes[1].set(title="ROC by Group", xlabel="FPR", ylabel="TPR", xlim=[0,1], ylim=[0,1.02])
    axes[1].legend(loc="lower right")
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)


def plot_precision_recall_curve(y_test, y_prob, sens_test):
    if y_prob is None:
        st.info("PR Curve requires binary classification. Skipping."); return
    st.markdown("### Precision-Recall Curve")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    pal = plt.cm.tab10.colors

    p, r, _ = precision_recall_curve(y_test, y_prob)
    axes[0].plot(r, p, lw=2, color="darkorange",
                 label=f"AP = {average_precision_score(y_test, y_prob):.3f}")
    axes[0].set(title="Overall PR", xlabel="Recall", ylabel="Precision", xlim=[0,1], ylim=[0,1.05])
    axes[0].legend(loc="lower left")

    for i, g in enumerate(np.unique(np.array(sens_test))):
        mask = np.array(sens_test) == g
        yt_g, yp_g = y_test[mask], y_prob[mask]
        if len(np.unique(yt_g)) < 2: continue
        pg, rg, _ = precision_recall_curve(yt_g, yp_g)
        axes[1].plot(rg, pg, lw=2, color=pal[i%10],
                     label=f"{g} (AP={average_precision_score(yt_g, yp_g):.3f})")
    axes[1].set(title="PR by Group", xlabel="Recall", ylabel="Precision", xlim=[0,1], ylim=[0,1.05])
    axes[1].legend(loc="lower left")
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)


def plot_model_specific(model, feature_names, model_type, top_n=20):
    st.markdown(f"### Model-Specific Insights — {model_type}")
    # Handle FairLearn wrapper
    base = getattr(model, "_best_estimator", None) or model

    if model_type == "Logistic Regression" and hasattr(base, "coef_"):
        coefs = base.coef_[0]
        idx = np.argsort(np.abs(coefs))[::-1][:top_n]
        colors = ["#e74c3c" if c < 0 else "#2ecc71" for c in coefs[idx]]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(np.array(feature_names)[idx][::-1], coefs[idx][::-1], color=colors[::-1])
        ax.axvline(0, color="black", lw=0.8, linestyle="--")
        ax.set(title=f"Top {top_n} Coefficients", xlabel="Value")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.caption("🟢 Positive → increases class 1 probability   🔴 Negative → decreases it")

    elif hasattr(base, "feature_importances_"):
        imp = base.feature_importances_
        idx = np.argsort(imp)[::-1][:top_n]
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=imp[idx][::-1], y=np.array(feature_names)[idx][::-1],
                    ax=ax, palette="viridis")
        ax.set(title=f"Top {top_n} Feature Importances ({model_type})", xlabel="Importance")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.caption("Higher importance = greater influence on predictions.")
    else:
        st.info(f"No feature importance or coefficient plot available for {model_type}.")


def show_group_metrics_table(group_df):
    st.markdown("### Detailed Group Metrics Table")
    disp = group_df.copy()
    for col in ["Accuracy", "F1-Score", "FPR", "FNR"]:
        if col in disp:
            disp[col] = disp[col].map(lambda v: f"{v:.2%}")
    st.dataframe(disp, use_container_width=True)


def show_cv_results(cv_results):
    if not cv_results:
        return
    st.markdown("### Cross-Validation Results (mean ± std)")
    rows = [{"Metric": k, "Mean": f"{v[0]:.4f}", "Std Dev": f"±{v[1]:.4f}",
             "Mean (%)": f"{v[0]:.2%}"} for k, v in cv_results.items()]
    st.dataframe(pd.DataFrame(rows).set_index("Metric"), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# BEFORE vs AFTER VISUALISATION
# ════════════════════════════════════════════════════════════════════════════

def plot_before_after(before, after, mitigation_label):
    st.markdown(f"### Before vs After — {mitigation_label}")

    metrics = ["Accuracy", "Precision (Weighted)", "Recall (Weighted)", "F1-Score (Weighted)"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

    # ── Overall metric bar chart ─────────────────────────────────────────
    x = np.arange(len(labels)); w = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    b_vals = [before["gen_metrics"][m] for m in metrics]
    a_vals = [after["gen_metrics"][m] for m in metrics]
    bars_b = ax.bar(x - w/2, b_vals, w, label="Before", color="#4e79a7", alpha=0.88)
    bars_a = ax.bar(x + w/2, a_vals, w, label="After",  color="#f28e2b", alpha=0.88)
    ax.bar_label(bars_b, labels=[f"{v:.1%}" for v in b_vals], padding=3, fontsize=8)
    ax.bar_label(bars_a, labels=[f"{v:.1%}" for v in a_vals], padding=3, fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.15)
    ax.set(title="Overall Metric Comparison", ylabel="Score"); ax.legend()
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ── Per-group accuracy before/after ─────────────────────────────────
    st.markdown("#### Group Accuracy: Before vs After")
    groups_b = before["group_df"].set_index("Group")
    groups_a = after["group_df"].set_index("Group")
    all_groups = sorted(set(groups_b.index) | set(groups_a.index))

    x_g = np.arange(len(all_groups))
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    b_acc = [groups_b.loc[g, "Accuracy"] if g in groups_b.index else 0 for g in all_groups]
    a_acc = [groups_a.loc[g, "Accuracy"] if g in groups_a.index else 0 for g in all_groups]
    bars_b2 = ax2.bar(x_g - w/2, b_acc, w, label="Before", color="#4e79a7", alpha=0.88)
    bars_a2 = ax2.bar(x_g + w/2, a_acc, w, label="After",  color="#f28e2b", alpha=0.88)
    ax2.bar_label(bars_b2, labels=[f"{v:.1%}" for v in b_acc], padding=3, fontsize=8)
    ax2.bar_label(bars_a2, labels=[f"{v:.1%}" for v in a_acc], padding=3, fontsize=8)
    ax2.set_xticks(x_g); ax2.set_xticklabels(all_groups); ax2.set_ylim(0, 1.15)
    ax2.set(title="Group Accuracy Before vs After", ylabel="Accuracy"); ax2.legend()
    plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)

    # ── FPR / FNR comparison table ───────────────────────────────────────
    st.markdown("#### Fairness Metrics Delta")
    rows = []
    for g in all_groups:
        b_row = groups_b.loc[g] if g in groups_b.index else {}
        a_row = groups_a.loc[g] if g in groups_a.index else {}
        rows.append({
            "Group": g,
            "Acc Before": f"{b_row.get('Accuracy', 0):.2%}",
            "Acc After":  f"{a_row.get('Accuracy', 0):.2%}",
            "FPR Before": f"{b_row.get('FPR', 0):.2%}",
            "FPR After":  f"{a_row.get('FPR', 0):.2%}",
            "FNR Before": f"{b_row.get('FNR', 0):.2%}",
            "FNR After":  f"{a_row.get('FNR', 0):.2%}",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Group"), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# MODEL COMPARISON CHARTS
# ════════════════════════════════════════════════════════════════════════════

def plot_model_comparison(results, is_binary):
    model_names = list(results.keys())
    palette = ["#4e79a7","#f28e2b","#59a14f","#e15759","#76b7b2","#edc948","#b07aa1","#ff9da7"]
    model_colors = dict(zip(model_names, palette))

    st.markdown("### Overall Metrics — All Models")
    rows = [{"Model": n, **{k: f"{v:.2%}" for k, v in r["overall"].items()}}
            for n, r in results.items()]
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

    metric_keys = ["Accuracy","Precision (Weighted)","Recall (Weighted)","F1-Score (Weighted)"]
    metric_labels = ["Accuracy","Precision","Recall","F1"]
    x = np.arange(len(metric_labels)); w = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (name, res) in enumerate(results.items()):
        vals = [res["overall"][k] for k in metric_keys]
        bars = ax.bar(x + i*w - (len(model_names)-1)*w/2, vals, w,
                      label=name, color=palette[i % len(palette)], alpha=0.88)
        ax.bar_label(bars, labels=[f"{v:.0%}" for v in vals], padding=2, fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.2); ax.set_ylabel("Score"); ax.set_title("Model Performance Comparison")
    ax.legend(loc="upper right"); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown("### Accuracy by Group — All Models")
    groups = sorted(results[model_names[0]]["group_df"]["Group"].unique())
    x_g = np.arange(len(groups))
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    for i, (name, res) in enumerate(results.items()):
        gdf = res["group_df"].set_index("Group")
        vals = [gdf.loc[g, "Accuracy"] if g in gdf.index else 0 for g in groups]
        bars = ax2.bar(x_g + i*w - (len(model_names)-1)*w/2, vals, w,
                       label=name, color=palette[i % len(palette)], alpha=0.88)
        ax2.bar_label(bars, labels=[f"{v:.0%}" for v in vals], padding=2, fontsize=7)
    ax2.set_xticks(x_g); ax2.set_xticklabels(groups)
    ax2.set_ylim(0, 1.2); ax2.set_ylabel("Accuracy"); ax2.set_title("Accuracy by Group per Model")
    ax2.legend(); plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)

    st.markdown("### Fairness Heatmaps by Group")
    for metric in ["F1-Score","FPR","FNR"]:
        heat = {name: {g: res["group_df"].set_index("Group").loc[g, metric]
                       if g in res["group_df"]["Group"].values else np.nan
                       for g in groups} for name, res in results.items()}
        heat_df = pd.DataFrame(heat, index=groups)
        cmap = "RdYlGn_r" if metric in ("FPR","FNR") else "RdYlGn"
        fig3, ax3 = plt.subplots(figsize=(max(6, len(model_names)*2.5), max(3, len(groups)*0.8)))
        sns.heatmap(heat_df, annot=True, fmt=".0%", cmap=cmap, ax=ax3,
                    linewidths=0.5, vmin=0, vmax=1)
        ax3.set_title(f"{metric} — Group × Model")
        plt.tight_layout(); st.pyplot(fig3); plt.close(fig3)

    if is_binary:
        st.markdown("### ROC Curves — All Models")
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        for name, res in results.items():
            if res["y_prob"] is not None:
                fv, tv, _ = roc_curve(res["y_test"], res["y_prob"])
                ax4.plot(fv, tv, lw=2, color=model_colors[name],
                         label=f"{name} (AUC={auc(fv,tv):.3f})")
        ax4.plot([0,1],[0,1],"--",color="gray",lw=1)
        ax4.set(xlabel="FPR",ylabel="TPR",title="ROC Comparison",xlim=[0,1],ylim=[0,1.02])
        ax4.legend(loc="lower right"); plt.tight_layout(); st.pyplot(fig4); plt.close(fig4)

        st.markdown("### Precision-Recall Curves — All Models")
        fig5, ax5 = plt.subplots(figsize=(8, 6))
        for name, res in results.items():
            if res["y_prob"] is not None:
                pv, rv, _ = precision_recall_curve(res["y_test"], res["y_prob"])
                ap = average_precision_score(res["y_test"], res["y_prob"])
                ax5.plot(rv, pv, lw=2, color=model_colors[name], label=f"{name} (AP={ap:.3f})")
        ax5.set(xlabel="Recall",ylabel="Precision",title="PR Comparison",xlim=[0,1],ylim=[0,1.05])
        ax5.legend(loc="lower left"); plt.tight_layout(); st.pyplot(fig5); plt.close(fig5)


# ════════════════════════════════════════════════════════════════════════════
# SUMMARY INSIGHTS
# ════════════════════════════════════════════════════════════════════════════

def generate_bias_insights(group_df):
    import streamlit_antd_components as sac
    st.markdown("### Summary Insights")

    if group_df.empty or len(group_df) < 2:
        sac.alert("Not enough groups to compare bias.", color="info", key="alert_not_enough")
        return

    acc_diff = group_df["Accuracy"].max() - group_df["Accuracy"].min()
    best = group_df.loc[group_df["Accuracy"].idxmax(), "Group"]
    worst = group_df.loc[group_df["Accuracy"].idxmin(), "Group"]
    best_val = group_df["Accuracy"].max()
    worst_val = group_df["Accuracy"].min()

    if acc_diff > 0.10:
        sac.alert(
            f"Potential Bias Detected in Accuracy! '{best}' scores {best_val:.2%} vs "
            f"'{worst}' at {worst_val:.2%} — a {acc_diff:.2%} gap.",
            color="warning", icon="exclamation-triangle-fill", key="alert_acc_bias")
    else:
        sac.alert(
            f"Accuracy is fair across groups. Max gap: {acc_diff:.2%}.",
            color="success", icon="check-circle-fill", key="alert_acc_fair")

    if "FPR" in group_df.columns and group_df["FPR"].sum() > 0:
        fpr_diff = group_df["FPR"].max() - group_df["FPR"].min()
        if fpr_diff > 0.10:
            sac.alert(f"FPR Bias Detected — {fpr_diff:.2%} disparity across groups.",
                      color="warning", icon="exclamation-circle", key="alert_fpr_bias")

    if "FNR" in group_df.columns and group_df["FNR"].sum() > 0:
        fnr_diff = group_df["FNR"].max() - group_df["FNR"].min()
        if fnr_diff > 0.10:
            sac.alert(f"FNR Bias Detected — {fnr_diff:.2%} disparity across groups.",
                      color="warning", icon="exclamation-circle", key="alert_fnr_bias")
