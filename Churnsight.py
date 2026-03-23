# ============================================================
#  ChurnSight — Customer Attrition Engine
#  Author : Vighan Raj Verma (@Vighan-coder)
#  GitHub : https://github.com/Vighan-coder/ChurnSight
# ============================================================
#
#  SETUP:
#    pip install pandas numpy scikit-learn xgboost imbalanced-learn shap streamlit matplotlib seaborn
#
#  RUN TRAINING:
#    python churnsight.py
#
#  RUN DASHBOARD:
#    streamlit run churnsight.py  (set RUN_APP = True below)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, f1_score)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ── Toggle this to launch the Streamlit dashboard ──────────
RUN_APP = False


# ════════════════════════════════════════════════════════════
#  1. SYNTHETIC DATA GENERATOR
#     (Replace with your real CSV by calling load_data())
# ════════════════════════════════════════════════════════════
def generate_data(n=5000, seed=42):
    """Generate a realistic telecom churn dataset."""
    np.random.seed(seed)
    df = pd.DataFrame({
        "CustomerID"       : range(1, n + 1),
        "Tenure"           : np.random.randint(1, 72, n),
        "MonthlyCharges"   : np.round(np.random.uniform(20, 120, n), 2),
        "TotalCharges"     : np.round(np.random.uniform(100, 8000, n), 2),
        "Contract"         : np.random.choice(["Month-to-month", "One year", "Two year"], n,
                                               p=[0.55, 0.25, 0.20]),
        "InternetService"  : np.random.choice(["DSL", "Fiber optic", "No"], n,
                                               p=[0.35, 0.45, 0.20]),
        "TechSupport"      : np.random.choice(["Yes", "No"], n),
        "OnlineSecurity"   : np.random.choice(["Yes", "No"], n),
        "PaymentMethod"    : np.random.choice(
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n),
        "SeniorCitizen"    : np.random.choice([0, 1], n, p=[0.84, 0.16]),
        "Dependents"       : np.random.choice(["Yes", "No"], n, p=[0.30, 0.70]),
        "PaperlessBilling" : np.random.choice(["Yes", "No"], n, p=[0.60, 0.40]),
        "NumSupportCalls"  : np.random.poisson(1.5, n),
    })
    # Churn probability driven by real-world factors
    churn_prob = (
        0.40 * (df["Contract"] == "Month-to-month").astype(int)
        + 0.15 * (df["InternetService"] == "Fiber optic").astype(int)
        + 0.10 * (df["TechSupport"] == "No").astype(int)
        + 0.10 * (df["OnlineSecurity"] == "No").astype(int)
        - 0.20 * (df["Tenure"] / 72)
        + 0.05 * (df["NumSupportCalls"] / 10)
        + np.random.normal(0, 0.05, n)
    )
    df["Churn"] = (churn_prob > 0.30).astype(int)
    return df


def load_data(path=None):
    """Load from CSV or generate synthetic data."""
    if path:
        return pd.read_csv(path)
    print("[ChurnSight] No CSV path given — using synthetic data.")
    return generate_data()


# ════════════════════════════════════════════════════════════
#  2. PREPROCESSING
# ════════════════════════════════════════════════════════════
def preprocess(df: pd.DataFrame):
    df = df.drop(columns=["CustomerID"], errors="ignore").copy()

    # Encode categorical columns
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y, X.columns.tolist()


# ════════════════════════════════════════════════════════════
#  3. SMOTE + TRAIN / EVAL
# ════════════════════════════════════════════════════════════
def train_and_evaluate(X, y, feature_names):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)

    # Handle class imbalance
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"[SMOTE] Resampled train size: {X_res.shape[0]} "
          f"(pos={y_res.sum()}, neg={(y_res==0).sum()})")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.80,
        colsample_bytree=0.80,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_res, y_res, cv=cv, scoring="roc_auc")
    print(f"[CV] ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    model.fit(X_res, y_res,
              eval_set=[(X_test, y_test)],
              verbose=False)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n── Classification Report ──────────────────────────")
    print(classification_report(y_test, y_pred, target_names=["Stay", "Churn"]))
    print(f"ROC-AUC : {roc_auc_score(y_test, y_proba):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")

    return model, X_test, y_test, y_proba, feature_names


# ════════════════════════════════════════════════════════════
#  4. VISUALISATIONS
# ════════════════════════════════════════════════════════════
def plot_roc(y_test, y_proba):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="#7cff67", lw=2, label=f"AUC = {auc:.3f}")
    plt.plot([0,1],[0,1], "w--", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — ChurnSight")
    plt.legend(); plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=150)
    plt.show()
    print("[Saved] roc_curve.png")


def plot_confusion(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=["Stay","Churn"], yticklabels=["Stay","Churn"])
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("[Saved] confusion_matrix.png")


def plot_shap(model, X_test, feature_names):
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_test)
    plt.figure()
    shap.summary_plot(shap_vals, X_test,
                      feature_names=feature_names,
                      plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=150)
    plt.show()
    print("[Saved] shap_summary.png")


# ════════════════════════════════════════════════════════════
#  5. STREAMLIT DASHBOARD  (streamlit run churnsight.py)
# ════════════════════════════════════════════════════════════
def run_streamlit_app():
    import streamlit as st

    st.set_page_config(page_title="ChurnSight", page_icon="🧠", layout="wide")
    st.title("🧠 ChurnSight — Customer Churn Predictor")
    st.markdown("Fill in the customer details below to predict churn probability.")

    col1, col2, col3 = st.columns(3)
    with col1:
        tenure         = st.slider("Tenure (months)", 1, 72, 12)
        monthly        = st.number_input("Monthly Charges ($)", 20.0, 120.0, 65.0)
        total          = st.number_input("Total Charges ($)", 100.0, 8000.0, 800.0)
        senior         = st.selectbox("Senior Citizen", [0, 1])
    with col2:
        contract       = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
        internet       = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
        tech_support   = st.selectbox("Tech Support", ["Yes","No"])
        online_sec     = st.selectbox("Online Security", ["Yes","No"])
    with col3:
        payment        = st.selectbox("Payment Method",
                             ["Electronic check","Mailed check","Bank transfer","Credit card"])
        dependents     = st.selectbox("Dependents", ["Yes","No"])
        paperless      = st.selectbox("Paperless Billing", ["Yes","No"])
        support_calls  = st.slider("Support Calls", 0, 10, 1)

    # Quick encode for prediction
    enc = {"Yes":1,"No":0,"Month-to-month":0,"One year":1,"Two year":2,
           "DSL":0,"Fiber optic":1,"No service":2,
           "Electronic check":0,"Mailed check":1,"Bank transfer":2,"Credit card":3}

    row = np.array([[tenure, monthly, total,
                     enc.get(contract,0), enc.get(internet,0),
                     enc.get(tech_support,0), enc.get(online_sec,0),
                     enc.get(payment,0), senior,
                     enc.get(dependents,0), enc.get(paperless,0),
                     support_calls]])

    df_raw  = generate_data()
    X, y, _ = preprocess(df_raw)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               stratify=y, random_state=42)
    sm = SMOTE(random_state=42)
    Xr, yr = sm.fit_resample(X_tr, y_tr)
    clf = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                        use_label_encoder=False, eval_metric="logloss",
                        random_state=42)
    clf.fit(Xr, yr)

    if st.button("🔍 Predict Churn"):
        proba = clf.predict_proba(row)[0][1]
        label = "🔴 LIKELY TO CHURN" if proba > 0.5 else "🟢 LIKELY TO STAY"
        st.metric("Prediction", label, f"{proba*100:.1f}% churn probability")
        st.progress(float(proba))


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if RUN_APP:
        run_streamlit_app()
    else:
        df                               = load_data()
        X, y, feat_names                 = preprocess(df)
        model, X_test, y_test, y_proba, _= train_and_evaluate(X, y, feat_names)
        y_pred                           = model.predict(X_test)

        plot_roc(y_test, y_proba)
        plot_confusion(y_test, y_pred)
        plot_shap(model, X_test, feat_names)
        print("\n[ChurnSight] Done! Check roc_curve.png, confusion_matrix.png, shap_summary.png")