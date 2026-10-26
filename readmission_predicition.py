import duckdb, numpy as np, pandas as pd, json, os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
                              recall_score, f1_score, confusion_matrix)
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
import shap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "./project2_results"; os.makedirs(OUT, exist_ok=True)
FIG = "./project2_figures"; os.makedirs(FIG, exist_ok=True)

MED = ["metformin","repaglinide","nateglinide","chlorpropamide","glimepiride","acetohexamide",
       "glipizide","glyburide","tolbutamide","pioglitazone","rosiglitazone","acarbose","miglitol",
       "troglitazone","tolazamide","examide","citoglipton","insulin","glyburide-metformin",
       "glipizide-metformin","glimepiride-pioglitazone","metformin-rosiglitazone","metformin-pioglitazone"]
types_sql = ", ".join(f"'{c}': 'VARCHAR'" for c in MED)
con = duckdb.connect()
con.execute(f"""CREATE TABLE d AS SELECT * FROM read_csv_auto(
    'diabetes_data_1999-2008/diabetic_data.csv', header=True, nullstr='?',
    sample_size=-1, types={{{types_sql}}})""")
df = con.execute("SELECT * FROM d").df()

# --- Step 1: features ---
df["y"] = (df["readmitted"] == "<30").astype(int)
num_feats = ["time_in_hospital","num_lab_procedures","num_procedures","num_medications",
             "number_outpatient","number_emergency","number_inpatient","number_diagnoses"]
cat_feats = ["race","gender","age","A1Cresult","max_glu_serum","change","diabetesMed",
             "admission_type_id","discharge_disposition_id","admission_source_id"]
for c in ["A1Cresult","max_glu_serum"]:
    df[c] = df[c].replace({"None": "Not tested"})
for c in cat_feats:
    df[c] = df[c].astype(str)
X = df[num_feats + cat_feats]
y = df["y"]
print(f"N={len(df):,}  readmit<30 rate={y.mean():.3f}")

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"train rate={ytr.mean():.3f}  test rate={yte.mean():.3f}")

pre = ColumnTransformer([
    ("num", StandardScaler(), num_feats),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_feats),
])
# fit preprocessing on TRAIN ONLY (leakage control), then transform both
pre.fit(Xtr)
Xtr_t, Xte_t = pre.transform(Xtr), pre.transform(Xte)
feat_names = num_feats + list(pre.named_transformers_["cat"].get_feature_names_out(cat_feats))

# --- Step 2: baselines ---
base_acc = max(yte.mean(), 1 - yte.mean())
# C=0.1 (stronger L2 regularization): a few rare discharge_disposition_id categories
# (e.g. "Expired") perfectly separate the classes, which otherwise drives
# unregularized logistic-regression coefficients toward +/-infinity
logreg = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1)
logreg.fit(Xtr_t, ytr)
logreg_proba = logreg.predict_proba(Xte_t)[:, 1]
odds_ratios = pd.Series(np.exp(logreg.coef_[0]), index=feat_names).sort_values(ascending=False)

# --- Step 3: better models (5-fold stratified CV on train) ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced",
                             random_state=42, n_jobs=-1)
rf_cv_auc = cross_val_score(rf, Xtr_t, ytr, cv=cv, scoring="roc_auc").mean()
rf.fit(Xtr_t, ytr)
rf_proba = rf.predict_proba(Xte_t)[:, 1]

# class-imbalance handling for HGB: per-sample weights (HGB has no scale_pos_weight)
spw = (ytr == 0).sum() / (ytr == 1).sum()
sw_tr = np.where(ytr == 1, spw, 1.0)
xgb_clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05,
                                          max_iter=300, random_state=42)
xgb_cv_auc = cross_val_score(xgb_clf, Xtr_t, ytr, cv=cv, scoring="roc_auc",
                              params={"sample_weight": sw_tr}).mean()
xgb_clf.fit(Xtr_t, ytr, sample_weight=sw_tr)
xgb_proba = xgb_clf.predict_proba(Xte_t)[:, 1]

# --- Step 4: evaluation ---
def eval_model(name, proba, thresh=0.5):
    pred = (proba >= thresh).astype(int)
    return {
        "model": name, "roc_auc": roc_auc_score(yte, proba),
        "pr_auc": average_precision_score(yte, proba),
        "precision": precision_score(yte, pred, zero_division=0),
        "recall": recall_score(yte, pred, zero_division=0),
        "f1": f1_score(yte, pred, zero_division=0),
        "confusion_matrix": confusion_matrix(yte, pred).tolist(),
    }

results = [
    {"model": "Baseline (predict nobody readmitted)", "roc_auc": None, "pr_auc": None,
     "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": base_acc,
     "confusion_matrix": confusion_matrix(yte, np.zeros(len(yte))).tolist()},
    eval_model("Logistic Regression", logreg_proba),
    eval_model("Random Forest", rf_proba),
    eval_model("GradBoost", xgb_proba),
]
results[1]["accuracy"] = (logreg_proba >= 0.5).astype(int).__eq__(yte).mean()
results[2]["accuracy"] = (rf_proba >= 0.5).astype(int).__eq__(yte).mean()
results[3]["accuracy"] = (xgb_proba >= 0.5).astype(int).__eq__(yte).mean()
results[2]["cv_auc_train"] = rf_cv_auc
results[3]["cv_auc_train"] = xgb_cv_auc
pd.DataFrame(results).to_csv(f"{OUT}/model_comparison.csv", index=False)
print(pd.DataFrame(results)[["model","accuracy","roc_auc","pr_auc","precision","recall","f1"]])

odds_ratios.to_csv(f"{OUT}/logreg_odds_ratios.csv")

# calibration plot (best model = xgb by roc_auc typically)
best_proba = xgb_proba if results[3]["roc_auc"] >= results[2]["roc_auc"] else rf_proba
best_name = "GradBoost" if best_proba is xgb_proba else "Random Forest"
frac_pos, mean_pred = calibration_curve(yte, best_proba, n_bins=10, strategy="quantile")
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
ax.plot(mean_pred, frac_pos, "o-", color="#DD8452", label=best_name)
ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")
ax.set_title("Calibration plot"); ax.legend()
fig.tight_layout(); fig.savefig(f"{FIG}/calibration_plot.png", dpi=150); plt.close(fig)

# ROC/PR curves
from sklearn.metrics import roc_curve, precision_recall_curve
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for name, proba in [("LogReg", logreg_proba), ("RF", rf_proba), ("GradBoost", xgb_proba)]:
    fpr, tpr, _ = roc_curve(yte, proba)
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(yte,proba):.3f})")
    p, r, _ = precision_recall_curve(yte, proba)
    axes[1].plot(r, p, label=f"{name} (AP={average_precision_score(yte,proba):.3f})")
axes[0].plot([0,1],[0,1],"k--", lw=0.8); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[0].set_title("ROC"); axes[0].legend(fontsize=8)
axes[1].axhline(yte.mean(), color="k", ls="--", lw=0.8, label=f"base rate={yte.mean():.2f}")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].set_title("Precision-Recall"); axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{FIG}/roc_pr_curves.png", dpi=150); plt.close(fig)

# --- Step 5: SHAP on GradBoost ---
explainer = shap.TreeExplainer(xgb_clf)
Xte_sample = Xte_t[:2000] if Xte_t.shape[0] > 2000 else Xte_t
sv = explainer.shap_values(Xte_sample)
shap.summary_plot(sv, Xte_sample, feature_names=feat_names, show=False, max_display=15)
plt.tight_layout(); plt.savefig(f"{FIG}/shap_summary.png", dpi=150); plt.close()

mean_abs_shap = pd.Series(np.abs(sv).mean(axis=0), index=feat_names).sort_values(ascending=False)
mean_abs_shap.head(15).to_csv(f"{OUT}/shap_top_features.csv")

top3 = mean_abs_shap.head(3).index.tolist()
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, feat in zip(axes, top3):
    idx = feat_names.index(feat)
    ax.scatter(Xte_sample[:, idx], sv[:, idx], s=4, alpha=0.4, color="#4C72B0")
    ax.set_xlabel(feat); ax.set_ylabel("SHAP value"); ax.set_title(feat, fontsize=9)
fig.tight_layout(); fig.savefig(f"{FIG}/shap_dependence_top3.png", dpi=150); plt.close(fig)

print("\nTop odds ratios (LogReg):\n", odds_ratios.head(8))
print("\nTop SHAP features (GradBoost):\n", mean_abs_shap.head(8))
print("\nDONE PART A")
