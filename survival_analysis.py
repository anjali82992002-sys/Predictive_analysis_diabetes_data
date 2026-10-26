import os, pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

OUT = "./project2_results"; FIG = "./project2_figures"
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)

df = pd.read_csv("heart_failure/heart_failure_clinical_records_dataset.csv")
print(f"N={len(df)}  deaths={df['DEATH_EVENT'].sum()}  "
      f"censored(alive at last follow-up)={ (1-df['DEATH_EVENT']).sum() }")
# censoring note: DEATH_EVENT=0 means the patient was alive at 'time' (days) --
# they still contribute follow-up information up to that point (right-censored),
# not "no event ever."

# --- Step 8: KM curves + log-rank, two groupings ---
def km_compare(df, group_col, label_fn, title, fname):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    groups = sorted(df[group_col].unique())
    results = {}
    for g in groups:
        mask = df[group_col] == g
        kmf = KaplanMeierFitter()
        kmf.fit(df.loc[mask, "time"], df.loc[mask, "DEATH_EVENT"], label=label_fn(g))
        kmf.plot_survival_function(ax=ax)
        results[g] = (df.loc[mask, "time"], df.loc[mask, "DEATH_EVENT"])
    ax.set_title(title); ax.set_xlabel("Days"); ax.set_ylabel("Survival probability")
    fig.tight_layout(); fig.savefig(f"{FIG}/{fname}", dpi=150); plt.close(fig)
    g0, g1 = groups[0], groups[1]
    t0, e0 = results[g0]; t1, e1 = results[g1]
    lr = logrank_test(t0, t1, event_observed_A=e0, event_observed_B=e1)
    return lr.test_statistic, lr.p_value

df["ef_low"] = (df["ejection_fraction"] < 30).astype(int)
stat1, p1 = km_compare(df, "ef_low", lambda g: "EF < 30%" if g else "EF >= 30%",
                        "Survival by ejection fraction", "km_ejection_fraction.png")
print(f"KM ejection_fraction<30: logrank chi2={stat1:.2f} p={p1:.4g}")

cr_median = df["serum_creatinine"].median()
df["cr_high"] = (df["serum_creatinine"] > 1.2).astype(int)  # >1.2 mg/dL = clinically elevated
stat2, p2 = km_compare(df, "cr_high", lambda g: "Serum creatinine > 1.2" if g else "Serum creatinine <= 1.2",
                        "Survival by serum creatinine", "km_serum_creatinine.png")
print(f"KM serum_creatinine>1.2: logrank chi2={stat2:.2f} p={p2:.4g}")

pd.DataFrame([
    {"comparison": "Ejection fraction <30% vs >=30%", "logrank_chi2": stat1, "p_value": p1},
    {"comparison": "Serum creatinine >1.2 vs <=1.2", "logrank_chi2": stat2, "p_value": p2},
]).to_csv(f"{OUT}/logrank_tests.csv", index=False)

# --- Step 9: Cox PH model ---
cox_feats = ["age", "ejection_fraction", "serum_creatinine", "serum_sodium",
             "anaemia", "diabetes", "high_blood_pressure", "sex", "smoking",
             "time", "DEATH_EVENT"]
cph = CoxPHFitter()
cph.fit(df[cox_feats], duration_col="time", event_col="DEATH_EVENT")
cph.print_summary()
hr_table = cph.summary[["coef", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
hr_table.columns = ["log_hr", "hazard_ratio", "hr_ci_lower", "hr_ci_upper", "p_value"]
hr_table.to_csv(f"{OUT}/cox_hazard_ratios.csv")
print(hr_table)

# proportional-hazards assumption check
try:
    assumption_ok = True
    from io import StringIO
    import contextlib
    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        cph.check_assumptions(df[cox_feats], p_value_threshold=0.05, show_plots=False)
    assumption_report = buf.getvalue()
except Exception as e:
    assumption_report = f"check_assumptions raised: {e}"
with open(f"{OUT}/cox_ph_assumption_check.txt", "w") as f:
    f.write(assumption_report)
print(assumption_report[:2000])

# forest plot (bonus)
fig, ax = plt.subplots(figsize=(6, 4.5))
y = np.arange(len(hr_table))
ax.errorbar(hr_table["hazard_ratio"], y,
            xerr=[hr_table["hazard_ratio"] - hr_table["hr_ci_lower"],
                  hr_table["hr_ci_upper"] - hr_table["hazard_ratio"]],
            fmt="o", color="#4C72B0", ecolor="#4C72B0", capsize=3)
ax.axvline(1, color="k", ls="--", lw=0.8)
ax.set_yticks(y); ax.set_yticklabels(hr_table.index)
ax.set_xlabel("Hazard ratio (95% CI)"); ax.set_title("Cox PH model -- hazard ratios")
fig.tight_layout(); fig.savefig(f"{FIG}/cox_forest_plot.png", dpi=150); plt.close(fig)

# --- Step 10: leakage-safe DEATH_EVENT classifier (time excluded on purpose) ---
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

feats_no_leak = [c for c in cox_feats if c not in ("time", "DEATH_EVENT")]
Xd = df[feats_no_leak]
yd = df["DEATH_EVENT"]
Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.25, stratify=yd, random_state=42)
sc = StandardScaler().fit(Xtr)
clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
auc_no_leak = roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1])

Xd_leak = df[feats_no_leak + ["time"]]
Xtr2, Xte2, ytr2, yte2 = train_test_split(Xd_leak, yd, test_size=0.25, stratify=yd, random_state=42)
sc2 = StandardScaler().fit(Xtr2)
clf2 = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc2.transform(Xtr2), ytr2)
auc_leak = roc_auc_score(yte2, clf2.predict_proba(sc2.transform(Xte2))[:, 1])
print(f"\nDEATH_EVENT classifier AUC -- WITHOUT time (correct): {auc_no_leak:.3f}")
print(f"DEATH_EVENT classifier AUC -- WITH time (leakage demo): {auc_leak:.3f}")
with open(f"{OUT}/leakage_demo.txt", "w") as f:
    f.write(f"AUC without 'time' (correct, no leakage): {auc_no_leak:.3f}\n"
            f"AUC with 'time' included (leakage demo): {auc_leak:.3f}\n"
            "time (follow-up duration) encodes the outcome itself -- short follow-up "
            "time with DEATH_EVENT=1 vs long follow-up is highly informative of the "
            "label by construction, not a real predictive clinical signal. It must "
            "never be used as a feature for predicting DEATH_EVENT.\n")

