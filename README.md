# Predictive Modelling & Survival Analysis
## 1. Use Case

Project 2 is the **modelling** counterpart to Project 1's descriptive/statistical work. It has two
independent halves built on two **different datasets**, deliberately chosen to cover the two
canonical supervised-learning shapes in clinical data:

| | Part A | Part B |
|---|---|---|
| **Question** | Can we **predict** which patients will be readmitted within 30 days? | **When** do heart-failure patients die, and what drives that hazard? |
| **Problem type** | Binary classification, heavily imbalanced | Time-to-event / survival analysis with censoring |
| **Dataset** | UCI Diabetes 130-US hospitals (same CSV as Project 1) | UCI Heart Failure Clinical Records |
| **N** | 101,766 encounters | 299 patients |
| **Outcome** | `readmitted == '<30'` (11.2% positive) | `DEATH_EVENT` (96 deaths, 203 censored) |
| **Core methods** | LogReg / RandomForest / GradientBoosting + SHAP | Kaplan-Meier, log-rank, Cox PH |

### 1.1 Why two parts

The project exists to demonstrate that **"predicting an event" and "predicting the time to an
event" are different problems requiring different machinery** — and that using classification
tooling on survival data (or vice versa) produces specific, nameable failures. Part B's Step 10
makes this literal by **deliberately demonstrating the leakage** that occurs when follow-up time is
fed to a classifier.

### 1.2 Headline findings

**Part A:** All three models land around **ROC-AUC ≈ 0.68** — a modest but real signal.
The 88.8%-accuracy "predict nobody is readmitted" baseline is **more accurate than every real
model** while being clinically worthless, which is the entire point of reporting **PR-AUC**
(≈ 0.22 vs. an 11.2% base rate) and **recall** (0.56–0.63) instead of accuracy.
`number_inpatient` is the dominant predictor by SHAP — **independently confirming Project 1's
finding** that it was the only variable with a non-trivial effect size.

**Part B:** Both clinical stratifications are strongly significant (ejection fraction < 30%:
log-rank p = 2.6e-09; serum creatinine > 1.2: p = 6.0e-08). The Cox model finds four significant
covariates: `age` (HR 1.047/yr), `ejection_fraction` (HR 0.954 per unit — **protective**),
`serum_creatinine` (HR 1.368), and `high_blood_pressure` (HR 1.564). The leakage demo shows AUC
jumping **0.752 → 0.864** purely from including `time`.
