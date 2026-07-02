# Mobile Overdraft Credit Risk — Alternative-Data PD Model

**This is a portfolio ML engineering project. It uses 100% synthetic data with a controlled data-generating process (DGP). The purpose is to demonstrate a correctly implemented lending ML pipeline — proper validation methodology, calibration, fairness auditing, CI gates, and serving — not to claim production-level performance on real Kenyan borrower data.**

The DGP (`scripts/data_simulation/generate_dataset.py`) is designed to produce a realistic AUC ceiling rather than a near-perfect one. Each borrower has a latent risk factor `θ ~ N(0,1)` that is never directly observable. Every feature (`overdraw_to_inflow_ratio`, `prior_cleared_rate`, `tenure_months`, `crb_flagged`, etc.) is a noisy, partial function of `θ` with deliberately varied loading strengths — so no combination of features fully determines creditworthiness. Default outcomes are Bernoulli draws from a logistic model with an idiosyncratic shock (representing job loss, medical emergency, device theft) that no downstream model can predict. A macro regime drift shifts default log-odds upward over 2022–2024, so the OOT test window is genuinely harder than the training period. The group-membership assignment (thin-file by tenure, CRB-flagged) is partially correlated with `θ` via a tunable entanglement parameter. AUC ≈ 0.79 is the result of these design choices — the range typical of real alternative-data overdraft models — not an incidental side-effect of simpler code. The verified claim is "the pipeline correctly operates at a realistic AUC, with appropriate calibration, fairness, and temporal generalisation methodology." That is a different claim from "this model would perform this well on real borrower data," which would require fitting to a real portfolio.

The project demonstrates: alternative behavioral data feature engineering, temporal out-of-time (OOT) validation, isotonic calibration, SHAP adverse-action reason codes, disparate-impact + FPR/FNR fairness audit with significance testing, a FastAPI serving layer with calibrated threshold, CI/CD with fairness and AUC release gates, and Docker/Lambda deployment.

---

End-to-end ML system that scores 30-day default risk on mobile-money overdraft ("nano-loan") episodes using alternative behavioral data — transaction frequency, airtime spend, savings activity, repayment history — rather than traditional bureau/income data. Built for the underbanked-lending use case common in East African fintech (e.g., Safaricom's Fuliza).

## Pipeline Verification Results (synthetic data — verified 2026-07-01)

Trained on 185,626 overdraft episodes (50,000 borrowers, seed 42).
The dataset spans 2022-01-01 to 2024-06-30. Split by wall-clock time:
train on draws before 2023-07-01 (98,355 episodes), validate on 2023-07-01 to 2023-12-31 (44,946 episodes),
OOT test on 2024-01-01 onward (42,325 episodes). 20.6% of OOT-test borrowers also appear in training — expected for repeat borrowers; the split controls for time period, not borrower identity.

These numbers answer "is the pipeline correct?" against known ground truth, not "how would this perform on real data?"

### Discrimination quality (threshold-free, OOT test set)

| Metric | Value |
|---|---|
| OOT AUC | 0.7936 |
| OOT PR-AUC | 0.3019 |
| Brier score | 0.0679 |
| Actual default rate (OOT) | 8.52% |
| Train AUC @ best iteration (53) | 0.8244 |
| Val AUC @ best iteration | 0.7965 |
| Train/val gap | 0.0279 |

AUC and PR-AUC are evaluated on the OOT test window (2024). The 0.0279 train/val gap confirms the model is not memorising the training period. Best iteration at 53 trees (early stopping on val AUC) reflects the lower signal-to-noise ratio relative to the old near-deterministic DGP.

**On the Brier score:** 0.0679 vs the naive baseline of 0.0779 (always predict the OOT mean of 8.52%). The model adds genuine lift over the no-skill baseline. Isotonic calibration fitted on the validation set corrects the raw probability scale; the spread score distribution (median 0.047, p90 0.187) means calibration quality can be verified across the full range, unlike the old bimodal distribution where 70% of scores were below 0.001.

### Operating points

The threshold is a business policy decision, not a model property. Thresholds are applied to **calibrated** probability scores.

| Operating point | Threshold | Precision | Recall | F1 | Flag rate |
|---|---|---|---|---|---|
| Deployment — val min-cost (C_fp=0.20, C_fn=1.0) | 0.1743 | 0.301 | 0.463 | 0.365 | 13.1% |
| Lender-conservative (configured) | 0.30 | 0.444 | 0.207 | 0.283 | 4.0% |

At the **deployment threshold (0.1743)** the model flags 13.1% of OOT episodes (higher than the actual 8.5% default rate) with 30.1% precision and 46.3% recall. At **threshold 0.30** the flag rate drops to 4.0% but recall falls to 20.7% — the model is then highly selective but misses most actual defaults. The deployment threshold is derived from the validation set only; the test set is never consulted for threshold selection.

Both operating points are saved in `models/metrics.json`. The deployed API loads the deployment threshold (0.1743) from `metrics.json` at startup.

### Fairness (four-fifths rule + FPR/FNR, OOT test at deployment threshold 0.1743)

FPR (false positive rate) = wrong-denial rate among creditworthy borrowers. FNR (false negative rate) = missed-default rate among actual defaulters.

| Proxy group | n | n_true_negative | Approval rate | DI ratio | FPR | FNR | Passes 4/5 rule |
|---|---|---|---|---|---|---|---|
| Thin file (tenure < 12mo) | 4,889 | 4,076 | 87.9% | 1.013 | 9.28% | 55.6% | ✓ |
| Established (tenure ≥ 12mo) | 37,436 | 30,770 | 86.8% | — | 10.1% | 53.5% | — |
| CRB-flagged | 2,268 | 1,820 | 85.8% | 0.986 | 11.65% | 60.1% | ✓ |
| CRB-clean | 40,057 | 33,026 | 87.0% | — | 9.92% | 53.4% | — |

Four two-tailed tests, Bonferroni-adjusted threshold α = 0.05/4 = 0.0125. Fisher's exact test is reported alongside the z-test; both agree at these event counts (FP counts well above np≥5 in all cases).

| Test | z | p (z-test) | p (Fisher) | Bonferroni |
|---|---|---|---|---|
| Tenure FPR | −1.720 | 0.085 | 0.086 | does not pass |
| Tenure FNR | 0.760 | 0.447 | 0.455 | does not pass |
| CRB FPR | 2.546 | 0.011 | 0.013 | marginal (z-test passes, Fisher marginal) |
| CRB FNR | 1.900 | 0.057 | 0.063 | does not pass |

**CRB FPR (flagged 11.65% vs clean 9.92%):** The gap is marginally significant — the z-test passes Bonferroni correction (p=0.011 < 0.0125) but Fisher's exact test does not (p=0.013 > 0.0125). The direction is consistent with the DGP: `crb_flagged` is correlated with latent risk `θ` (fairness_entanglement=0.4), so the model correctly uses it as a predictor but this propagates systematic over-scoring to the creditworthy tail of the CRB-flagged subpopulation — approximately 240 false denials among 1,820 creditworthy CRB-flagged borrowers vs 3,637 among 33,026 CRB-clean ones (1.17× higher FPR). The DI ratio (0.986) does not trigger the four-fifths rule, but the FPR disparity is a meaningful fairness concern at the sub-population level.

**Tenure FPR (thin-file 9.28% vs established 10.1%):** Not significant (p = 0.085 z-test, 0.086 Fisher). Thin-file borrowers have a slightly lower FPR than established borrowers — the opposite of a fairness concern for this group.

**CRB FNR (flagged 60.1% vs clean 53.4%):** p = 0.057 (z-test), 0.063 (Fisher); does not survive Bonferroni correction. Marginal but directionally consistent — the TP count for CRB-flagged is only 83 borrowers, giving limited power. See "Findings Interpretation" below for analysis of why the higher-risk group has higher FNR.

**All FNR values are high (54–60%):** At the deployment threshold (0.1743) the model flags 13.1% of OOT episodes at AUC=0.79. The high FNR reflects genuine model uncertainty — the DGP's idiosyncratic shock means many defaulters score below the threshold because their shock-driven episode outcome was not predictable from observed features. See "Findings Interpretation" below.

The CI pipeline enforces the four-fifths rule as a release gate.

### Findings Interpretation

Two findings from the validation run require explicit interpretation before this model could be considered production-ready.

**FNR of 54–60% at the deployment threshold — verified modeling limitation, not a threshold configuration error**

At the deployment threshold (0.1743), overall recall is 46.3% — the model catches 46.3% of actual defaults and misses the remaining 53.7% (overall FNR). At the group level, FNR runs from 53.5% (established borrowers) to 60.1% (CRB-flagged). Under the cost ratio (C_fn=1.0, C_fp=0.20), the deployment threshold is set to minimise total expected cost; the high FNR is not a consequence of threshold selection.

**Verification — oracle FNR floor derived from the DGP:**

Since the dataset is fully synthetic we can quantify the irreducible component. The DGP generates defaults as `Bernoulli(sigmoid(risk_logit + shock))` where `shock ~ N(0, 4)` is drawn independently each episode and is unobservable to any downstream classifier (DGP line 329). The `risk_logit` includes the latent θ term (`_W_THETA * theta = 0.7 * theta`, line 319), which the trained model can only see through noisy feature proxies — it is never in the feature set.

We re-ran the DGP with the original seed (42) to extract `risk_logit` per OOT episode, then computed the Bayes-optimal score `p_oracle = E_shock[sigmoid(risk_logit + shock)]` via Monte Carlo (K=2,000 shock samples). This oracle has access to everything the model has access to *plus* the unobservable latent θ — it is the ceiling for any feature-based classifier.

Results (oracle vs trained model, OOT test period):

- **Oracle AUC: 0.806** vs model AUC: 0.794 — model captures 98.5% of the oracle's discrimination ability.
- **Oracle FNR at 13.1% flag rate: 52.7%** — the theoretical minimum FNR when all features including latent θ are known.
- **Trained model FNR: 53.75%** — 1.06pp above the oracle floor (2.0% relative gap).

Note: the oracle and model operate on slightly different episode sets (42,691 re-generated vs 42,325 from features.csv) because the original data generation and the re-simulation diverge at a later point in the RNG sequence. On the 14,052 episodes where both sets overlap exactly, the gap narrows to 0.6pp (oracle FNR 51.75%, model FNR 52.34%). The qualitative conclusion is consistent across both comparisons.

The interpretation: over 98% of the model's FNR is attributable to the idiosyncratic shock — defaults driven by unobservable life events that no combination of the available features can predict. The remaining <2% gap represents the information loss from not observing θ directly, which feature engineering improvements cannot materially close without new signal types (cross-lender debt exposure, health-payment patterns, informal network stress indicators).

**Threshold sweep — FNR across the full operating range:**

| Threshold | Flag rate | FPR | FNR |
|---|---|---|---|
| 0.05 | 47.1% | 43.7% | 16.4% |
| 0.08 | 34.5% | 30.8% | 24.8% |
| 0.10 | 24.2% | 20.5% | 35.4% |
| 0.13 | 16.7% | 13.3% | 46.8% |
| **0.1743 (deployed)** | **13.1%** | **10.0%** | **53.8%** |
| 0.20 | 8.9% | 6.4% | 63.8% |
| 0.25 | 6.4% | 4.3% | 70.9% |
| 0.30 | 4.0% | 2.4% | 79.3% |

FNR is monotonically increasing as threshold increases. At any operationally feasible flag rate (≤20%), FNR remains above 35%. To reduce FNR below 25%, the model would need to flag ≥35% of episodes and accept FPR ≥30% — operationally infeasible for a revolving nano-loan product. Threshold adjustment cannot substitute for additional predictive signal.

**This model should not be the sole credit control.** At 53.7% overall FNR, roughly half of future defaulters are approved. A layered architecture is required: (1) this model as a first-pass screen reducing the approved pool default rate from 8.52% to ~5.2%, (2) dynamic limit assignment for borderline approvals (scores 0.10–0.17) starting at 50% of assessed limit and scaling with observed repayment behaviour, and (3) behavioural triggers during the loan term to intervene before the 30-day outcome window closes. Without those additional layers, the FNR translates directly to credit losses this model was never designed to prevent alone.

**CRB-flagged group has higher FNR (60.1%) than CRB-clean (53.4%) — hypothesis and production-readiness**

The direction is counterintuitive. `crb_flagged` is a positive predictor of default risk; if the model exploits this signal it should catch a higher proportion of CRB-flagged defaults (lower FNR), not fewer. The observed gap is +6.7pp in the wrong direction.

**Hypothesis: composition effect from partial observability.** The DGP encodes CRB flagging as a noisy correlate of latent risk θ (fairness_entanglement=0.4). The model uses `crb_flagged` as a feature, inflating scores for CRB-flagged borrowers relative to CRB-clean borrowers with equivalent observable behavioural profiles. This inflation correctly catches a higher proportion of "predictable" CRB-flagged defaults — those with high latent θ whose elevated scores push them above the threshold. Removing these from the false-negative pool leaves disproportionately the "low-θ high-shock" cases: borrowers whose observable signals look acceptable but who experienced a large unobservable negative event. These constitute a larger fraction of CRB-flagged total defaults than CRB-clean total defaults, because the model was more effective at extracting the predictable component from the CRB-flagged population. The result is a higher residual FNR for that group — not because the model failed, but because it succeeded more completely at catching the predictable subset.

**Does this require investigation before production deployment?**

Yes. Two issues make this a mandatory pre-deployment item rather than an acceptable residual:

1. **Double adverse impact.** CRB-flagged borrowers face higher FPR (11.65% vs 9.92%, marginally significant, z-test p=0.011) *and* higher FNR (60.1% vs 53.4%). They are wrongly declined more often *and* when they do default they slip through the model at higher rate. The combination is the pattern most likely to attract regulatory scrutiny under Kenya's Data Protection Act and the CBK Digital Credit Providers Act 2022, which require documented differential error-rate assessment.

2. **DGP-vs-real-data ambiguity.** Under the synthetic DGP the higher CRB FNR is explained by the composition effect above. In real portfolio data the same pattern could instead indicate a **feature gap**: if CRB-flagged borrowers systematically access credit from informal or competing lenders (a dimension absent from this feature set), their default behaviour may respond to drivers the model has no signal for. Whether the gap is a composition artefact or a feature gap cannot be determined from synthetic data. A real deployment would require retrospective analysis of missed CRB-flagged defaults — specifically whether they cluster in distinct behavioural segments the current feature set provides no coverage for. If they do, additional signals (cross-lender debt exposure, informal network stress indicators) would be required before the model could be considered production-ready for that population.

### Calibration (score reliability, OOT test)

Score distribution: median OOT score 0.047, p90 = 0.187. The score is spread continuously across the full range — no longer bimodal near 0 and 1. All 10 percentile-based bins from `metrics.json` have at least 2,000 observations, so calibration quality is verifiable across the full score range.

| Score band | n | Mean predicted | Actual default rate |
|---|---|---|---|
| (−0.001, 0.0131] | 6,300 | 0.0086 | 0.0117 |
| (0.0131, 0.016] | 4,678 | 0.0160 | 0.0203 |
| (0.016, 0.0251] | 2,179 | 0.0241 | 0.0257 |
| (0.0251, 0.0282] | 5,961 | 0.0282 | 0.0332 |
| (0.0282, 0.0471] | 3,163 | 0.0432 | 0.0509 |
| (0.0471, 0.0592] | 3,576 | 0.0574 | 0.0543 |
| (0.0592, 0.083] | 5,284 | 0.0804 | 0.0757 |
| (0.083, 0.113] | 2,799 | 0.1056 | 0.1154 |
| (0.113, 0.187] | 4,604 | 0.1553 | 0.1733 |
| (0.187, 1.0] | 3,781 | 0.3366 | 0.3451 |

What this shows and what it doesn't:

- **High-score band (>0.187, n=3,781):** 33.7% predicted vs 34.5% actual — 0.8pp gap. Calibration holds well in the deployment threshold zone.
- **Low-score bands (below 0.013):** Mild over-prediction (0.86% predicted vs 1.17% actual). The model assigns non-trivial probability mass to borrowers who rarely default in this range.
- **Middle range systematic mild under-prediction** across 0.013–0.187 (actual slightly above predicted in most bins). Consistent direction but small absolute gaps.
- No calibration holes: the old DGP produced an (0.020, 0.030] bin with n=6 due to bimodal score concentration. The new score distribution fills the full range.
- The deployment threshold (0.1743) sits within the second-highest bin (0.113, 0.187] — the strongest calibration evidence is in the adjacent top decile, where 33.7% predicted closely tracks 34.5% actual, and calibration in the threshold-adjacent band is directly observable.

### Threshold grading transparency

The deployment threshold is derived from the validation set by minimising expected cost (`C_fp=0.20, C_fn=1.0`). Precision (30.1%), recall (46.3%), and flag rate (13.1%) reported in `metrics.json` are graded entirely on the OOT test set. The threshold-selection set and the grading set are disjoint. No test-set threshold search is reported — picking a threshold by searching test labels and grading at that threshold on the same labels is a selection bias, smaller than calibration leakage but structurally the same.

### Data leakage notes

- *Feature leakage (fixed):* An earlier version included `fee_to_principal_ratio` as a feature. Accumulated fees encode repayment duration — at the point of the draw decision the loan hasn't cleared yet, so the total fees aren't known. Including them inflates AUC to near 1.0. The current feature set contains only information legitimately available at the moment of the draw decision; `fee_to_principal_ratio` is intentionally excluded from `build_features.py`.
- *Group leakage (fixed):* An earlier split used stratified random sampling, allowing the same borrower to appear in both train and test. The model could memorise borrower-level patterns rather than generalising to new borrowers. The pipeline now writes `data/processed/episode_ids.csv` alongside features.
- *Temporal leakage (fixed):* `GroupShuffleSplit` prevents borrower memorisation but not period-level leakage — it validates "does this generalise to other borrowers in the same period," not "does this generalise forward in time," which is what a deployed model actually does. The pipeline now splits by wall-clock time: train on 2022–mid-2023, validate on 2023H2, OOT test on 2024. The OOT AUC of 0.7936 is evaluated on a future time window the model never saw, which is the only validation that matters for deployment. The DGP's macro regime drift (higher default log-odds in 2024) means the OOT test is a genuine generalisation test rather than a restatement of in-sample performance.
- *Prior history look-ahead (confirmed clean):* `prior_cleared_rate` and `prior_roll_rate` are derived from `prior_cleared_within_24h_count` and `prior_rolled_past_30d_count`. In the data generator (`scripts/data_simulation/generate_dataset.py` lines 379–383), rolling counters are written to the row BEFORE being incremented with the current episode's outcome. Episode N's prior counts contain only outcomes from episodes 0..N-1. Episode draw dates are assigned in strict chronological order with a minimum 30-day gap between episodes, so each outcome window closes before the next episode's priors are computed.

## Why this project exists

Most public "credit risk" portfolio projects use bureau-style datasets
(income, employment, credit history) that don't reflect how alternative-data
lending actually works for unbanked/thin-file populations. This project
specifically targets that gap: a short-tenor, mobile-first overdraft product
scored on behavioral signals, with the regulatory and fairness considerations
that kind of model attracts.

## Architecture

```
Synthetic data generator  →  Data validation gate  →  Feature engineering
        →  LightGBM PD model  →  SHAP reason codes  →  Fairness audit (4/5 rule)
        →  FastAPI serving layer  →  Docker  →  AWS Lambda (free tier)
        →  CI/CD with fairness + AUC gates  →  Drift monitoring (PSI)
```

| Stage | What it does | Where |
|---|---|---|
| Data | 185,626 overdraft episodes (2022–2024, 50k borrowers, 7.6% default rate) — committed as static source data | `data/raw/overdraft_lending_data.csv` |
| Validation | Schema, range, drift (PSI), leakage (correlation), PIT consistency, AUC gate, fairness gate — two callable entry points; fails loudly on violations | `src/validation/` |
| Modeling | LightGBM binary classifier, temporal OOT split (train 2022–2023H1, val 2023H2, OOT test 2024), early stopping, isotonic calibration | `src/models/train.py` |
| Explainability | SHAP-based reason codes per prediction — the adverse-action-notice requirement a regulated lender actually needs | `src/models/train.py`, `src/api/main.py` |
| Fairness | Disparate-impact audit (4/5ths rule) + FPR/FNR by group at deployment threshold | `src/models/train.py` |
| Serving | FastAPI `/score` endpoint returning probability, decision, and reason codes | `src/api/main.py` |
| Monitoring | PSI-based drift detection between training and live feature distributions | `src/monitoring/drift.py` |
| CI/CD | GitHub Actions: tests → train → fairness gate → AUC gate → Docker build | `.github/workflows/ci.yml` |
| Deployment | Lambda container image + API Gateway + S3, free-tier only | `infra/AWS_DEPLOYMENT.md` |

## Quickstart

```bash
pip install -r requirements.txt

# 1. Validate + build features (dataset is committed — no generation step needed)
python -m src.features.build_features

# 2. Train, evaluate, fairness-audit
python -m src.models.train

# 3. Serve locally
uvicorn src.api.main:app --reload --port 8000
# POST to http://localhost:8000/score  (see src/api/main.py for the request schema)

# 4. Run tests
PYTHONPATH=. pytest tests/ -v
```


## Data & Model Validation

The pipeline enforces a two-layer validation harness in `src/validation/`. Every layer produces a structured `ValidationReport` with per-check pass/fail results, severity levels, and machine-readable details. The entry points are:

```python
from src.validation import validate_data, validate_model
from src.features.build_features import FEATURE_SCHEMA

# Before training or serving — validate the incoming DataFrame
report = validate_data(df, schema=FEATURE_SCHEMA, reference_df=train_df)
report.assert_passed()  # raises ValidationError listing all error-severity failures

# After training — validate model quality before promoting the artifact
from src.validation import ModelValidationConfig, FairnessConfig
config = ModelValidationConfig(
    min_auc=0.65,
    fairness_configs=[
        FairnessConfig("crb_flagged", group_a_label=1, group_b_label=0),
    ],
)
report = validate_model(y_true, proba_cal, threshold=0.183, config=config)
report.assert_passed()
```

### Layer 1: Data validation (`validate_data`)

| Check | What it catches | Fails as |
|---|---|---|
| Schema / missing columns | Required columns absent | error |
| Range bounds | `utilization_rate > 1`, negative counts, etc. | error |
| Allowed values | `crb_flagged` outside `{0, 1}` | error |
| Null / missingness rate | Feature null rate above schema limit | error |
| Distribution drift (PSI) | Population shift in serving vs training data | warning (PSI > 0.10) / error (PSI > 0.25) |
| Feature-target leakage | |Pearson r| ≥ 0.95 between a feature and the target | error |
| Point-in-time consistency | `prior_cleared_count > prior_overdraw_count` (future outcomes in history) | error |
| Target rate plausibility | Default rate outside [0.5%, 50%] | error |

**Leakage check limitations — read this before concluding the model is leakage-free:**

The Pearson-correlation check flags features with |r| ≥ 0.95 against the target. This catches: (1) the target column itself accidentally included in the feature list, and (2) any single feature that is nearly a deterministic function of the outcome (e.g., `fee_to_principal_ratio` encoding repayment duration).

It does **not** catch:

- **Temporal leakage in a feature's computation window** — if a feature is derived from transactions that fall within the outcome window (e.g., a 30-day spend aggregate computed over the same 30 days in which the default is measured), the correlation may be moderate rather than extreme and will not cross the 0.95 threshold.
- **Compound leakage across multiple features** — two features that individually have moderate target correlation (r ≈ 0.6 each) could together reconstruct the outcome near-perfectly; Pearson correlation on individual features cannot detect this.
- **Indirect leakage via monotone transformations** — a log, sigmoid, or rank transformation of a leaking feature reduces the raw correlation; depending on the transformation, the feature could evade the 0.95 threshold.
- **Category-level or look-ahead leakage in group aggregates** — target encoding, group-level mean imputation, or fold-aware aggregations computed without proper train/test isolation.

The **point-in-time consistency check** (`prior_cleared_count ≤ prior_overdraw_count`) provides a structural invariant check specifically for the rolling history counters. It catches impossible look-ahead values in those columns but cannot generalise to arbitrary features.

**For complete leakage assurance, manual code review of every feature derivation pipeline is required.** The correlation and PIT checks reduce the risk of obvious leakage but do not guarantee leakage-freedom.

### Layer 2: Model validation (`validate_model`)

| Check | What it catches | Fails as |
|---|---|---|
| AUC gate | OOT AUC below 0.65 (minimum viable discrimination) | error |
| Train/OOT gap | Gap > 0.15 (memorising training period, not generalising) | warning |
| Disparate impact (4/5 rule) | Approval rate ratio below 0.80 for any configured group | error |
| FPR / FNR ratio | Group FPR or FNR ratio above 2.0× | warning |
| Forbidden features | Named leaking or disallowed features in the model's feature list | error |

Every training run writes `models/validation_report.json` with the full structured report. The CI pipeline enforces the AUC gate and the four-fifths rule as hard release gates independently of this validator.

### Cost-ratio threshold selection

The deployment threshold is selected by minimising expected cost on the validation set rather than by maximising F1. F1 implicitly assigns equal cost to false positives and false negatives — a wrong assumption for lending, where missing a defaulter typically costs far more than declining a creditworthy borrower.

```
threshold* = argmin_t  C_fp × n_FP(t) + C_fn × n_FN(t)

Under a well-calibrated model, this empirical minimum converges to:
threshold* = C_fp / (C_fp + C_fn)   [Bayes-optimal decision rule]
```

**Deriving C_fp and C_fn from Fuliza-style nano-loan unit economics:**

*C_fn = 1.0 — cost of approving a borrower who defaults (normalised to principal)*

A defaulted nano-loan is listed to CRB and written off. Mobile lenders without collateral typically recover 0–20% of principal through collection enforcement. Assumption A1: **LGD = 100% (zero recovery)** — the conservative upper bound. Normalised to principal = 1: C_fn = 1.0.

*C_fp = 0.20 — cost of declining a creditworthy borrower*

Built up from three components:

1. **Per-draw net margin** — Fuliza charges a tiered daily fee (approximately 0.5–1% per day documented in Safaricom's product terms). After subtracting funding cost (~18% p.a. for mobile money lenders in Kenya), net margin per draw is approximately **1.5% of principal** (A2).

2. **Expected remaining draws per loyal customer** — Fuliza is a revolving facility; repeat users typically draw 3–5 times per year. Assuming a 3-year customer relationship, a loyal user takes approximately **13 draws over their lifetime** (A3: 4 draws/year × 3 years ≈ 13).

3. **Churn probability on a wrong declination** — a single false declination does not guarantee the borrower leaves, but in a competitive mobile-money market (M-Pesa vs Airtel Money vs others), incorrect denials increase churn. Assumption A4: **100% churn** — all 13 remaining draws are permanently foregone. This is a deliberate conservative upper bound; the true churn probability per false decline is probably 20–40%, but using 100% makes C_fp larger and produces an operationally feasible threshold.

C_fp = 13 draws × 1.5% net margin = **0.195 ≈ 0.20** (normalised to principal).

**Resulting threshold:** 0.20 / (0.20 + 1.0) = **0.167** (Bayes-optimal). Val-set empirical minimum lands at 0.1743 — close agreement, consistent with adequate calibration.

**Flag rate at empirical threshold 0.1743:** 13.1% of OOT episodes. Operationally feasible, unlike the 85% flag rate produced by per-episode-only cost (C_fp = 0.015 at 1.5% margin with no lifetime value).

**Assumptions are labelled explicitly:** A1 (100% LGD), A2 (1.5% net margin/draw), A3 (13 lifetime draws), A4 (100% churn on false decline). None of these are fitted to a real portfolio — they are order-of-magnitude estimates consistent with publicly documented Fuliza product mechanics. A real deployment would replace these with observed portfolio statistics. The framework is correct; the cost parameters are calibrated to produce a tractable threshold, not claimed to be precise.

## Data & methodology

The dataset contains 185,626 overdraft episodes across 50,000 borrowers
(2022–2024 observation window, 7.6% 30-day default rate overall; 8.5% in the 2024 OOT window due to macro regime drift).

Two levels of features reflect how a Fuliza-style product actually works:

1. **Borrower-level signals** — tenure, transaction volume, savings activity,
   CRB flag status, voice/data usage — calibrated against publicly documented
   factors Safaricom has stated drive Fuliza limit assignment (usage volume,
   repayment timeliness, savings habits, CRB status — *not* income or
   employment data).
2. **Episode-level signals** — draw amount, utilization rate, overdraw-to-inflow
   ratio, prior repayment history — the information present at the moment of
   the draw decision.

The target (`defaulted_30d = 1`) is defined as failing to clear the outstanding
balance within the 30-day window before CRB-listing risk escalates.

**Known limitation:** the dataset is calibrated against publicly documented
product mechanics, not fitted to a real lender's actual portfolio statistics.
Risk relationships (e.g., CRB-flag → higher default rate) are directionally
correct but the exact magnitudes are illustrative. State this plainly if asked
in an interview — overclaiming data realism is the single biggest credibility
risk in this project.

## Fairness considerations

Alternative-data credit scoring is *more* fairness-sensitive than traditional
scoring, not less — behavioral/social-style features can become regulatory-
risky proxy-discrimination vectors if unaudited. This project treats that as
a first-class concern, not a footnote:

- `tenure_months` and `crb_flagged` are used as proxies for thin-file /
  new-to-system borrowers (the population most exposed to unfair outcomes)
- Every training run computes a disparate-impact ratio against the
  four-fifths rule for both proxy groups and writes `models/fairness_report.json`
- The CI pipeline **fails the build** if either group violates the four-fifths
  rule — fairness is a release gate, not a dashboard nobody looks at

A real deployment would audit against actual protected-class data subject
to local regulatory requirements (Kenya's Data Protection Act, CBK Digital
Credit Providers Act 2022), not these proxies alone.

## Limitations (stated explicitly, not hidden)

- **Synthetic data — this is the headline limitation.** AUC 0.79 comes from a DGP with latent risk, idiosyncratic shocks, and regime drift — not from training on real borrowers. The pipeline is verified to correctly operate at this AUC under realistic but synthetic conditions. Metric values should not be extrapolated to real Kenyan mobile-money data without fitting to an actual portfolio.
- **Threshold cost parameters are derived from assumptions, not a fitted portfolio.** The Bayes-optimal decline rule is: decline when `P(default | score) >= C_fp / (C_fp + C_fn)`. `C_fp=0.20` is derived from 13 lifetime draws × 1.5% net margin with 100% churn assumed; `C_fn=1.0` assumes 100% LGD and zero recovery (see the "Cost-ratio threshold selection" section for full derivation and labelled assumptions). **This is structurally better than F1 maximisation** (which assigns equal FP/FN cost — wrong for lending), but the specific parameter values are order-of-magnitude estimates calibrated to produce a tractable threshold, not fitted to observed portfolio loss data. A real deployment must replace A1–A4 with observed statistics. The 0.015 threshold from per-episode cost alone (C_fp=0.015) has 85.1% flag rate — operationally infeasible. Multi-draw lifetime value is load-bearing in constructing a usable threshold. Cross-lender debt exposure (unobservable in this dataset) is also a major omitted predictor.
- No model registry, auth, or audit logging in the serving layer — production-readiness gaps
- No human-review path for declined applications, which most regulatory frameworks require
- Drift monitoring uses PSI only; a production system would pair this with outcome-based monitoring (actual default rates vs. predicted), which requires a feedback loop this demo doesn't have
- Single-region, single-environment deployment — no staging/prod separation
- **None of this rigor transfers automatically to real transaction data.** The DGP is calibrated against documented Fuliza product mechanics, not fitted to a real lender's portfolio statistics. Risk relationships (e.g., overdraw-to-inflow → default, CRB-flag → higher FPR) are directionally correct but exact magnitudes are illustrative. Actual alternative-data models trained on real borrower data will differ in feature importance ranking, calibration curve shape, and fairness disparity magnitudes — sometimes materially. What has been validated here is that the pipeline methodology is correct against controlled ground truth.

## Tech stack

Python · LightGBM · SHAP · FastAPI · Docker · AWS Lambda · API Gateway ·
S3 · EventBridge · GitHub Actions · pytest

## License

MIT — see [LICENSE](LICENSE). Use freely; just don't claim this is real
lender data.
