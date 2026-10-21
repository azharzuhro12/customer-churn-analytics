# Learning Notes — Customer Churn Analytics

Plain-language notes on what was done, why, and what could go wrong. Written so the
project owner can explain every choice in an interview.

---

## Step 1 — Preparing the data

**What I did.** Loaded the 7,043-row IBM Telco CSV, fixed `TotalCharges`, converted
`SeniorCitizen` from 0/1 to No/Yes, created `churn_flag` (1 = churned), and wrote the
result to a SQLite table called `customers`.

**Why `TotalCharges` needed fixing.** The column arrives as text because 11 customers
have a blank value instead of a number. If you skip this, any average or model that
uses the column breaks or silently treats it as a category. The blanks are not random
missing data: all 11 have `tenure = 0`, i.e. they signed up but have not been billed
yet. So the honest value is **0**, not the column average. I wrote assertions that
prove this (`blank.sum() == 11` and every blank row has `tenure == 0`) and that
`customerID` is unique, so a future data refresh cannot silently break the assumption.

**Why SQLite.** It is a single file with no server to install, it lets the analysis be
written in real SQL, and the dashboard can read the same source as the notebooks.

**What could go wrong.**
- The assertion `blank.sum() == 11` is dataset-specific. If IBM updates the file, the
  pipeline stops loudly instead of guessing — that is the intended behaviour, but the
  number must then be reviewed, not blindly changed.
- "Signed up, not billed" is an inference from `tenure = 0`. If a future dataset had a
  blank for a long-tenured customer, 0 would be wrong; the assertion catches that.
- SQLite stores dates as text and has weaker typing than Postgres. For this portfolio
  project that is fine; for production, column types and constraints should be enforced.

**Result.** 7,043 customers, 1,869 churned (26.54%), $456,116.60 monthly revenue, of
which $139,130.85 (30.50%) belongs to customers who churned.

---

## Step 2 — SQL analysis and findings

**What I did.** Wrote nine named queries in `sql/queries.sql` (headline overview,
contract, tenure, payment method, internet service, support add-ons, contract × tenure,
segment ranking, high-risk segment), ran them with `src/analysis.py`, and saved each to
`reports/tables/<name>.csv`.

**Why SQL.** The business questions are naturally "group by X" questions, and SQL is
the shared language between analysts and stakeholders. The ranking query uses
`SUM() OVER ()` and `RANK() OVER` to compute revenue share and rank without a self-join,
and `contract_by_tenure` uses a CTE to group tenure once and reuse it.

**Top findings (all from the run).**
1. **Contract is the strongest split:** month-to-month customers churn at 42.71%, one
   year at 11.27%, two year at 2.83%.
2. **Churn is front-loaded:** 47.44% of customers in their first 12 months churn, versus
   9.51% after 48 months.
3. **Payment method:** electronic check 45.29%, vs 15.24% for credit card automatic.
4. **Fiber optic:** 41.89% churn, versus 18.96% DSL and 7.40% for no internet.
5. **Support add-ons matter:** among internet customers, no TechSupport 41.64% vs 15.17%
   with it; no OnlineSecurity 41.77% vs 14.61%.
6. **The worst combination** — month-to-month + fiber optic + tenure ≤ 12 — is 916
   customers, 643 churned (70.20%), carrying $53,178.30 of monthly revenue, which is
   **38.22% of all churned monthly revenue**.
7. **One segment carries the money:** month-to-month fiber is rank 1 by churned revenue
   with $100,482.00, 72.22% of all churned monthly revenue.

**What could go wrong.**
- These are **associations, not causes**. Contract and retention may both be driven by
  customer commitment or sales-channel effects; the SQL cannot tell us which.
- Revenue here is the current `MonthlyCharges` of churned customers. It is a snapshot
  "monthly revenue at risk", not audited lost revenue over time.
- Small cells matter: the two-year 0–12 group has 68 customers and 0 churners. A 0%
  rate there is fragile and should not be read as proof of a perfect product.
- Grouping tenure into fixed bands hides the shape inside each band. The charts add
  detail, but the bands remain a simplification.

---

## Step 3 — Modelling

**What I did.** Built one sklearn `Pipeline` (scaler for numbers, one-hot encoder for
categories) and compared three classifiers with 5-fold stratified cross-validation
**on the training set only**. Then I evaluated the winner once on a 20% hold-out test
set, drew a gains curve, computed permutation importance, and saved out-of-fold risk
scores for all 7,043 customers.

**Why the pipeline.** The scaler and one-hot encoder are fitted inside each CV fold.
If I scaled the whole dataset before splitting, the training folds would "know" the
test fold's means — a subtle leak. Keeping preprocessing inside the pipeline makes that
impossible and lets the saved model accept raw input.

**Why exclude `customerID` and `gender`.** The ID is a label, not a feature; a model can
memorise it. `gender` was excluded on purpose: it is not actionable (you cannot change
it) and adds no useful signal, so including it would only invite unfair targeting.

**How the winner was chosen.** 5-fold stratified CV ROC-AUC on the training rows:

| model | CV ROC-AUC |
|---|---|
| logistic regression | 0.8462 ± 0.0125 |
| random forest | 0.8444 ± 0.0089 |
| hist gradient boosting | 0.8372 ± 0.0071 |

The differences are inside one standard deviation, so the honest conclusion is **the
models are essentially tied**, and the simplest one wins: logistic regression. This is
a common and useful result to report — the extra complexity of tree ensembles does not
pay for itself here.

**Test-set results (touched once).** ROC-AUC **0.8418**, PR-AUC **0.6325**, precision
**0.5026**, recall **0.7834**, F1 **0.6123** at the default 0.5 threshold. Confusion
matrix: 745 true negatives, 290 false positives, 81 false negatives, 293 true positives.
Because of `class_weight="balanced"`, the model leans towards catching churners: it
finds 78% of them, but half of the customers it flags do not churn.

**Gains curve.** Contacting the top 10% by risk reaches 28.58% of all churners in the
test set; the top 20% reaches **49.68%**; the top 30% reaches 64.89%. A random campaign
would reach 10/20/30%, so targeting roughly doubles the churners reached per contact at
20%.

**Permutation importance (test set, ROC-AUC drop).** `tenure` 0.1561, then
`InternetService` 0.0432, `MonthlyCharges` 0.0410, `Contract` 0.0394. Note how the
importance concentrates: only four features move the metric by more than 0.01.

**Out-of-fold scores.** `customer_scores` uses `cross_val_predict`: every customer is
scored by a model trained on the other folds, never on their own label. The dashboard
uses these scores, so a customer's displayed risk is not inflated by leakage. The same
property makes the campaign backtest honest.

**What could go wrong.**
- **The probability scale is not calibrated.** Class weights deliberately shift scores
  up: the average OOF risk is 0.4144 while the actual churn rate is 26.54%. Use the
  scores to **rank** customers, not as literal churn probabilities. Calibration
  (`CalibratedClassifierCV`) would be the next step if business needed exact rates.
- **Correlated features share importance.** `tenure` and `TotalCharges` carry overlapping
  information, so permuting one can understate the pair's joint value.
- **Permutation importance is not causality.** It says the model relies on a feature; it
  does not say changing that feature changes churn.
- **The 0.5 threshold is a choice, not a fact.** A retention team with a small budget
  would pick a higher threshold (fewer, more precise contacts); one optimising recall
  would lower it. The gains curve is the right tool to choose that threshold.
- **No time validation.** The data is a snapshot, so a random split can put "future"
  patterns in training. It measures ranking skill, not next-quarter performance.
- **CV differences are small.** Reporting 0.8462 vs 0.8444 as a "win" without mentioning
  the ±0.0125 spread would overstate the result.

## Step 4 — Charts

**What I did.** Generated five PNGs into `assets/`: churn by contract, churn by tenure,
a contract × tenure heatmap, permutation importance, and the gains curve.

**Why these choices.**
- **One blue for every single-series chart.** The bars encode magnitude by length; a
  second color would add meaning that is not there. The heatmap uses one blue ramp
  light→dark because churn rate is an ordered quantity.
- **Direct value labels.** A reader should not have to estimate a bar against a hidden
  axis, so the exact percentage sits at each bar tip.
- **No gridlines, no chart junk.** The value labels carry the numbers; gridlines would
  only add ink. Axis lines are hairline and recessive.
- **Titles state the finding**, not the chart type: "Most churn happens in the first
  year" instead of "Churn by tenure".
- **Gains curve has a legend** because it plots two series (model vs random), and the
  20% point is annotated because that is the decision-relevant threshold.

**What could go wrong.**
- The heatmap has small cells: the 0.00% two-year × 0–12 cell is only 68 customers.
  Color intensity can make a fragile cell look as solid as a large one.
- Labels are rounded to two decimals, so a chart value can differ from a raw
  recomputation in the third decimal.
- The gains curve is drawn on the **test set**, not on the OOF scores the dashboard
  uses. The two are consistent in method but not identical samples.
- A chart title is an interpretation. "15x" compares held rates, not individual risk,
  and does not prove that changing the contract would change churn.

## Step 5 — Dashboard

**What I did.** Built `app.py` with four tabs (Overview, At-risk customers, Retention
campaign backtest, Predict a customer), sidebar filters for contract, internet service
and tenure, and a shared blue visual language. Tested it with
`streamlit.testing.v1.AppTest` and captured screenshots with Playwright.

**Why these choices.**
- **Out-of-fold scores power the at-risk and campaign tabs.** `customer_scores` comes
  from `cross_val_predict`, so a customer is never ranked by a model that trained on
  their own label. That keeps the display honest without showing the raw training fit.
- **Rules for the suggested action, not a second model.** A retention team can read,
  challenge and change the rules: risk band + contract + Tech Support. A second opaque
  model would be harder to defend and no better for five actions.
- **The campaign is a transparent scenario.** Value = monthly charges of the churners
  actually reached × success rate × months kept. Net = value − cost per contact ×
  contacts. Sliders make every assumption visible, and the warning states that the
  output is not audited return.
- **Scores rank, they do not calibrate.** Class weights inflate the average score
  (0.4144 OOF vs a 26.54% churn rate), so the dashboard says "percentile", not
  "probability".
- **Empty filters stop with a warning** instead of rendering empty charts, which would
  look like zero churn rather than "no data".

**Default-view numbers from the app.**
- At the 0.50 risk threshold: **1,419 active customers**, **$106,360** monthly revenue
  at risk.
- Campaign with $25/contact, 25% success, 12 months kept: best share **55.6%**, net
  **$288,758**; at a fixed 20% contacted, net **$196,814** versus **$48,243** for random.
- Prediction form default (12 months, $70 monthly, DSL, month-to-month): score
  **0.596**, 66th percentile, action "Free Tech Support trial + review".

**What could go wrong.**
- **The "best share" is optimistic.** It is chosen by taking the maximum of the same
  curve it is reported on. Treat it as an upper bound; the fixed 20% number is a more
  honest rule.
- **The backtest assumes the future looks like this snapshot.** It reuses historical
  churn labels as if a campaign could have saved exactly those customers. Real offers
  change behaviour, and success may differ by segment.
- **Cost and success are inputs, not measurements.** Change one slider and the ranking
  of strategies can change. The dashboard says so on screen.
- **OOF scores are still from one random split scheme.** They are leakage-safe for the
  customer's own label, but they do not simulate a future time period.
- **The prediction form is not a score for a real person.** It is a what-if tool on a
  sample dataset; treating its output as a decision about an individual would be an
  ethical and statistical mistake.
- **A rule-based action is a hypothesis.** "Offer annual contract discount" is a
  reasonable retention play, but the data shows an association, not proven uplift.

**Testing.** `python -m pytest tests -q` runs five AppTest checks: load + four tabs,
sidebar filter changes the KPIs, empty filter warns and stops, risk slider changes the
count, and the prediction form submits and returns a score. All five passed. Playwright
screenshots are in `assets/dashboard_*.png`.

## Step 6 — README and verification

**What I did.** Wrote the portfolio README from the actual report files, added a
recommendations table, the model comparison and test metrics, the dashboard description
and an explicit limitations section. Then I deleted `data/churn.db`, `models/` and
`reports/`, reran `prepare → train → analysis`, and diffed the new outputs against
copies of the old ones.

**Verification result.** These files were byte-identical after the clean rerun:
`reports/metrics.json`, `reports/model_comparison.csv`,
`reports/permutation_importance.csv`, `reports/gains_curve.csv` and
`reports/tables/overview.csv`. The AppTest suite passed 5/5 after the rebuild. A scan
found no absolute paths or secrets in the repository.

**Why this matters.** A portfolio project is only credible if a reviewer can reproduce
it. Writing numbers by hand invites drift — one retrain and the README is wrong. The
diff check proves the fixed seed and the deterministic split do what they claim.

**What could go wrong.**
- **README drift.** If the model, features or cleaning change, the README must be
  regenerated or re-checked; the numbers are not auto-injected. The source of truth is
  `reports/`, not the prose.
- **Stale screenshots.** Screenshots are snapshots of the app at one moment. A UI change
  does not update them automatically; retake with Playwright after layout changes.
- **Dashboard figures are display-rounded.** `$106,360` in the README is the app's
  rounded display of `$106,359.65`; the exact value is derivable from `customer_scores`
  and `customers`. Report tables keep two decimals.
- **Generated artefacts are gitignored.** A fresh clone has no database, model or
  reports; the reviewer must run the three commands. The raw CSV is committed so the
  pipeline has an input.
- **Pinned versions.** `requirements.txt` pins the exact versions used here. A different
  Python or library version may change numerical details even with the same seed.
- **The repo is not a git repository yet.** If the owner wants a commit history, run
  `git init` and commit; `.gitignore` is already prepared.
