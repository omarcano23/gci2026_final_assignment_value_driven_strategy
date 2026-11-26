# A Value-Driven Machine Learning Strategy for Company A

**GCI World 2026 - Final Assignment | Omar Cano | UCLA MASDS**

> Same commercial resources. Better decisions. Higher expected retention value.

A telecommunications consulting PoC: market and portfolio screening, customer-level EDA, feature engineering, multi-metric model selection, a frozen retrospective evaluation, an operational targeting policy, and a conditional marginal business case. The final recommendation is readiness and a controlled pilot, not commercial rollout.

## Headline evidence

| Evidence | Result | Interpretation |
|---|---:|---|
| D-028 portfolio screening | P01 score 3.15/5 | Historical priority for PoC, not an economic winner |
| D-030 Q0062 | ROC-AUC 0.692193; AP 0.683851 | Retrospective sample discrimination |
| D-030 value policy | 137.72057 vs 105.89184 UM/1K (+30.06%) | Same weighted 5% capacity, hypothetical 1% prior; exposure, not realized revenue |
| D-030 paired delta | 31.82873 UM/1K; 95% CI [24.58435, 39.33849] | All 3 preregistered criteria passed |
| D-032 annual base | Net -2,299 / -347 / +9,411 UM at 100K / 250K / 1M | Conditional scenarios, hypothetical scale, no measured profit |

## Repository map

```
notebooks/       Self-contained, executed submission notebook (.ipynb)
evidence/        Aggregate F3-F7 results; no customer-level records
economics/       D-032 original simulator, inputs, seven frozen outputs
model/           Original F5 EDA and F6 Q0062 development code/configs
operations/      Original illustrative hybrid queue allocator
tools/           Safe verification and private-input preparation
docs/            Frozen decision records, references, audit and submission checklist
requirements-notebook.txt
requirements-model.txt
SHA256SUMS.json   Release integrity manifest
```

## Quick start (no private data)

Use Python 3.13.5 where possible. The aggregate verifier uses only the standard library.

```bash
python tools/verify_release.py
python tools/verify_aggregate.py
```

The second command recreates R02 in an isolated temporary directory and compares all seven output hashes. It checks D-030 arithmetic and pass criteria from frozen aggregates; it does not reproduce D-030 from customer rows.

To open or run the notebook locally:

```bash
python -m venv .venv
# POSIX: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-notebook.txt
python -m pip install jupyterlab
jupyter lab notebooks/CompanyA_Final_Notebook.ipynb
```

Run All. The notebook embeds its required aggregate evidence and executable source, so it can also be uploaded as a standalone file to Omnicampus. The default run uses temporary storage, requires no download from this repository, and leaves the optional private-data reconstruction disabled. Charts are optional; numerical checks do not require matplotlib.

For an automated notebook run (does not overwrite the delivered notebook):

```bash
python tools/run_notebook_check.py
```

## Authorized development reconstruction (optional)

Full raw-data reproduction requires the original official assignment ZIP **and** private Plan Maestro v2.0 ZIP with the frozen split manifests. Do not upload either to GitHub. The original script hashes the source bytes and reads ID keys for filtering; only development fields are parsed. This is a new local reconstruction of the historical development recipe, not a refit of the official champion and not a reserve evaluation. The 30K reserve was consumed exactly once at D-030; no reserve evaluation script is distributed.

```bash
python -m pip install -r requirements-model.txt
python tools/prepare_private_inputs.py /absolute/path/GCI_F8_Cierre_Plan_Maestro_v2.0.zip
python private_work/diagnostico/fase5/src/eda_f5.py --source /absolute/path/official_assignment.zip --master /absolute/path/GCI_F8_Cierre_Plan_Maestro_v2.0.zip --out private_work/eda_reproduction
cd private_work/diagnostico/fase6/corte05
python receta_candidata.py /absolute/path/official_assignment.zip --rebuild-oof
```

Compare the reproduced `results/candidate05_validation.npy` SHA-256 to `model/expected_internal_prediction_sha256.txt` at repository root. The notebook's opt-in final cell provides the same route and checks. Outputs including IDs, prediction vectors and fitted models must remain private. Historic versions are pinned; transitive dependencies and OS are not fully locked. The original development reconstruction was verified during F6; this new release does **not** claim a new full training run or a new holdout evaluation. If the exact private manifests are unavailable, the default aggregate notebook still runs but the original split cannot be recreated by this release alone.

## Evidence boundaries

- **Observed:** development labels, revenue, missingness, and associations; not causal responses.
- **Confirmatory offline:** D-030 frozen retrospective gate, hypothetical label shift, not external/temporal validation.
- **Exploratory:** F7 internal deciles, profiles and operational routing; 75/25 is a proposed governance rule, not an optimum.
- **Simulated:** R02 one wave and annual case; UM currency unknown; assumptions on scale, response, margin, persistence and costs. Only differential costs cancel when genuinely equivalent.
- **Proposed:** P0 readiness, randomized P1 action/control and P2 matched-policy pilot. No measured churn reduction, P&L uplift or rollout authorization.

## Privacy and submission

Keep the repository **private** unless the course explicitly authorizes public sharing of derived results. No raw data, split IDs, individual predictions, fitted model objects or official course documents are included. The executive PDF is a separate submission, not duplicated here. No license for course data or derived evidence is granted. Do not create an open-source license for restricted materials without authorization.

The official submission is the PDF (<=15 pages), references in Omnicampus, and the reproducible notebook (.ipynb/.py or multi-file ZIP). A GitHub repo is optional, not a substitute. Rename the PDF and notebook using the exact Omnicampus account name. Add real AI chat URLs and matching references; no links fabricated. See `docs/SUBMISSION_CHECKLIST.md`, `docs/REFERENCES_OMNICAMPUS.md` and `docs/FINAL_SLIDE_MAP.md`. Earlier F9 audit and storyboard files are historical context, not the final slide map.

[Official portal](https://edu.omnicamp.us/en/courses/170/reports/125/) | [LightGBM documentation](https://lightgbm.readthedocs.io/en/stable/) | [Ascarza, Retention Futility](https://doi.org/10.1509/jmr.16.0163)
