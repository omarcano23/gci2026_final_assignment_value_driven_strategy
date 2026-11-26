# F9.1 | Storyboard proposal — for approval

Three narratives evaluated: model-first (technically clear but weak business selection); ROI-first (executive but risks implying measured profit); decision-first (recommended, connects market, portfolio, EDA, model, execution and conditional value). The 15 slides below are proposed, not final.

## 01 — Same commercial resources. Better decisions.
**Objective:** Establish thesis and conditional ask.  
**Evidence:** PoC D-030; no commercial effect measured.  
**Visual:** Two matched decision paths.  
**Transition:** Why now?

## 02 — A growing wireless market is under pressure to retain value
**Objective:** Explain industry and customers.  
**Evidence:** GSMA 5.8B unique subscribers; CTIA US 469→603M connections; pricing -4.1%; AT&T 0.90%, T-Mobile 0.93% 2025 monthly postpaid phone churn; competition/portability; reference only.  
**Visual:** Small trend + peer churn + competitive pressures.  
**Transition:** Where can Company A act?

## 03 — Seven opportunities compete; retention is the strongest testable PoC
**Objective:** Show evidence-led selection not predetermined churn.  
**Evidence:** F3 coverage, historical D-028; P02–06 marginal ND, P04 challenger, P07 not eligible; unchanged 70% diagnostic.  
**Visual:** Seven-row screen with available outcomes, matched comparator, historical decision.  
**Transition:** What does the selected data show?

## 04 — Risk and customer value are not concentrated in the same places
**Objective:** Client-specific EDA and quality.  
**Evidence:** 70K dev; 49.56% sample labels; 96.56% cells, 26.58% complete rows; lifecycle, decline, failure, overage; revenue-only captures 17.22% raw exposure.  
**Visual:** 3 small association bars + risk vs exposure matrix.  
**Transition:** Which decision should we improve?

## 05 — Predict future churn to rank scarce outreach slots
**Objective:** Define unit, objective, target and KPI.  
**Evidence:** Days 31–60; one customer record; historic revenue proxy; exposure per 1K technical eligible; no causal label; unknown dates.  
**Visual:** Timeline and target/policy diagram.  
**Transition:** How do we learn safely?

## 06 — Build features from business mechanisms, not identifiers
**Objective:** Explain feature engineering and controls.  
**Evidence:** Reconstructed trends, ratios, minimum denominator, lifecycle, ambiguity, train-only transforms; exact feature sets in recipe.  
**Visual:** Raw to engineered examples and excluded features.  
**Transition:** Which model best serves the decision?

## 07 — Select by economic targeting performance, not AUC alone
**Objective:** Model competition and validation design.  
**Evidence:** 52.5K train/17.5K internal/30K protected; 5 folds, OOF Platt; Q0062 0.8 P072 + 0.2 expert; practical tie; B4 logistic, boosting.  
**Visual:** Split diagram + ensemble and selection rationale.  
**Transition:** Does the locked model improve decisions?

## 08 — Q0062 captures 30.06% more exposure at the same capacity
**Objective:** Present confirmatory gate and exact model metrics.  
**Evidence:** D-030: 29,904 eligible; top5 weighted at 1%; B1 105.89 vs Q0062 137.72; delta 31.83 CI [24.58,39.34]; AUC .6922 AP .6839 vs B4 .5954/.5759; B4 value 115.36; risk recall tradeoff.  
**Visual:** Bar chart with labels, delta and 3/3 gate.  
**Transition:** How should we allocate broader capacity?

## 09 — Broader coverage has diminishing returns
**Objective:** Translate exploratory deciles into productivity.  
**Evidence:** F7 internal: top20 54.7% exposure recall vs 43.9%; +24.70% exposure; one-wave simulation +19.07 UM/1K margin and -45.3% contact cost/save.  
**Visual:** Cumulative decile curve + one-wave inset.  
**Transition:** How will commercial teams use this?

## 10 — Embed targeting in the existing commercial operating model
**Objective:** Explain matched policies and action orchestration.  
**Evidence:** 75/25 hybrid, identical business lane; billing/service/lifecycle/disengagement; CRM, suppression, channel/minute constraints; proposed not optimized.  
**Visual:** Two matched swimlanes + action examples.  
**Transition:** What incremental value is possible?

## 11 — Incremental margin is positive in the base scenario; net value depends on scale
**Objective:** Show central annual case under matched resources.  
**Evidence:** 100K/250K/1M margin 1301/3253/13011; Analytics 3600; net -2299/-347/+9411 UM; NPV -2289/-470/+8629; 15% response, 55% margin, 6mo, 1% prior.  
**Visual:** Diverging net bars + margin/cost bridge.  
**Transition:** How robust is the investment?

## 12 — Investment viability depends on response, persistence and own cost
**Objective:** Show downside and decision threshold.  
**Evidence:** 100K conserv/base/favorable net -3574/-2299/+7800; NPV break-even ~289K conditional; 160 build + 80 run hours ×15 UM; no guaranteed scale; 36m separate if space.  
**Visual:** Scenario bars + cost cap/break-even.  
**Transition:** What evidence is still missing?

## 13 — Test action effectiveness and targeting separately
**Objective:** Explain causal validation and risk.  
**Evidence:** P0 readiness, P1 action vs control, P2 real BAU vs matched policy randomize pre-ranking; ITT margin per eligible/capacity; actual costs, permissions, power.  
**Visual:** Staged pilot diagram with gates.  
**Transition:** What should directors authorize?

## 14 — Authorize readiness and a measured pilot, not rollout
**Objective:** Close with decision and ownership.  
**Evidence:** Analytics PoC pass; real BAU, margin, dates, response, capacity unknown; commercial/finance/IT/privacy owner gates; stop if no positive marginal NPV.  
**Visual:** Decision tree and roadmap.  
**Transition:** Evidence and reproducibility

## 15 — References and reproducibility
**Objective:** Meet GCI requirements.  
**Evidence:** Dataset & guide; CTIA/GSMA/SEC/FCC; ACCA; Ascarza; methods and actual AI URLs in submission field; repo recipe and frozen aggregates.  
**Visual:** Compact reference list; inline source tags on slides.  
**Transition:** End

