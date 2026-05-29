# Design — Multiplicative (Poisson) Doubly-Robust DiD with Interference

**Date:** 2026-05-29
**Package:** `DidInterference.jl` (`~/projects/software/DidInterference.jl`)
**Status:** Design approved; pending spec review → implementation plan
**Motivation:** count outcomes (e.g. crime incidents per cell×month) are poorly served by
the additive `log(1+y)` estimand — zeros make it non-scale-invariant and uninterpretable
(Chen & Roth 2024). Add a multiplicative (rate-ratio) doubly-robust estimand on the
exponential/Poisson scale, preserving the package's interference exposure mapping and
double robustness.

---

## 1. Estimand & identifying assumption

Counts `Y_it ≥ 0`; own treatment `W∈{0,1}`; exposure `G` (integer); target level `g`;
covariates `Z`; pre-period count `Y_pre`.

The existing estimators assume **additive** parallel trends on `ΔY = Y_post − Y_pre`. The
new path assumes **multiplicative (ratio) parallel trends**:

> `E[Y_post(0) | W, G=g, Z] = E[Y_pre | W, G=g, Z] · ρ(g, Z)`

In words: absent its own treatment, a unit's expected count grows by a factor `ρ(g,Z)`
that may depend on exposure and covariates but **not** on own treatment `W`. This is the
exponential/Poisson common-trends assumption (Wooldridge 2023).

**Estimand — multiplicative ATT at exposure `g`:**

> `θ(g) = E[Y_post(1) | W=1,G=g] / E[Y_post(0) | W=1,G=g] − 1`

A percentage effect on the treated-at-`g`: scale-invariant, defined with zeros, directly
interpretable ("θ×100% more/fewer events"). Reported as `θ(g)` (and `log(1+θ)` internally
for aggregation).

## 2. Estimator (the core)

Write `θ(g) = N/D − 1`:

- **Numerator** `N = E[Y_post | W=1, G=g]` — the treated-at-`g` mean; identified directly.
- **Denominator** `D = E[Y_post(0) | W=1, G=g]` — the counterfactual; carries the DR machinery.

**Outcome nuisance.** `μ0(X) = E[Y_post | W=0, G=g, X]`, `X=(Z, Y_pre)`, fit by **Poisson
QMLE** (GLM, log link) on the control-at-`g` cells.
- Poisson QMLE is consistent for the conditional mean under any true distribution
  (linear-exponential-family / GMT robustness).
- `Y_pre` enters as a **covariate, not an offset**, so `Y_pre = 0` cells are fine (no
  `log 0`). This is the deliberate zero-handling choice.

**DR counterfactual mean** (AIPW for `E[Y(0)|treated]`), reusing the existing treatment
propensity `p = P(W=1|·)` and exposure propensities `π_{wg}` (same models as `_dr_atte`):

> `D̂ = (1/n_1g) Σ_i 1{G=g} [ W·μ0(X_i) + (1−W)·(p/(1−p))·(Y_post,i − μ0(X_i)) ]`

`D̂` is consistent if **either** `μ0` **or** the propensity model is correct — double
robustness, preserved on the multiplicative scale.

**Inference.** `θ = N/D − 1`; delta-method influence function

> `IF_θ = (1/D)·IF_N − (N/D²)·IF_D`

where `IF_N`, `IF_D` are the empirical influence functions of the two means (the DR-mean IF
for `D`, the simple-mean IF for `N`, both restricted to `G=g` with the exposure-propensity
factors as in the additive engine). Neyman orthogonality ⇒ the plug-in IF (nuisances treated
as known) is first-order valid, matching the additive engine's treatment of SEs. SE = sqrt
of summed squared empirical IF.

**Staggered aggregation.** Aggregate per-`(c,t)` cells on the **log scale**:
`log(1+θ_ct) = log N_ct − log D_ct`, using the **same per-cell weights as the additive
aggregator** (`n_total`-proportional). Stack the per-cell log-ratio IFs with the existing
joint-IF machinery (linear in per-cell IFs, so it carries over unchanged), then exponentiate
the weighted aggregate. CIs are formed on the log scale and exponentiated. The `event_time`
and `cohort` sub-aggregates are formed the same way.

**Known risk.** Correctness of the IF + log-scale aggregation. Mitigated by the §4 Monte
Carlo (bias/coverage). Thin exposure strata can still destabilise the Poisson fit — handled
downstream by cohort coarsening, exactly as in the additive case.

## 3. API & integration

- New keyword on **all three** public functions: `family = :gaussian` (default — current
  additive behaviour, fully backward-compatible) vs `family = :poisson`.
- Internally dispatch to the current `_dr_atte` (additive) or a new
  `_dr_atte_mult` in `src/dr_atte_mult.jl`.
- No new public function names → clean parity with the R `didint` API later.
- Returned NamedTuple keeps its field names; under `:poisson`, `estimate`/`ci` carry the
  multiplicative ATT (`θ`). Add `family`/`scale` to the returned object so downstream code
  knows the scale.

**Files**
| File | Change |
|---|---|
| `src/dr_atte_mult.jl` | NEW — multiplicative DR engine: Poisson-QMLE `μ0`, AIPW `D̂`, `N̂`, `θ`, IF, SE, CI, counts. |
| `src/did_int_2x2.jl` | add `family` kw; dispatch; return scale. |
| `src/did_int_dynamic.jl` | add `family` kw; per-period multiplicative; dispatch. |
| `src/did_int_staggered.jl` | add `family` kw; per-cell multiplicative + log-scale joint-IF aggregation. |
| `src/DidInterference.jl` | include new file. |
| `test/runtests.jl` | multiplicative smoke tests + MC bias/coverage (parallel). |
| `docs/src/...` | reference `@example` entries + a short multiplicative vignette. |

## 4. Validation (mandatory, parallelized)

Monte Carlo against a **multiplicative + spillover count DGP**:

> `Y_post ~ Poisson(exp(α + βZ + δ·W + ψ·G·W + trend))`, with a known true `θ(g=1)`.

- Check **bias ≈ 0** and **95% CI coverage ≈ 0.95**, first for `did_int_2x2`
  (clean closed-form truth), then `did_int_staggered`.
- Reps parallelized (`Threads.@threads` / `pmap`) — standing rule: MC always parallel.
- Acceptance: |bias| within MC noise of 0; coverage in ~[0.92, 0.97]. If coverage is off,
  the IF/aggregation is wrong → fix before reporting any applied number.

## 5. Downstream (separate, after the method is validated)

Rerun the dispensary paper's spillover and direct arms with `family = :poisson`; report the
multiplicative ATT alongside the existing `log(1+y)` numbers; assess whether the
displacement signal survives the count model.

## Scope guards (YAGNI)

- **No R `didint` port this round** — noted as a follow-up.
- **No new public function names** — `family` keyword only.
- Dynamic-event-study wrapper included only because full parity across the three estimators
  was chosen.
- No refactor of the additive engine beyond the dispatch hook.

## References

- Xu, Ruonan (2023). "Difference-in-Differences with Interference." arXiv:2306.12003.
- Xu, Ruonan (2026). "Dynamic Difference-in-Differences with Interference." *AEA P&P* 116.
- Wooldridge, J. (2023). "Simple approaches to nonlinear difference-in-differences with
  panel data." *Econometrics Journal*.
- Chen, J. & Roth, J. (2024). "Logs with zeros? Some problems and solutions." *QJE*.
- Santos Silva & Tenreyro (2006). "The log of gravity." *REStat* (PPML consistency).
- Sant'Anna & Zhao (2020). "Doubly robust difference-in-differences estimators." *J. Econometrics*.
