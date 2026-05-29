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

## 2. Estimator (the core) — DR ratio-of-ratios

The multiplicative ATT at exposure `g` is the **ratio of growth ratios**:

> `θ(g) = [ E[Y_post|W=1,g] / E[Y_pre|W=1,g] ] / [ E[Y_post|W=0,g] / E[Y_pre|W=0,g] ] − 1`

In words: how much faster crime grew where a unit got its own treatment than where it
didn't, at exposure `g`. **Why this is right and the lagged-outcome design is not:** the
treated growth ratio uses the *same* treated-at-`g` units in numerator and denominator, so
the latent baseline `E[exp(α+βZ)|·]` cancels — no need to condition on the noisy count
`Y_pre`, hence no lagged-DV / mean-reversion bias. Group means stay positive even when
individual `Y_pre = 0`, so zeros are handled.

**Four AIPW means.** Reuse the existing treatment propensity `p=P(W=1|Z)` and exposure
propensities `π_1g, π_0g` (same models as `_dr_atte`). Add **four Poisson-QMLE outcome
models** on `Z` (log link, consistent for the mean under any distribution — GMT robustness):
`m_post1, m_pre1` fit on `(W=1,Ig=1)`; `m_post0, m_pre0` fit on `(W=0,Ig=1)`. Then per unit
`i` with `Ig=1{G=g}`:

```
ā_post1 = mean_i Ig·[ m_post1(Z) + W/(p·π1g)·(Y_post − m_post1(Z)) ]        → E[Y_post(1)|g]
ā_pre1  = mean_i Ig·[ m_pre1(Z)  + W/(p·π1g)·(Y_pre  − m_pre1(Z))  ]        → E[Y_pre |W=1,g]
ā_post0 = mean_i Ig·[ m_post0(Z) + (1−W)/((1−p)·π0g)·(Y_post − m_post0(Z)) ]→ E[Y_post(0)|g]
ā_pre0  = mean_i Ig·[ m_pre0(Z)  + (1−W)/((1−p)·π0g)·(Y_pre  − m_pre0(Z))  ]→ E[Y_pre |W=0,g]
```

Each `ā` is a standard AIPW mean — consistent if **either** its outcome model **or** the
propensity is right (double robustness, preserved on each of the four means). Then

> `θ(g) = (ā_post1 · ā_pre0) / (ā_pre1 · ā_post0) − 1`.

**Inference (log scale).** `ℓ = log ā_post1 − log ā_pre1 − log ā_post0 + log ā_pre0`
(`= log(1+θ)`). Each `ā` has a per-unit AIPW influence function `ψ^a_i` (= its bracketed
summand minus `ā`). By the delta method the per-unit IF of `ℓ` is

> `IF_i = ψ_i^post1/ā_post1 − ψ_i^pre1/ā_pre1 − ψ_i^post0/ā_post0 + ψ_i^pre0/ā_pre0`.

Neyman orthogonality ⇒ the plug-in IF (nuisances treated as known) is first-order valid, as
in the additive engine. `SE_ℓ = sqrt(Σ IF_i² / n²)`; report `θ = exp(ℓ)−1` with CI
`exp(ℓ ± z·SE_ℓ) − 1`.

**Staggered aggregation.** Aggregate per-`(c,t)` cells on the **log scale**:
`ℓ_ct = log(1+θ_ct)` (the per-cell log ratio-of-ratios), using the **same per-cell weights as the additive
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
