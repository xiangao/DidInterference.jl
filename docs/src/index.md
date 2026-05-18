# DidInterference.jl

Doubly robust difference-in-differences with spatial interference, in
Julia. Companion to the R package
[`didint`](https://github.com/xiangao/didint).

Implements the doubly robust direct ATT estimators of:

- **Xu, Ruonan (2023).** "Difference-in-Differences with Interference."
  [arXiv:2306.12003](https://arxiv.org/abs/2306.12003).
- **Xu, Ruonan (2026).** "Dynamic Difference-in-Differences with
  Interference." *AEA Papers and Proceedings* 116: 58–63.

## Why

Standard DiD assumes one unit's outcome does not depend on another's
treatment (SUTVA). With spatial spillovers — a treated municipality
affecting deforestation in neighbouring ones, a vaccinated household
reducing transmission to nearby households, a labour-market policy in
one commuting zone changing flows to its neighbours — the canonical
DiD estimand loses its causal interpretation. Xu's framework keeps the
identifying logic of conditional parallel trends but adds an exposure
mapping ``G_{it} = G(i, W_{-it})`` for unit ``i``'s neighbourhood
treatment status.

## Three estimators

- [`did_int_2x2`](@ref) — two-period, common adoption (Xu 2023, 2×2 case).
- [`did_int_dynamic`](@ref) — event study with common adoption timing
  (Xu 2026 §I).
- [`did_int_staggered`](@ref) — staggered adoption with not-yet-treated
  comparison groups, joint-IF aggregation across cells (Xu 2026 §II).

All three share a single doubly robust core: three propensity models
(cohort, treated-exposure, comparison-exposure) plus two
outcome-change regressions, combined with the standard DR plug-in
formula. Standard errors come from the empirical influence function.
An optional `trim` argument matches Xu's 0.01 trimming in the Brazil
application.

## Install

```julia
using Pkg
Pkg.add(url = "https://github.com/xiangao/DidInterference.jl")
```

## At a glance

```julia
using DidInterference, DataFrames

res = did_int_2x2(my_panel;
    yname      = :Y_post,
    yname_pre  = :Y_pre,
    treat      = :W,
    exposure   = :G,
    g          = 1,
    covariates = [:z1, :z2],
    trim       = 0.01,
)
println(res.estimate)   # DR direct ATT at G == 1
println(res.ci)         # 95% CI
```

See the **Vignettes** in the sidebar for runnable end-to-end examples,
and the [Reference](@ref API-Reference) page for the full API.

The companion R package
[`didint`](https://github.com/xiangao/didint) ships a Brazil Amazon
Priority List replication vignette (Xu 2026 Section III). The two
packages use the same DR formula and trim / aggregation behaviour, so
estimates match to MC noise.
