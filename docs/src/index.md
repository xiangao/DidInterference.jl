# DidInterference.jl

Difference-in-differences with spatial interference, in Julia. Companion to the R package
[`didint`](https://github.com/xiangao/didint).

Implements the doubly robust direct ATT estimators of:

- **Xu, Ruonan (2023).** "Difference-in-Differences with Interference."
  [arXiv:2306.12003](https://arxiv.org/abs/2306.12003).
- **Xu, Ruonan (2026).** "Dynamic Difference-in-Differences with
  Interference." *AEA Papers and Proceedings* 116: 58–63.

## Why

Standard DiD assumes one unit's outcome does not depend on another unit's
treatment. This is often the wrong assumption for spatial policies. Xu's setup
keeps the conditional-parallel-trends comparison but adds an exposure mapping
``G_{it} = G(i, W_{-it})`` for unit ``i``'s neighborhood treatment status.

## Three estimators

- [`did_int_2x2`](@ref) — two-period, common adoption (Xu 2023, 2×2 case).
- [`did_int_dynamic`](@ref) — event study with common adoption timing
  (Xu 2026 §I).
- [`did_int_staggered`](@ref) — staggered adoption with not-yet-treated
  comparison groups, joint-IF aggregation across cells (Xu 2026 §II).

All three use the same ingredients: three propensity models, two outcome-change
regressions, and the doubly robust plug-in formula. Standard errors come from
the empirical influence function. The optional `trim` argument matches Xu's
Brazil application.

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

See the vignettes in the sidebar for examples and the [Reference](@ref
API-Reference) page for the API.

The companion R package
[`didint`](https://github.com/xiangao/didint) ships a Brazil Amazon
Priority List replication vignette (Xu 2026 Section III). The two
packages use the same DR formula and trim / aggregation behaviour, so
estimates match to MC noise.
