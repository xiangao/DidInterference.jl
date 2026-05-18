# DidInterference.jl

Doubly robust difference-in-differences with spatial interference, in Julia. Companion to the R package [`didint`](https://github.com/xiangao/didint).

Implements the doubly robust direct ATT estimators of:

- Xu, Ruonan (2023). "Difference-in-Differences with Interference." [arXiv:2306.12003](https://arxiv.org/abs/2306.12003).
- Xu, Ruonan (2026). "Dynamic Difference-in-Differences with Interference." *AEA Papers and Proceedings* 116: 58–63.

## Why

Standard DiD assumes one unit's outcome doesn't depend on another's treatment. With spatial spillovers — a treated municipality affecting its neighbours, a vaccine reducing transmission across the social network — the canonical DiD estimand loses its causal interpretation. Xu's framework keeps DiD's identifying logic while explicitly modelling the exposure mapping.

## Three estimators

- `did_int_2x2` — two-period, common adoption (Xu 2023, 2×2 case).
- `did_int_dynamic` — event study with common adoption timing (Xu 2026 §I).
- `did_int_staggered` — staggered adoption with not-yet-treated comparison groups, joint-IF aggregation across cells (Xu 2026 §II).

All three use the same core: three propensity-score models (cohort, treated-exposure, comparison-exposure) + two outcome-change regressions + the standard DR plug-in formula. Standard errors come from the empirical influence function. An optional `trim` argument matches Xu's 0.01 trimming used in the Brazil application.

## Install

```julia
using Pkg
Pkg.add(url = "https://github.com/xiangao/DidInterference.jl")
```

## Minimal example

```julia
using DidInterference, DataFrames

# wide-format 2-period data
res = did_int_2x2(my_panel;
    yname     = :Y_post,
    yname_pre = :Y_pre,
    treat     = :W,
    exposure  = :G,
    g         = 1,
    covariates = [:z1, :z2],
    trim       = 0.01)

println(res.estimate)   # DR direct ATT at G == 1
println(res.ci)         # 95% CI
```

## Vignettes & examples

| Resource | Description |
|---|---|
| [`examples/lattice_dgp.jl`](https://github.com/xiangao/DidInterference.jl/blob/master/examples/lattice_dgp.jl) | Runnable worked example: simulates a 2×2 lattice DGP with binary direct + spillover effects, fits `did_int_2x2` with `trim = 0.01`, runs a 100-rep Monte Carlo (bias, empirical SD, coverage), then demonstrates `did_int_staggered` on a 3-cohort 5-period lattice. Run with `julia --project=. examples/lattice_dgp.jl`. |
| [`test/runtests.jl`](https://github.com/xiangao/DidInterference.jl/blob/master/test/runtests.jl) | 12 tests across all three estimators, including the z-dependent-treatment-effect regression test that pins the paper's full-population estimand. |

The companion R package [`didint`](https://github.com/xiangao/didint) ships a full Brazil Amazon Priority List replication vignette (Xu 2026 Section III) — see [`vignettes/brazil_amazon.Rmd`](https://github.com/xiangao/didint/blob/master/vignettes/brazil_amazon.Rmd) for the worked end-to-end real-data example. The two packages use the same DR formula and the same trim / aggregation behaviour, so estimates match to MC noise.
