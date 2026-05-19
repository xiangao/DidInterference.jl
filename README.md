# DidInterference.jl

[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://xiangao.github.io/DidInterference.jl/)

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

## Documentation & vignettes

Full documentation: **<https://xiangao.github.io/DidInterference.jl/>**

| Page | Description |
|---|---|
| [Home](https://xiangao.github.io/DidInterference.jl/) | Overview, install, motivation |
| [Getting Started](https://xiangao.github.io/DidInterference.jl/dev/vignettes/01_getting_started/) | 2×2 base case on a synthetic lattice DGP, single fit + 100-rep Monte Carlo for bias/coverage |
| [Staggered Adoption](https://xiangao.github.io/DidInterference.jl/dev/vignettes/02_staggered/) | Multi-cohort DR DATT with joint-IF aggregation and per-cohort/event-time aggregates |
| [Reference](https://xiangao.github.io/DidInterference.jl/dev/reference/) | Full API. Each function has its docstring followed by a live `@example` block showing real output |

The companion R package [`didint`](https://github.com/xiangao/didint) ships a Brazil Amazon Priority List replication vignette (Xu 2026 Section III) — see its [pkgdown site](https://xiangao.github.io/didint/) for the end-to-end real-data walkthrough. The two packages use the same DR formula and trim / aggregation behaviour, so estimates match to MC noise.
