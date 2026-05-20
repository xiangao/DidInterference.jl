# DidInterference.jl

[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://xiangao.github.io/DidInterference.jl/)

`DidInterference.jl` implements difference-in-differences estimators that allow
spatial interference. It is the Julia companion to the R package
[`didint`](https://github.com/xiangao/didint).

Implements the doubly robust direct ATT estimators of:

- Xu, Ruonan (2023). "Difference-in-Differences with Interference." [arXiv:2306.12003](https://arxiv.org/abs/2306.12003).
- Xu, Ruonan (2026). "Dynamic Difference-in-Differences with Interference." *AEA Papers and Proceedings* 116: 58–63.

## Why

Standard DiD assumes one unit's outcome does not depend on another unit's
treatment. That is often too strong for spatial policies. A municipality on the
priority list can affect nearby municipalities; a health intervention can
change exposure in a network. Xu's setup keeps the DiD comparison but adds an
exposure mapping, so the estimand is explicit about the spillover structure.

## Three estimators

- `did_int_2x2` — two-period, common adoption (Xu 2023, 2×2 case).
- `did_int_dynamic` — event study with common adoption timing (Xu 2026 §I).
- `did_int_staggered` — staggered adoption with not-yet-treated comparison groups, joint-IF aggregation across cells (Xu 2026 §II).

All three use the same ingredients: three propensity-score models, two
outcome-change regressions, and the doubly robust plug-in formula. Standard
errors come from the empirical influence function. The optional `trim` argument
matches the trimming used in Xu's Brazil application.

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

The companion R package [`didint`](https://github.com/xiangao/didint) includes
the Brazil Amazon Priority List replication vignette from Xu (2026, Section
III). The two packages use the same DR formula and the same trimming and
aggregation conventions.
