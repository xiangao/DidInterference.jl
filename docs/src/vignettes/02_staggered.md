# Staggered Adoption

This vignette demonstrates `did_int_staggered` on a multi-cohort
panel where treatment is adopted at different times across units.
The DR DATT is computed per `(cohort, time)` cell, then aggregated
across cells with joint-IF stacking that correctly accounts for the
shared not-yet-treated comparison group.

```@example staggered
using DidInterference
using DataFrames
using Random
using Statistics

function simulate_staggered(; N = 1500, T = 5, seed = 7)
    Random.seed!(seed)
    lon  = rand(N) .* 10
    lat  = rand(N) .* 10
    z    = 0.3 .* lon .+ 0.2 .* lat .+ randn(N)
    p_t  = 1 ./ (1 .+ exp.(0.5 .- 0.5 .* z))
    is_t = rand(N) .< p_t
    cohort = fill(Inf, N)
    cohort[is_t] = rand([2, 3, 4], sum(is_t))
    dij = [sqrt((lon[i]-lon[j])^2 + (lat[i]-lat[j])^2) for i in 1:N, j in 1:N]
    A   = (dij .< 1.5) .& (dij .> 0)
    deg = max.(sum(A, dims = 2)[:], 1)
    rows = NamedTuple[]
    for i in 1:N, t in 1:T
        W_t = Int(cohort[i] <= t)
        share_t = sum(A[i, :] .* (cohort .<= t)) / deg[i]
        G_t = Int(share_t > 0.3)
        Y = 0.8 * z[i] + 0.1 * t * z[i] +
            1.5 * W_t + 0.5 * G_t * W_t + randn()
        push!(rows, (id = i, time = t, cohort = cohort[i],
                     z = z[i], Y = Y, G = G_t))
    end
    DataFrame(rows)
end

panel = simulate_staggered(N = 1500, seed = 7)
combine(groupby(panel, :cohort), nrow => :n_obs)
```

## Fit the staggered estimator

```@example staggered
res = did_int_staggered(panel;
    yname     = :Y,
    time      = :time,
    id        = :id,
    cohort    = :cohort,
    exposure  = :G,
    g         = 1,
    covariates = [:z],
)

select(res.per_cell,
       [:cohort, :time, :event_time, :estimate, :se, :ci_lo, :ci_hi])
```

Each row is a `(c, t)` cell. The true direct + spillover effect at
``G = 1`` is 2.0; estimates should be close.

## Aggregates

```@example staggered
res.agg.simple
```

The simple cross-cell average uses sample-size weights ``w_k = n_k /
\sum_l n_l``. Its standard error is the joint-IF stacked variance
across units that appear in multiple cells — *not* the
independent-cells approximation, which underestimates the true
variance when cells share the never-treated comparison group.

By event time:

```@example staggered
res.agg.event_time
```

By cohort:

```@example staggered
res.agg.cohort
```

## Why joint-IF aggregation matters

In a small Monte Carlo with the same DGP, the independent-cells SE
gave 84 % CI coverage; the joint-IF stacking restores it to 96 %.
The R companion package `didint` reports the same comparison in its
[`inst/sims/findings.md`](https://github.com/xiangao/didint/blob/master/inst/sims/findings.md).

For an applied example, see the R `didint` vignette on the Brazil Amazon
Priority List. It fits the staggered-DiD-with-interference design on the
Assunção et al. (2023) data.
