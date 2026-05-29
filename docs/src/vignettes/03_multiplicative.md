# Multiplicative DiD for Count Outcomes

Count outcomes — events per plant, crimes per district, hospital admissions per
county — require care. The additive parallel trends assumption asks whether the
*level* change would have been the same for treated and control units. But for
counts, a proportional model is more natural: units that start with twice as many
events tend to *gain* twice as many events under a parallel shock.

There is a second problem. When zeros are common, the workaround `log(1 + Y)` is
not scale-invariant: multiplying Y by a constant shifts the estimated effect, so
the "effect" depends on the unit of measurement. Chen and Roth (2024) recommend
using a multiplicative (Poisson-based) model instead.

## Estimand

Under multiplicative parallel trends, the causal estimand at exposure level `g` is

```math
\theta(g) = \frac{E[Y_1(1,g)] / E[Y_0(1,g)]}{E[Y_1(0,g)] / E[Y_0(0,g)]} - 1
```

This is the ratio of the treated group's growth ratio to the control group's
counterfactual growth ratio, minus one. When `θ(g) = 0.40`, treated units at
exposure `g` have 40 % more events than their multiplicative-parallel-trends
counterfactual. The estimand is identified by the four AIPW means, each fit with
a Poisson quasi-MLE outcome model.

Pass `family = :poisson` to any of the three estimators to get this multiplicative
ATT. The default `family = :gaussian` is the existing additive estimator.

## Example: staggered adoption with count outcomes

```@example mult_staggered
using DidInterference
using DataFrames
using Random
using Distributions
using Statistics

function simulate_count_staggered(; N = 1500, T = 5, seed = 42)
    Random.seed!(seed)
    z      = randn(N)
    p_t    = 1 ./ (1 .+ exp.(0.5 .- 0.5 .* z))
    is_t   = rand(N) .< p_t
    cohort = fill(Inf, N)
    cohort[is_t] = rand([2.0, 3.0, 4.0], sum(is_t))

    rows = NamedTuple[]
    for i in 1:N, t in 1:T
        W_t = Int(cohort[i] <= t)
        G_t = rand() < 0.4 ? 1 : 0
        lam = exp(0.5 + 0.3 * z[i] + 0.3 * W_t + 0.35 * G_t * W_t)
        Y   = float(rand(Poisson(lam)))
        push!(rows, (id = i, time = t, cohort = cohort[i],
                     z = z[i], Y = Y, G = G_t))
    end
    DataFrame(rows)
end

panel = simulate_count_staggered(N = 1500, seed = 42)
combine(groupby(panel, :cohort), nrow => :n_obs)
```

## Fit the multiplicative staggered estimator

```@example mult_staggered
res = did_int_staggered(panel;
    yname      = :Y,
    time       = :time,
    id         = :id,
    cohort     = :cohort,
    exposure   = :G,
    g          = 1,
    covariates = [:z],
    family     = :poisson,
)

select(res.per_cell,
       [:cohort, :time, :event_time, :estimate, :se, :ci_lo, :ci_hi])
```

Each row is a `(cohort, time)` cell. The true multiplicative ATT at `G = 1` is
`exp(0.35) − 1 ≈ 0.419`; units treated at exposure `g = 1` have roughly 42 %
more events than their multiplicative-parallel-trends counterfactual.

## Aggregates

```@example mult_staggered
res.agg.simple
```

By event time:

```@example mult_staggered
res.agg.event_time
```

By cohort:

```@example mult_staggered
res.agg.cohort
```

## Reference

Chen, J., and Roth, J. (2024). "Logs with Zeros? Some Problems and Solutions."
*Quarterly Journal of Economics* 139(2): 891–936.
