# Getting Started

This vignette walks through the 2×2 base case of `DidInterference.jl`
on a small synthetic lattice DGP. We simulate a binary direct effect
of size 1.5 and a binary spillover at exposure level ``g = 1`` of
size 0.5, so the true DR DATT at ``G = 1`` is **2.0**. The estimator
should recover this without bias.

```@example getting_started
using DidInterference
using DataFrames
using Random
using Statistics

function simulate_2x2(; N = 2000, seed = 1)
    Random.seed!(seed)
    lon = rand(N) .* 10
    lat = rand(N) .* 10
    z   = 0.3 .* lon .+ 0.2 .* lat .+ randn(N)
    W   = Int.(rand(N) .< (1 ./ (1 .+ exp.(0.5 .- 0.6 .* z))))
    dij = [sqrt((lon[i]-lon[j])^2 + (lat[i]-lat[j])^2) for i in 1:N, j in 1:N]
    A   = (dij .< 1.5) .& (dij .> 0)
    deg = max.(sum(A, dims = 2)[:], 1)
    share = vec((A * W) ./ deg)
    G = Int.(share .> median(share))
    Y_pre  = 0.8 .* z .+ randn(N)
    Y_post = Y_pre .+ 0.2 .* z .+ 1.5 .* W .+ 0.5 .* G .* W .+ randn(N)
    DataFrame(W = W, G = G, z = z, Y_pre = Y_pre, Y_post = Y_post)
end

df = simulate_2x2(N = 2000, seed = 1)
first(df, 5)
```

## Single fit

```@example getting_started
res = did_int_2x2(df;
    yname      = :Y_post,
    yname_pre  = :Y_pre,
    treat      = :W,
    exposure   = :G,
    g          = 1,
    covariates = [:z],
    trim       = 0.01,   # Xu (2026) uses 0.01 in the Brazil application
)

(estimate = round(res.estimate, digits = 3),
 se       = round(res.se,       digits = 3),
 ci       = round.(res.ci,      digits = 3),
 truth    = 2.0)
```

The estimate sits close to the truth of 2.0 within roughly one
standard error.

## Small Monte Carlo: bias and coverage

Repeating across 100 replicates to check that the bias goes to zero
and the influence-function SE matches the empirical standard
deviation across reps.

```@example getting_started
reps  = 100
truth = 2.0
ests  = Float64[]
covs  = Bool[]
for r in 1:reps
    d   = simulate_2x2(N = 1500, seed = r)
    out = did_int_2x2(d;
        yname = :Y_post, yname_pre = :Y_pre,
        treat = :W, exposure = :G, g = 1,
        covariates = [:z], trim = 0.01)
    push!(ests, out.estimate)
    push!(covs, truth >= out.ci[1] && truth <= out.ci[2])
end

(bias         = round(mean(ests) - truth, digits = 3),
 empirical_sd = round(std(ests),          digits = 3),
 coverage_95  = round(mean(covs),         digits = 3))
```

The empirical standard deviation of the point estimates and the
influence-function-based mean SE typically agree to within sampling
noise. Coverage of the 95 % CI should be close to the nominal level.

## Optional: trim

`trim = 0.01` drops units whose cohort or exposure propensity score
falls below 0.01 or above 0.99. This is the same trimming Xu uses in
the Brazil application and substantially shortens the right tail of
the estimator under poor overlap.

```@example getting_started
res_notrim = did_int_2x2(df;
    yname = :Y_post, yname_pre = :Y_pre,
    treat = :W, exposure = :G, g = 1,
    covariates = [:z])     # no trim

(estimate   = round(res_notrim.estimate, digits = 3),
 n_dropped  = res_notrim.n_dropped)
```

Without trimming, individual replicates with units near the
propensity-score boundary can produce noisy estimates.
