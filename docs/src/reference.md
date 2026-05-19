# [API Reference](@id API-Reference)

```@meta
CurrentModule = DidInterference
```

Each function's docstring describes the API; an `@example` block
immediately after shows it running with real output.

## 2×2 base case

```@docs
did_int_2x2
```

### Example

```@example dr_2x2
using DidInterference, DataFrames
using Random
Random.seed!(1)

N = 600
df = DataFrame(
    W = rand(0:1, N),
    G = rand(0:1, N),
    z = randn(N),
    Y_pre = randn(N))
df.Y_post = df.Y_pre .+ 0.2 .* df.z .+ 1.5 .* df.W .+ 0.5 .* df.G .* df.W .+ randn(N)

res = did_int_2x2(df;
    yname = :Y_post, yname_pre = :Y_pre,
    treat = :W, exposure = :G, g = 1,
    covariates = [:z], trim = 0.01)
(estimate = round(res.estimate, digits = 3),
 ci       = round.(res.ci, digits = 3))
```

## Dynamic event study

```@docs
did_int_dynamic
```

### Example

```@example dyn
using DidInterference, DataFrames
using Random
Random.seed!(2)

N, T_post = 500, 4
df = DataFrame(W = rand(0:1, N), G = rand(0:1, N),
               z = randn(N), Y_pre = randn(N))
for k in 1:T_post
    df[!, Symbol("Y_post_$k")] =
        df.Y_pre .+ 0.2 .* df.z .+ 1.5 .* df.W .+ 0.5 .* df.G .* df.W .+ randn(N)
end

res = did_int_dynamic(df;
    yname_pre = :Y_pre,
    ynames = [Symbol("Y_post_$k") for k in 1:T_post],
    treat = :W, exposure = :G, g = 1,
    covariates = [:z], trim = 0.01)
res.per_period
```

## Staggered adoption

```@docs
did_int_staggered
```

### Example

```@example stagg
using DidInterference, DataFrames
using Random
Random.seed!(7)

N, T = 1000, 5
cohort = rand([2.0, 3.0, Inf], N)
rows = NamedTuple[]
for i in 1:N, t in 1:T
    W_t = Int(cohort[i] <= t)
    G_t = rand() < 0.4 ? 1 : 0
    z_i = randn()
    Y   = 0.5 * z_i + 1.5 * W_t + 0.5 * G_t * W_t + randn()
    push!(rows, (id = i, time = t, cohort = cohort[i],
                 z = z_i, Y = Y, G = G_t))
end
df = DataFrame(rows)

res = did_int_staggered(df;
    yname = :Y, time = :time, id = :id,
    cohort = :cohort, exposure = :G, g = 1, covariates = [:z])
first(res.per_cell, 5)
```
