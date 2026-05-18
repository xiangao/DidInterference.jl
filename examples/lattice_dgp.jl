# Worked example: doubly robust DiD with spatial interference on a
# synthetic lattice DGP. Mirrors the R `inst/sims/mc_validation.R` in
# the companion didint package.
#
# Run from the package root:
#   julia --project=. examples/lattice_dgp.jl

using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = joinpath(@__DIR__, ".."))
Pkg.instantiate()

using DidInterference
using DataFrames
using Random
using Statistics

# ---------------------------------------------------------------------------
# DGP: 2x2 lattice with binary direct + spillover effects.
#   - z is a smooth function of (lon, lat) plus noise
#   - W = 1{logit(0.5 z - 0.5) > U}
#   - G = 1{share of treated neighbours within 1.5 > median(share)}
#   - Y_post = Y_pre + 0.2 z + 1.5 W + 0.5 G W + noise
# Truth: direct + spillover at g = 1 is 2.0.
# ---------------------------------------------------------------------------
function simulate_2x2(; N = 1500, seed = 1)
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
    DataFrame(W = W, G = G, z = z, Y_pre = Y_pre, Y_post = Y_post,
              lon = lon, lat = lat)
end

# ---------------------------------------------------------------------------
# 2x2 estimator — single run
# ---------------------------------------------------------------------------
println("=== did_int_2x2: single replicate ===")
df = simulate_2x2(N = 2000, seed = 1)
res = did_int_2x2(df;
    yname      = :Y_post,
    yname_pre  = :Y_pre,
    treat      = :W,
    exposure   = :G,
    g          = 1,
    covariates = [:z],
    trim       = 0.01)
println("estimate = ", round(res.estimate, digits = 3),
        "   se = ",   round(res.se,       digits = 3),
        "   95% CI = (", round(res.ci[1], digits = 3),
        ", ",            round(res.ci[2], digits = 3), ")",
        "   truth = 2.0")

# ---------------------------------------------------------------------------
# Small Monte Carlo: 100 reps, check bias / coverage
# ---------------------------------------------------------------------------
println("\n=== Monte Carlo: 100 reps, N = 1500 ===")
reps = 100
truth = 2.0
ests = Float64[]
covs = Bool[]
for r in 1:reps
    d = simulate_2x2(N = 1500, seed = r)
    try
        out = did_int_2x2(d;
            yname = :Y_post, yname_pre = :Y_pre,
            treat = :W, exposure = :G, g = 1,
            covariates = [:z], trim = 0.01)
        push!(ests, out.estimate)
        push!(covs, truth >= out.ci[1] && truth <= out.ci[2])
    catch
    end
end
println("bias        = ", round(mean(ests) - truth, digits = 3))
println("empirical SD = ", round(std(ests),           digits = 3))
println("coverage 95% = ", round(mean(covs),          digits = 3))

# ---------------------------------------------------------------------------
# Staggered estimator — one run on a 3-cohort lattice
# ---------------------------------------------------------------------------
println("\n=== did_int_staggered: 3 cohorts, 5 periods, N = 1500 ===")
function simulate_staggered(; N = 1500, T = 5, seed = 7)
    Random.seed!(seed)
    lon = rand(N) .* 10
    lat = rand(N) .* 10
    z   = 0.3 .* lon .+ 0.2 .* lat .+ randn(N)
    p_t = 1 ./ (1 .+ exp.(0.5 .- 0.5 .* z))
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
        Y = 0.8 * z[i] + 0.1 * t * z[i] + 1.5 * W_t + 0.5 * G_t * W_t + randn()
        push!(rows, (id = i, time = t, cohort = cohort[i],
                     z = z[i], Y = Y, G = G_t))
    end
    DataFrame(rows)
end

d = simulate_staggered(N = 1500, seed = 7)
res_s = did_int_staggered(d;
    yname = :Y, time = :time, id = :id,
    cohort = :cohort, exposure = :G, g = 1, covariates = [:z])
println("Per (cohort, time) cells:")
println(res_s.per_cell[!, [:cohort, :time, :event_time, :estimate, :se]])
println("\nSimple cross-cell aggregate:")
println("  estimate = ", round(res_s.agg.simple.estimate, digits = 3),
        "   se = ", round(res_s.agg.simple.se, digits = 3),
        "   truth = 2.0")
