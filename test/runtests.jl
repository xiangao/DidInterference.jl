using Test
using DidInterference
using DataFrames
using Random
using LinearAlgebra
using Statistics
using StatsBase
using Distributions

# Lattice DGP with binary direct + spillover effects. Same structure as
# the R smoke test in tests/testthat/test-2x2-smoke.R.
function simulate_2x2(; N = 1500, direct = 1.5, spill = 0.5,
                       z_dep = false, seed = 42)
    Random.seed!(seed)
    lon = rand(N) .* 10
    lat = rand(N) .* 10
    z   = 0.3 .* lon .+ 0.2 .* lat .+ randn(N)
    W   = Int.(rand(N) .< (1 ./ (1 .+ exp.(0.5 .- 0.6 .* z))))
    # Distance matrix and neighbour share of treated
    dij = [sqrt((lon[i]-lon[j])^2 + (lat[i]-lat[j])^2) for i in 1:N, j in 1:N]
    A   = (dij .< 1.5) .& (dij .> 0)
    deg = max.(sum(A, dims = 2)[:], 1)
    share = vec((A * W) ./ deg)
    G = Int.(share .> median(share))
    Y_pre  = 0.8 .* z .+ randn(N)
    de_effect = z_dep ? (1.5 .+ 0.3 .* z) : fill(direct, N)
    Y_post = Y_pre .+ 0.2 .* z .+ de_effect .* W .+ spill .* G .* W .+ randn(N)
    DataFrame(W = W, G = G, z = z, Y_pre = Y_pre, Y_post = Y_post,
              lon = lon, lat = lat)
end

@testset "did_int_2x2 smoke" begin
    df = simulate_2x2(N = 1500, seed = 1)
    res = did_int_2x2(df; yname = :Y_post, yname_pre = :Y_pre,
                      treat = :W, exposure = :G, g = 1,
                      covariates = [:z])
    @test isa(res.estimate, Real)
    @test res.se > 0
    truth = 1.5 + 0.5
    @test abs(res.estimate - truth) < 4 * res.se
end

@testset "did_int_2x2 errors on missing exposure level" begin
    df = simulate_2x2(N = 300, seed = 2)
    df.G[df.G .== 1] .= 0       # eliminate g = 1
    @test_throws ArgumentError did_int_2x2(
        df; yname = :Y_post, yname_pre = :Y_pre,
        treat = :W, exposure = :G, g = 1, covariates = [:z])
end

@testset "did_int_2x2 z-dependent: full-population estimand" begin
    Random.seed!(0)
    reps = 60
    ests = Float64[]
    for r in 1:reps
        df = simulate_2x2(N = 1500, z_dep = true, seed = r)
        try
            e = did_int_2x2(df; yname = :Y_post, yname_pre = :Y_pre,
                            treat = :W, exposure = :G, g = 1,
                            covariates = [:z], trim = 0.01).estimate
            push!(ests, e)
        catch end
    end
    # Truth (avg over full pop) ≈ 2.0 + 0.3 * E[z|full] ≈ 2.0 + 0.3 * 2.5 = 2.75
    @test abs(mean(ests) - 2.75) < 0.20
end

@testset "did_int_dynamic" begin
    Random.seed!(11)
    N, T_post = 1500, 4
    lon = rand(N) .* 10; lat = rand(N) .* 10
    z = 0.3 .* lon .+ 0.2 .* lat .+ randn(N)
    W = Int.(rand(N) .< (1 ./ (1 .+ exp.(0.5 .- 0.6 .* z))))
    dij = [sqrt((lon[i]-lon[j])^2 + (lat[i]-lat[j])^2) for i in 1:N, j in 1:N]
    A   = (dij .< 1.5) .& (dij .> 0)
    deg = max.(sum(A, dims = 2)[:], 1)
    share = vec((A * W) ./ deg)
    G = Int.(share .> median(share))
    Y_pre = 0.8 .* z .+ randn(N)
    df = DataFrame(W = W, G = G, z = z, Y_pre = Y_pre, lon = lon, lat = lat)
    for k in 1:T_post
        df[!, Symbol("Y_post_$k")] = Y_pre .+ 0.2 * k .* z .+
                                     1.5 .* W .+ 0.5 .* G .* W .+ randn(N)
    end
    res = did_int_dynamic(df;
        yname_pre = :Y_pre,
        ynames = [Symbol("Y_post_$k") for k in 1:T_post],
        treat = :W, exposure = :G, g = 1,
        covariates = [:z], trim = 0.01)
    @test nrow(res.per_period) == T_post
    @test all(abs.(res.per_period.estimate .- 2.0) .< 3 .* res.per_period.se)
    @test res.agg.se < maximum(res.per_period.se)
end

@testset "did_int_staggered" begin
    Random.seed!(7)
    N, T = 1500, 5
    lon = rand(N) .* 10; lat = rand(N) .* 10
    z = 0.3 .* lon .+ 0.2 .* lat .+ randn(N)
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
        push!(rows, (id = i, time = t, cohort = cohort[i], z = z[i], Y = Y, G = G_t))
    end
    df = DataFrame(rows)
    res = did_int_staggered(df;
        yname = :Y, time = :time, id = :id, cohort = :cohort,
        exposure = :G, g = 1, covariates = [:z])
    @test nrow(res.per_cell) > 0
    @test all(res.per_cell.cohort .∈ Ref([2, 3, 4]))
    @test all(res.per_cell.time .>= res.per_cell.cohort)
    @test abs(res.agg.simple.estimate - 2.0) < 0.30
end

@testset "_dr_atte_mult smoke" begin
    using DidInterference: _dr_atte_mult
    Random.seed!(11)
    N = 4000
    z = randn(N)
    W = Int.(rand(N) .< 0.5)
    Ig = Int.(rand(N) .< 0.5)
    Ypre  = rand.(Poisson.(exp.(0.5 .+ 0.3 .* z)))
    μpost = exp.(0.5 .+ 0.3 .* z .+ 0.2 .+ 0.4 .* (W .* Ig))
    Ypost = rand.(Poisson.(μpost))
    res = _dr_atte_mult(W, Ig, DataFrame(z = z), float.(Ypre), float.(Ypost))
    @test res.scale == :multiplicative
    @test res.se > 0
    @test abs(res.estimate - (exp(0.4) - 1)) < 4 * res.se
end

@testset "did_int_2x2 poisson" begin
    Random.seed!(7); N = 4000
    z = randn(N); W = Int.(rand(N) .< 0.5); Ig = Int.(rand(N) .< 0.5)
    Ypre  = rand.(Poisson.(exp.(0.5 .+ 0.3 .* z)))
    Ypost = rand.(Poisson.(exp.(0.5 .+ 0.3 .* z .+ 0.2 .+ 0.4 .* (W .* Ig))))
    df = DataFrame(Ypre = float.(Ypre), Ypost = float.(Ypost), G = Ig, W = W, z = z)
    res = did_int_2x2(df; yname = :Ypost, yname_pre = :Ypre, treat = :W,
                      exposure = :G, g = 1, covariates = [:z], family = :poisson)
    @test res.family == :poisson
    @test abs(res.estimate - (exp(0.4) - 1)) < 4 * res.se
    res_g = did_int_2x2(df; yname = :Ypost, yname_pre = :Ypre, treat = :W,
                        exposure = :G, g = 1, covariates = [:z])
    @test res_g.exposure_g == 1
end
