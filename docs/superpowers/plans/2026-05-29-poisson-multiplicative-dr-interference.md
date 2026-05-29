# Multiplicative (Poisson) DR DiD with Interference — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a multiplicative (rate-ratio) doubly-robust estimand to `DidInterference.jl` for count outcomes, exposed as `family = :poisson` on all three estimators, validated by Monte Carlo.

**Architecture:** A new shared engine `_dr_atte_mult` implements the DR *ratio-of-ratios* ATT at exposure `g` under multiplicative parallel trends (spec §2): four AIPW means (post/pre × treated/control), reusing the additive engine's propensity + exposure-propensity models, with Poisson-QMLE outcome models. Estimate and influence function live on the **log scale** (`ℓ = log(1+θ)`) so the existing dynamic/staggered aggregators carry over; the user-facing estimate is `θ = exp(ℓ)−1`. The three public functions gain a `family` keyword (`:gaussian` default = current behavior) that dispatches additive vs multiplicative.

**Tech Stack:** Julia, `GLM.jl` (Binomial/Logit propensities, Poisson/Log outcome models), `DataFrames.jl`, `Distributions.jl`. Tests via `Test` stdlib; MC validation parallelized with `Threads.@threads`.

**Companion spec:** `docs/superpowers/specs/2026-05-29-poisson-multiplicative-dr-interference-design.md`

---

## File structure

| File | Responsibility |
|---|---|
| `src/dr_atte_mult.jl` | NEW. The multiplicative DR engine `_dr_atte_mult(W, Ig, Z, Ypre, Ypost; trim, alpha)` → NamedTuple with `estimate=θ`, `se`, `ci`, `logest=ℓ`, `se_log`, `influence` (log-scale IF), `scale`, counts. |
| `src/DidInterference.jl` | Modify: `include("dr_atte_mult.jl")`. |
| `src/did_int_2x2.jl` | Modify: add `family` kw; under `:poisson` call `_dr_atte_mult` with pre/post levels. |
| `src/did_int_staggered.jl` | Modify: add `family` kw; per-cell multiplicative fit + log-scale joint-IF aggregation. |
| `src/did_int_dynamic.jl` | Modify: add `family` kw; per-period multiplicative + log-scale average. |
| `test/runtests.jl` | Modify: add multiplicative smoke tests + MC bias/coverage testsets. |
| `docs/src/reference.md`, `docs/src/vignettes/03_multiplicative.md` | Modify/NEW: `@example` + short vignette. |

---

## Task 1: Multiplicative DR engine `_dr_atte_mult`

**Files:** Create `src/dr_atte_mult.jl`; modify `src/DidInterference.jl`; test in `test/runtests.jl`.

- [ ] **Step 1: Write the engine.** Create `src/dr_atte_mult.jl`:

```julia
"""
Internal: doubly-robust MULTIPLICATIVE (ratio-of-ratios) ATT at exposure `g`
under multiplicative parallel trends. Mirrors `_dr_atte` on the count scale.

θ(g) = [E[Y_post|W=1,g]/E[Y_pre|W=1,g]] / [E[Y_post|W=0,g]/E[Y_pre|W=0,g]] − 1,
estimated from four AIPW means; estimate and influence are returned on the log
scale (ℓ = log(1+θ)) for aggregation, plus the user-facing θ and CI.
"""
function _dr_atte_mult(W::AbstractVector{<:Integer},
                       Ig::AbstractVector{<:Integer},
                       Z::DataFrame,
                       Ypre::AbstractVector{<:Real},
                       Ypost::AbstractVector{<:Real};
                       trim::Union{Nothing,Real} = nothing,
                       alpha::Real = 0.05)
    n0 = length(W)
    n0 == length(Ig) == length(Ypre) == length(Ypost) == nrow(Z) ||
        throw(ArgumentError("_dr_atte_mult: input lengths do not match"))
    any(==(1), Ig) || throw(ArgumentError("_dr_atte_mult: no units with G = g"))
    any(i -> W[i] == 1 && Ig[i] == 1, eachindex(W)) ||
        throw(ArgumentError("_dr_atte_mult: no treated units at G = g"))
    any(i -> W[i] == 0 && Ig[i] == 1, eachindex(W)) ||
        throw(ArgumentError("_dr_atte_mult: no control units at G = g"))

    rhs = sum(term.(Symbol.(names(Z))))

    # propensity p(z) = P(W=1|z)
    p_hat = predict(glm(term(:_W) ~ rhs, hcat(DataFrame(_W = W), Z),
                        Binomial(), LogitLink()))
    # exposure propensities on treated / control subsamples
    ti = findall(==(1), W); ci = findall(==(0), W)
    pi1 = predict(glm(term(:_Ig) ~ rhs, hcat(DataFrame(_Ig = Ig[ti]), Z[ti, :]),
                      Binomial(), LogitLink()), hcat(DataFrame(_Ig = Ig), Z))
    pi0 = predict(glm(term(:_Ig) ~ rhs, hcat(DataFrame(_Ig = Ig[ci]), Z[ci, :]),
                      Binomial(), LogitLink()), hcat(DataFrame(_Ig = Ig), Z))

    # optional trim on extreme propensities (parity with _dr_atte)
    keep = trues(n0); n_dropped = 0
    if trim !== nothing
        keep = (p_hat .> trim) .& (p_hat .< 1 - trim) .&
               (pi1 .> trim) .& (pi1 .< 1 - trim) .&
               (pi0 .> trim) .& (pi0 .< 1 - trim)
        n_dropped = sum(.!keep)
    end

    # Poisson-QMLE outcome models on Z, fit on (W=1,Ig=1) and (W=0,Ig=1)
    m1 = (W .== 1) .& (Ig .== 1)
    m0 = (W .== 0) .& (Ig .== 1)
    pfit(y, mask) = glm(term(:_Y) ~ rhs, hcat(DataFrame(_Y = y[mask]), Z[mask, :]),
                        Poisson(), LogLink())
    Zall = hcat(DataFrame(_Y = zeros(n0)), Z)
    mpost1 = predict(pfit(Ypost, m1), Zall); mpre1 = predict(pfit(Ypre, m1), Zall)
    mpost0 = predict(pfit(Ypost, m0), Zall); mpre0 = predict(pfit(Ypre, m0), Zall)

    # AIPW per-unit summands: imputation for all + augmentation on subgroup
    wt1 = (W .* Ig) ./ (p_hat .* pi1)
    wt0 = ((1 .- W) .* Ig) ./ ((1 .- p_hat) .* pi0)
    s_post1 = mpost1 .+ wt1 .* (Ypost .- mpost1)
    s_pre1  = mpre1  .+ wt1 .* (Ypre  .- mpre1)
    s_post0 = mpost0 .+ wt0 .* (Ypost .- mpost0)
    s_pre0  = mpre0  .+ wt0 .* (Ypre  .- mpre0)

    idx = findall(keep); n = length(idx)
    ap1 = mean(s_post1[idx]); ar1 = mean(s_pre1[idx])
    ap0 = mean(s_post0[idx]); ar0 = mean(s_pre0[idx])
    (ap1 > 0 && ar1 > 0 && ap0 > 0 && ar0 > 0) ||
        throw(ArgumentError("_dr_atte_mult: non-positive AIPW mean (cannot take log)"))

    ℓ = log(ap1) - log(ar1) - log(ap0) + log(ar0)
    θ = exp(ℓ) - 1
    ifℓ = (s_post1[idx] .- ap1) ./ ap1 .- (s_pre1[idx] .- ar1) ./ ar1 .-
          (s_post0[idx] .- ap0) ./ ap0 .+ (s_pre0[idx] .- ar0) ./ ar0
    se_ℓ = sqrt(sum(ifℓ .^ 2) / n^2)
    z = quantile(Normal(), 1 - alpha / 2)
    return (estimate = θ, se = exp(ℓ) * se_ℓ,
            ci = (exp(ℓ - z * se_ℓ) - 1, exp(ℓ + z * se_ℓ) - 1),
            logest = ℓ, se_log = se_ℓ, influence = ifℓ, scale = :multiplicative,
            keep_idx = idx, n_total = n,
            n_treated = sum(W[idx] .== 1), n_control = sum(W[idx] .== 0),
            n_at_g = sum(Ig[idx] .== 1), n_dropped = n_dropped)
end
```

- [ ] **Step 2: Register the include.** In `src/DidInterference.jl`, add after `include("dr_atte.jl")`:

```julia
include("dr_atte_mult.jl")
```

- [ ] **Step 3: Write a failing smoke test.** In `test/runtests.jl`, append:

```julia
@testset "_dr_atte_mult smoke" begin
    using DidInterference: _dr_atte_mult
    Random.seed!(11)
    N = 4000
    z = randn(N)
    W = Int.(rand(N) .< 0.5)
    Ig = Int.(rand(N) .< 0.5)
    Ypre  = rand.(Poisson.(exp.(0.5 .+ 0.3 .* z)))
    # true multiplicative ATT at g=1: exp(0.4)-1; control growth exp(0.2)
    μpost = exp.(0.5 .+ 0.3 .* z .+ 0.2 .+ 0.4 .* (W .* Ig))
    Ypost = rand.(Poisson.(μpost))
    res = _dr_atte_mult(W, Ig, DataFrame(z = z), float.(Ypre), float.(Ypost))
    @test res.scale == :multiplicative
    @test res.se > 0
    @test abs(res.estimate - (exp(0.4) - 1)) < 4 * res.se
end
```

- [ ] **Step 4: Run it.** `cd ~/projects/software/DidInterference.jl && julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: the new testset passes (estimate near `exp(0.4)-1 ≈ 0.492`); existing tests still pass.

- [ ] **Step 5: Commit.**
```bash
git add src/dr_atte_mult.jl src/DidInterference.jl test/runtests.jl
git commit -m "feat: _dr_atte_mult — DR multiplicative ratio-of-ratios engine"
```

---

## Task 2: `family` keyword on `did_int_2x2`

**Files:** Modify `src/did_int_2x2.jl`; test in `test/runtests.jl`.

- [ ] **Step 1: Add the keyword + dispatch.** In `src/did_int_2x2.jl`, change the signature to add `family::Symbol = :gaussian` (after `g`), and replace the final two lines (`out = _dr_atte(...)` and `return merge(...)`) with:

```julia
    if family === :gaussian
        out = _dr_atte(W, Ig, Z, dY; trim = trim, alpha = alpha)
    elseif family === :poisson
        Ypre  = float.(data[!, yname_pre]); Ypost = float.(data[!, yname])
        out = _dr_atte_mult(W, Ig, Z, Ypre, Ypost; trim = trim, alpha = alpha)
    else
        throw(ArgumentError("did_int_2x2: family must be :gaussian or :poisson"))
    end
    return merge(out, (exposure_g = g, family = family))
```

- [ ] **Step 2: Write a failing test** for the public path. Append to `test/runtests.jl`:

```julia
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
    # default family unchanged
    res_g = did_int_2x2(df; yname = :Ypost, yname_pre = :Ypre, treat = :W,
                        exposure = :G, g = 1, covariates = [:z])
    @test res_g.exposure_g == 1
end
```

- [ ] **Step 3: Run** `julia --project=. -e 'using Pkg; Pkg.test()'`. Expected: both new testsets pass; all prior tests pass.

- [ ] **Step 4: Commit.**
```bash
git add src/did_int_2x2.jl test/runtests.jl
git commit -m "feat: family=:poisson on did_int_2x2"
```

---

## Task 3: Monte-Carlo validation gate (2x2)

**Files:** Test in `test/runtests.jl`. This is the correctness gate for the influence function.

- [ ] **Step 1: Write the MC testset.** Append to `test/runtests.jl`:

```julia
@testset "did_int_2x2 poisson MC bias/coverage" begin
    truth = exp(0.4) - 1
    R = 300
    ests = Vector{Float64}(undef, R); cov = falses(R)
    Threads.@threads for r in 1:R
        rng = MersenneTwister(1000 + r); N = 3000
        z = randn(rng, N); W = Int.(rand(rng, N) .< 0.5); Ig = Int.(rand(rng, N) .< 0.5)
        Ypre  = rand.(rng, Poisson.(exp.(0.5 .+ 0.3 .* z)))
        Ypost = rand.(rng, Poisson.(exp.(0.5 .+ 0.3 .* z .+ 0.2 .+ 0.4 .* (W .* Ig))))
        df = DataFrame(Ypre = float.(Ypre), Ypost = float.(Ypost), G = Ig, W = W, z = z)
        res = did_int_2x2(df; yname = :Ypost, yname_pre = :Ypre, treat = :W,
                          exposure = :G, g = 1, covariates = [:z], family = :poisson)
        ests[r] = res.estimate
        cov[r] = res.ci[1] <= truth <= res.ci[2]
    end
    bias = mean(ests) - truth
    coverage = mean(cov)
    @info "2x2 poisson MC" bias coverage
    @test abs(bias) < 0.02
    @test 0.90 <= coverage <= 0.98
end
```

- [ ] **Step 2: Run with threads** `julia --project=. -t auto -e 'using Pkg; Pkg.test()'`
Expected: `bias` within ±0.02 of 0, `coverage` in [0.90, 0.98]. **If coverage is off, STOP** — the influence function in Task 1 is wrong; fix `_dr_atte_mult` before continuing (this gate protects every downstream number).

- [ ] **Step 3: Commit.**
```bash
git add test/runtests.jl
git commit -m "test: MC bias/coverage gate for poisson did_int_2x2"
```

---

## Task 4: `family` on `did_int_staggered` + log-scale aggregation

**Files:** Modify `src/did_int_staggered.jl`; test in `test/runtests.jl`.

- [ ] **Step 1: Add the keyword + per-cell dispatch.** Add `family::Symbol = :gaussian` to the signature. At the per-`(c,t)` cell, the additive code builds `dY` and calls `_dr_atte`. Wrap it:

```julia
        if family === :gaussian
            out = try
                _dr_atte(W, Ig, Z_sm, dY; trim = trim, alpha = alpha)
            catch e; @warn "cell skipped: $(sprint(showerror,e))"; nothing end
            out === nothing && continue
            push!(rows_buf, (cohort = c_val, time = t_val, event_time = t_val - c_val,
                estimate = out.estimate, se = out.se, ci_lo = out.ci[1], ci_hi = out.ci[2],
                n_total = out.n_total, n_at_g = out.n_at_g, n_dropped = out.n_dropped))
        else  # :poisson — store the LOG-scale estimate for aggregation
            Ypre_sm = m[!, :_Y_pre]; Ypost_sm = m[!, yname]
            out = try
                _dr_atte_mult(W, Ig, Z_sm, Ypre_sm, Ypost_sm; trim = trim, alpha = alpha)
            catch e; @warn "cell skipped: $(sprint(showerror,e))"; nothing end
            out === nothing && continue
            push!(rows_buf, (cohort = c_val, time = t_val, event_time = t_val - c_val,
                estimate = out.logest, se = out.se_log, ci_lo = out.logest - out.se_log,
                ci_hi = out.logest + out.se_log,
                n_total = out.n_total, n_at_g = out.n_at_g, n_dropped = out.n_dropped))
        end
        push!(ifs_buf, out.influence)
        push!(cell_ids_buf, ids_sm[out.keep_idx])
```

Note: for `:poisson`, `per_cell.estimate` and `ifs_buf` are on the **log scale** (`ℓ_ct`). The existing `agg_one` joint-IF machinery then aggregates `ℓ` exactly as before. Only the final reporting must exponentiate.

- [ ] **Step 2: Exponentiate the aggregates for poisson.** After `agg_simple = agg_one(...)`, `agg_event`, `agg_cohort` are computed, add (guarded by `family === :poisson`) a transform that maps each aggregate's `(estimate, ci)` from log to multiplicative scale:

```julia
    expmap(a) = (estimate = exp(a.estimate) - 1, se = exp(a.estimate) * a.se,
                 ci = (exp(a.ci[1]) - 1, exp(a.ci[2]) - 1), n_cells = a.n_cells)
    if family === :poisson
        agg_simple = expmap(agg_simple)
        transform!(agg_event,  [:estimate, :ci_lo, :ci_hi] =>
            ByRow((e, lo, hi) -> (exp(e) - 1, exp(lo) - 1, exp(hi) - 1)) =>
            [:estimate, :ci_lo, :ci_hi])
        transform!(agg_cohort, [:estimate, :ci_lo, :ci_hi] =>
            ByRow((e, lo, hi) -> (exp(e) - 1, exp(lo) - 1, exp(hi) - 1)) =>
            [:estimate, :ci_lo, :ci_hi])
    end
```

Add `family = family` to the returned NamedTuple.

- [ ] **Step 3: Write a staggered poisson test.** Append to `test/runtests.jl` a 3-cohort multiplicative DGP (mirror the additive staggered test in the file, but counts):

```julia
@testset "did_int_staggered poisson" begin
    Random.seed!(3); N = 2500; T = 5
    z = randn(N)
    cohort = rand([2.0, 3.0, Inf], N)
    dij = [abs(z[i] - z[j]) for i in 1:N, j in 1:N]; A = (dij .< 0.3) .& (dij .> 0)
    deg = max.(vec(sum(A, dims = 2)), 1)
    rows = NamedTuple[]
    for i in 1:N, t in 1:T
        Wt = Int(cohort[i] <= t)
        Gt = Int(sum(A[i, :] .* (cohort .<= t)) / deg[i] > 0.3)
        μ = exp(0.4 + 0.3z[i] + 0.05t + 0.4*Wt + 0.3*Gt*Wt)  # ATT at g=1: exp(0.7)-1
        push!(rows, (id = i, time = t, cohort = cohort[i], z = z[i],
                     Y = float(rand(Poisson(μ))), G = Gt))
    end
    d = DataFrame(rows)
    res = did_int_staggered(d; yname = :Y, time = :time, id = :id, cohort = :cohort,
                            exposure = :G, g = 1, covariates = [:z], family = :poisson)
    @test res.family == :poisson
    @test abs(res.agg.simple.estimate - (exp(0.7) - 1)) < 4 * res.agg.simple.se
end
```

- [ ] **Step 4: Run** `julia --project=. -t auto -e 'using Pkg; Pkg.test()'`. Expected: estimate near `exp(0.7)-1 ≈ 1.01`; all prior tests pass.

- [ ] **Step 5: Commit.**
```bash
git add src/did_int_staggered.jl test/runtests.jl
git commit -m "feat: family=:poisson on did_int_staggered (log-scale joint-IF aggregation)"
```

---

## Task 5: `family` on `did_int_dynamic`

**Files:** Modify `src/did_int_dynamic.jl`; test in `test/runtests.jl`.

- [ ] **Step 1: Add keyword + per-period dispatch.** Add `family::Symbol = :gaussian`. `did_int_dynamic` calls `did_int_2x2` per post-period; pass `family` through. The per-period `f.estimate` will then be `θ_k` (multiplicative) and `f.influence` the log-scale IF. For aggregation, average on the **log scale**: replace the aggregation block with a family-aware version:

```julia
    n_units = length(fits[1].influence)
    if family === :gaussian
        if_avg = sum(fits[k].influence for k in 1:K) ./ K
        est_avg = mean(per_period.estimate)
        se_avg  = sqrt(sum(if_avg .^ 2) / n_units^2)
        agg = (simple_avg = est_avg, se = se_avg,
               ci = (est_avg - z*se_avg, est_avg + z*se_avg))
    else  # poisson: fits[k].influence is log-scale; average ℓ then exponentiate
        if_avg = sum(fits[k].influence for k in 1:K) ./ K
        ℓ_avg  = mean(log1p.(per_period.estimate))
        se_ℓ   = sqrt(sum(if_avg .^ 2) / n_units^2)
        agg = (simple_avg = exp(ℓ_avg) - 1, se = exp(ℓ_avg) * se_ℓ,
               ci = (exp(ℓ_avg - z*se_ℓ) - 1, exp(ℓ_avg + z*se_ℓ) - 1))
    end
```

(Pass `family = family` into the `did_int_2x2` call, and add `z = quantile(Normal(), 1 - alpha/2)` before the block if not already in scope.)

- [ ] **Step 2: Write a test.** Append a common-timing multiplicative event-study DGP (wide format, `Y_pre` + `Y_post_k`), asserting `res.agg.simple_avg` is near the true `exp(δ+ψ)-1` within `4*se`. Use `treat`, `exposure = :G`, `g = 1`, `family = :poisson`. (Model each `Y_post_k ~ Poisson(exp(base + 0.4*W + 0.3*G*W))`, truth `exp(0.7)-1`.)

```julia
@testset "did_int_dynamic poisson" begin
    Random.seed!(5); N = 3000; K = 3
    z = randn(N); W = Int.(rand(N) .< 0.5); G = Int.(rand(N) .< 0.5)
    df = DataFrame(W = W, G = G, z = z,
                   Y_pre = float.(rand.(Poisson.(exp.(0.4 .+ 0.3 .* z)))))
    for k in 1:K
        df[!, Symbol("Y_post_$k")] =
            float.(rand.(Poisson.(exp.(0.4 .+ 0.3 .* z .+ 0.4 .* W .+ 0.3 .* G .* W))))
    end
    res = did_int_dynamic(df; yname_pre = :Y_pre,
        ynames = [Symbol("Y_post_$k") for k in 1:K], treat = :W, exposure = :G,
        g = 1, covariates = [:z], family = :poisson)
    @test abs(res.agg.simple_avg - (exp(0.7) - 1)) < 4 * res.agg.se
end
```

- [ ] **Step 3: Run** `julia --project=. -t auto -e 'using Pkg; Pkg.test()'`. Expected: passes; prior tests pass.

- [ ] **Step 4: Commit.**
```bash
git add src/did_int_dynamic.jl test/runtests.jl
git commit -m "feat: family=:poisson on did_int_dynamic (log-scale period averaging)"
```

---

## Task 6: Documentation

**Files:** Modify `docs/src/reference.md`; create `docs/src/vignettes/03_multiplicative.md`; modify `docs/make.jl` pages list and `README.md`.

- [ ] **Step 1:** Add an executable `@example` block to `docs/src/reference.md` next to the `did_int_2x2` entry showing `family = :poisson` on a small count DGP and printing `res.estimate` (per the rule that doc examples must run in `@example` blocks).
- [ ] **Step 2:** Write `docs/src/vignettes/03_multiplicative.md`: motivation (counts/zeros, Chen-Roth), the multiplicative-parallel-trends estimand, a runnable `did_int_staggered(...; family=:poisson)` example, and a one-line interpretation ("θ×100% more events at exposure g"). Add it to the `pages` list in `docs/make.jl`.
- [ ] **Step 3:** Add a short "Count outcomes (`family=:poisson`)" section to `README.md`.
- [ ] **Step 4: Build docs** `julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate(); include("docs/make.jl")'`
Expected: docs build without `@example` errors.
- [ ] **Step 5: Commit.**
```bash
git add docs/ README.md
git commit -m "docs: multiplicative (poisson) family — reference example + vignette"
```

---

## Task 7: Re-run the dispensary paper with `family=:poisson`

**Files:** `~/projects/claude/crime_dispensary_interference/code/06_estimate.jl` (add a poisson pass); `report/paper.qmd` (comparison table). This task is in the **paper** repo, not the package.

- [ ] **Step 1:** In `06_estimate.jl`, after the existing `log1p` runs, add a parallel set of calls passing raw counts (`crime_total`/`property`/`violent`) with `family = :poisson` (no `log1p`), reusing the same `cohort_*`/`G_dir`/`G0` columns and half-year binning. Store under keys `*_pois`. Print the multiplicative ATT (`θ`) and CI.
- [ ] **Step 2: Run** `julia -t auto code/06_estimate.jl`. Expected: spillover `θ` for total/property/violent with CIs; compare sign/magnitude to the `log(1+y)` results.
- [ ] **Step 3:** Add a column to the results table in `report/paper.qmd` reporting the multiplicative ATT next to the additive one, with one sentence on whether the displacement signal survives. Re-render `quarto render report/paper.qmd --to pdf`.
- [ ] **Step 4: Commit (paper repo).**
```bash
cd ~/projects/claude/crime_dispensary_interference
git add code/06_estimate.jl report/paper.qmd output/paper.pdf
git commit -m "feat: poisson (multiplicative) robustness for spillover/direct arms"
```

---

## Self-review notes

- **Spec coverage:** estimand/ratio-of-ratios §2 → Task 1; double robustness §2 → Task 1 (four AIPW means); API `family` §3 → Tasks 2,4,5; zero handling (group means) → Task 1 (positivity check); log-scale aggregation §2 → Tasks 4,5; MC validation §4 → Task 3 (2x2 gate) + Tasks 4,5 (point-estimate checks); downstream rerun §5 → Task 7. All covered.
- **Type/name consistency:** `_dr_atte_mult` returns `logest`, `se_log`, `influence` (log-scale), `estimate`/`se`/`ci` (θ-scale), `scale`, counts — used consistently by Tasks 2 (θ-scale), 4 and 5 (log-scale `logest`/`influence`, then `expmap`). `family` keyword name identical across all three functions. `expmap` defined in Task 4 Step 2.
- **No placeholders:** Tasks 1–5 contain full code; Task 6/7 steps specify exact files, commands, and the content to add (Task 5 Step 2 and Task 7 give the DGP/edits explicitly).
- **Known risk:** the influence function (Task 1) — gated by Task 3's MC bias/coverage. If that fails, fix Task 1 before proceeding. Thin per-cohort cells in the staggered poisson path are handled the same way as the additive case (cohort coarsening lives in the paper, Task 7), not in the package.
