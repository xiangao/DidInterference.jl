"""
    did_int_dynamic(data; yname_pre, ynames, treat, exposure, g,
                    covariates, event_time=nothing, trim=nothing, alpha=0.05)

Dynamic event-study DR DATT under common adoption timing
(Xu 2026, Section I). For each post-period outcome column in
`ynames`, runs `did_int_2x2` against `yname_pre`. Returns per-period
estimates and a simple average aggregate whose SE stacks per-period
influence functions (same units across periods).

# Returns
NamedTuple with `per_period::DataFrame` (rows: event_time, estimate, se,
ci_lo, ci_hi) and `agg::NamedTuple` with the cross-period average.

# Examples
```julia
using DidInterference, DataFrames, Random
Random.seed!(2)
N, T_post = 600, 4
df = DataFrame(W = rand(0:1, N), G = rand(0:1, N),
               z = randn(N), Y_pre = randn(N))
for k in 1:T_post
    df[!, Symbol("Y_post_\$k")] =
        df.Y_pre .+ 0.2 .* df.z .+ 1.5 .* df.W .+ 0.5 .* df.G .* df.W .+ randn(N)
end

res = did_int_dynamic(df;
    yname_pre = :Y_pre,
    ynames = [Symbol("Y_post_\$k") for k in 1:T_post],
    treat = :W, exposure = :G, g = 1,
    covariates = [:z], trim = 0.01)
res.per_period
```
"""
function did_int_dynamic(data::DataFrame;
                         yname_pre::Symbol,
                         ynames::Vector{Symbol},
                         treat::Symbol,
                         exposure::Symbol,
                         g,
                         covariates::Vector{Symbol},
                         event_time::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                         trim::Union{Nothing,Real} = nothing,
                         alpha::Real = 0.05)

    K = length(ynames)
    et = isnothing(event_time) ? (0:(K-1)) : event_time
    length(et) == K || throw(ArgumentError("event_time length must match ynames"))

    fits = Vector{Any}(undef, K)
    rows = Vector{NamedTuple}(undef, K)
    for k in 1:K
        f = did_int_2x2(data;
                        yname     = ynames[k],
                        yname_pre = yname_pre,
                        treat     = treat,
                        exposure  = exposure,
                        g         = g,
                        covariates = covariates,
                        trim      = trim,
                        alpha     = alpha)
        fits[k] = f
        rows[k] = (event_time = et[k],
                   estimate   = f.estimate,
                   se         = f.se,
                   ci_lo      = f.ci[1],
                   ci_hi      = f.ci[2])
    end
    per_period = DataFrame(rows)

    # Aggregate: simple average; IFs share the same N units across periods,
    # so the aggregate IF is the mean across periods.
    n_units = length(fits[1].influence)
    if_avg = sum(fits[k].influence for k in 1:K) ./ K
    est_avg = mean(per_period.estimate)
    se_avg  = sqrt(sum(if_avg .^ 2) / n_units^2)
    z = quantile(Normal(), 1 - alpha / 2)

    return (per_period = per_period,
            agg = (simple_avg = est_avg,
                   se = se_avg,
                   ci = (est_avg - z*se_avg, est_avg + z*se_avg)),
            fits = fits,
            exposure_g = g,
            alpha = alpha)
end
