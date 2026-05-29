"""
    did_int_staggered(data; yname, time, id, cohort, exposure, g,
                      covariates, pre_period=nothing, cohorts=nothing,
                      times=nothing, trim=nothing, alpha=0.05)

Staggered-adoption DR DATT with interference (Xu 2026, Section II).
For each cell `(c, t)` with `t >= c`, restricts to
`S_M = {C = c OR C > t}`, computes the DR DATT at exposure `g` with
`W = 1{C == c}`, and aggregates using the joint-IF stacking that
correctly accounts for shared units across cells.

# Returns
NamedTuple with `per_cell::DataFrame`, `agg` (with `simple`,
`event_time`, `cohort` sub-aggregates), `influence::Vector{Vector}`,
`cell_ids::Vector{Vector}`, `exposure_g`, `pre_period`, `alpha`.

# Examples
```julia
using DidInterference, DataFrames, Random
Random.seed!(7)
N, T = 1500, 5
# Cohorts: half treated at t=2, quarter at t=3, quarter never
cohort = rand([2.0, 3.0, Inf], N)
rows = NamedTuple[]
for i in 1:N, t in 1:T
    W_t   = Int(cohort[i] <= t)
    G_t   = rand() < 0.4 ? 1 : 0
    z_i   = randn()
    Y     = 0.5 * z_i + 1.5 * W_t + 0.5 * G_t * W_t + randn()
    push!(rows, (id = i, time = t, cohort = cohort[i],
                 z = z_i, Y = Y, G = G_t))
end
df = DataFrame(rows)

res = did_int_staggered(df;
    yname = :Y, time = :time, id = :id,
    cohort = :cohort, exposure = :G, g = 1, covariates = [:z])
res.per_cell
```
"""
function did_int_staggered(data::DataFrame;
                           yname::Symbol, time::Symbol, id::Symbol,
                           cohort::Symbol, exposure::Symbol, g,
                           covariates::Vector{Symbol},
                           pre_period::Union{Nothing,Real} = nothing,
                           cohorts::Union{Nothing,AbstractVector} = nothing,
                           times::Union{Nothing,AbstractVector} = nothing,
                           trim::Union{Nothing,Real} = nothing,
                           alpha::Real = 0.05,
                           family::Symbol = :gaussian)

    C_vals = data[!, cohort]
    finite_cohorts = sort(unique(skipmissing(filter(isfinite, C_vals))))
    isempty(finite_cohorts) &&
        throw(ArgumentError("did_int_staggered: no finite cohorts"))
    c_underbar = minimum(finite_cohorts)
    pre = pre_period === nothing ? c_underbar - 1 : pre_period
    cohorts_use = cohorts === nothing ? finite_cohorts : collect(cohorts)
    T_max = maximum(data[!, time])
    times_use = times === nothing ? collect(c_underbar:Int(T_max)) : collect(times)

    # Pre-period outcome per unit
    pre_df = data[data[!, time] .== pre, [id, yname]]
    rename!(pre_df, yname => :_Y_pre)

    rows_buf      = NamedTuple[]
    ifs_buf       = Vector{Vector{Float64}}()
    cell_ids_buf  = Vector{Vector{Any}}()
    all_ids = unique(data[!, id])

    for c_val in cohorts_use, t_val in times_use
        t_val < c_val && continue

        dt = data[data[!, time] .== t_val, :]
        m  = innerjoin(dt, pre_df, on = id)
        nrow(m) == 0 && continue

        C_i  = m[!, cohort]
        G_it = m[!, exposure]
        Y_t  = m[!, yname]
        Y_pre = m[!, :_Y_pre]
        Z    = select(m, covariates)
        ids_t = m[!, id]

        in_sm = (C_i .== c_val) .| (C_i .> t_val)
        sum(in_sm) == 0 && continue

        W  = Int.(C_i[in_sm] .== c_val)
        Ig = Int.(G_it[in_sm] .== g)
        dY = Y_t[in_sm] .- Y_pre[in_sm]
        Z_sm = Z[in_sm, :]
        ids_sm = ids_t[in_sm]

        # Skip cells lacking units in (W, G=g) subsets
        if !any(==(1), Ig) ||
           !any(i -> W[i] == 1 && Ig[i] == 1, eachindex(W)) ||
           !any(i -> W[i] == 0 && Ig[i] == 1, eachindex(W))
            @warn "did_int_staggered: cell (c=$(c_val), t=$(t_val)) skipped (empty subset)"
            continue
        end

        if family === :poisson
            Ypre_sm  = m[in_sm, :_Y_pre]
            Ypost_sm = m[in_sm, yname]
            out = try
                _dr_atte_mult(W, Ig, Z_sm, Ypre_sm, Ypost_sm; trim = trim, alpha = alpha)
            catch e
                @warn "did_int_staggered: cell (c=$(c_val), t=$(t_val)) failed: $(sprint(showerror, e))"
                nothing
            end
            out === nothing && continue
            push!(rows_buf, (cohort = c_val, time = t_val,
                             event_time = t_val - c_val,
                             estimate = out.logest, se = out.se_log,
                             ci_lo = out.logest - out.se_log,
                             ci_hi = out.logest + out.se_log,
                             n_total = out.n_total, n_at_g = out.n_at_g,
                             n_dropped = out.n_dropped))
            push!(ifs_buf, out.influence)
            push!(cell_ids_buf, ids_sm[out.keep_idx])
        else
            out = try
                _dr_atte(W, Ig, Z_sm, dY; trim = trim, alpha = alpha)
            catch e
                @warn "did_int_staggered: cell (c=$(c_val), t=$(t_val)) failed: $(sprint(showerror, e))"
                nothing
            end
            out === nothing && continue
            push!(rows_buf, (cohort = c_val, time = t_val,
                             event_time = t_val - c_val,
                             estimate = out.estimate, se = out.se,
                             ci_lo = out.ci[1], ci_hi = out.ci[2],
                             n_total = out.n_total, n_at_g = out.n_at_g,
                             n_dropped = out.n_dropped))
            push!(ifs_buf, out.influence)
            push!(cell_ids_buf, ids_sm[out.keep_idx])
        end
    end

    isempty(rows_buf) &&
        throw(ArgumentError("did_int_staggered: no cell could be estimated"))

    per_cell = DataFrame(rows_buf)

    # --- joint-IF aggregation -----------------------------------------------
    function agg_one(idx::AbstractVector{<:Integer})
        ests = per_cell.estimate[idx]
        ns   = per_cell.n_total[idx]
        w    = ns ./ sum(ns)
        est_avg = sum(w .* ests)
        h = Dict{Any, Float64}()
        for (j, kk) in enumerate(idx)
            psi_k = ifs_buf[kk] .+ per_cell.estimate[kk]   # un-center
            n_k   = per_cell.n_total[kk]
            ids_k = cell_ids_buf[kk]
            contrib = w[j] .* (psi_k .- per_cell.estimate[kk]) ./ n_k
            for (i_id, c_val) in zip(ids_k, contrib)
                h[i_id] = get(h, i_id, 0.0) + c_val
            end
        end
        se_avg = sqrt(sum(v^2 for v in values(h)))
        z = quantile(Normal(), 1 - alpha / 2)
        return (estimate = est_avg, se = se_avg,
                ci = (est_avg - z*se_avg, est_avg + z*se_avg),
                n_cells = length(idx))
    end

    agg_simple = agg_one(1:nrow(per_cell))

    agg_event = DataFrame(
        [let a = agg_one(findall(==(et), per_cell.event_time))
            (event_time = et, estimate = a.estimate, se = a.se,
             ci_lo = a.ci[1], ci_hi = a.ci[2], n_cells = a.n_cells)
         end
         for et in sort(unique(per_cell.event_time))])

    agg_cohort = DataFrame(
        [let a = agg_one(findall(==(c_val), per_cell.cohort))
            (cohort = c_val, estimate = a.estimate, se = a.se,
             ci_lo = a.ci[1], ci_hi = a.ci[2], n_cells = a.n_cells)
         end
         for c_val in sort(unique(per_cell.cohort))])

    if family === :poisson
        ℓ = agg_simple.estimate; s = agg_simple.se
        agg_simple = (estimate = exp(ℓ) - 1, se = exp(ℓ) * s,
                      ci = (exp(agg_simple.ci[1]) - 1, exp(agg_simple.ci[2]) - 1),
                      n_cells = agg_simple.n_cells)
        for tbl in (agg_event, agg_cohort)
            tbl.estimate = exp.(tbl.estimate) .- 1
            tbl.ci_lo    = exp.(tbl.ci_lo)    .- 1
            tbl.ci_hi    = exp.(tbl.ci_hi)    .- 1
        end
    end

    return (per_cell = per_cell,
            agg = (simple = agg_simple,
                   event_time = agg_event,
                   cohort = agg_cohort),
            influence = ifs_buf,
            cell_ids  = cell_ids_buf,
            exposure_g = g,
            pre_period = pre,
            alpha = alpha,
            family = family)
end
