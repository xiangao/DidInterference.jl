"""
Internal: doubly robust direct ATT at exposure level `g`.

Mirrors the R `.dr_atte()` helper in `didint`. Takes already-prepared
per-unit vectors and returns a NamedTuple with the estimate, SE, CI,
influence function, post-trim keep-index, and counts.

Both [`did_int_2x2`](@ref) and [`did_int_staggered`](@ref) build their
inputs and call this helper.

# Arguments
- `W::Vector{Int}` — treated-group indicator (0/1).
- `Ig::Vector{Int}` — exposure-at-g indicator (0/1).
- `Z::DataFrame` — covariates.
- `dY::Vector{<:Real}` — outcome change `Y_t - Y_pre`.
- `trim::Union{Nothing,Real}` — propensity-score trim threshold.
- `alpha::Real` — significance level (default 0.05).

# Returns
NamedTuple with fields `estimate, se, ci, influence, keep_idx,
n_total, n_treated, n_control, n_at_g, n_dropped`.
"""
function _dr_atte(W::AbstractVector{<:Integer},
                  Ig::AbstractVector{<:Integer},
                  Z::DataFrame,
                  dY::AbstractVector{<:Real};
                  trim::Union{Nothing,Real} = nothing,
                  alpha::Real = 0.05)

    n0 = length(W)
    n0 == length(Ig) == length(dY) == nrow(Z) ||
        throw(ArgumentError("_dr_atte: input lengths do not match"))
    any(==(1), Ig) || throw(ArgumentError("_dr_atte: no units with G = g"))
    any(i -> W[i] == 1 && Ig[i] == 1, eachindex(W)) ||
        throw(ArgumentError("_dr_atte: no treated units at G = g (cannot fit m_1g)"))
    any(i -> W[i] == 0 && Ig[i] == 1, eachindex(W)) ||
        throw(ArgumentError("_dr_atte: no control units at G = g (cannot fit m_0g)"))

    covs = names(Z)
    formula_rhs = Term(:_dummy) ~ sum(term.(Symbol.(covs)))   # rhs only used as a template
    rhs_terms = sum(term.(Symbol.(covs)))

    # --- propensity model p(z) = P(W = 1 | z) -------------------------------
    pdf = hcat(DataFrame(_W = W), Z)
    fit_p = glm(term(:_W) ~ rhs_terms, pdf, Binomial(), LogitLink())
    p_hat = predict(fit_p)

    # --- exposure propensities on treated / control subsamples --------------
    treated_idx = findall(==(1), W)
    control_idx = findall(==(0), W)

    pi1_df = hcat(DataFrame(_Ig = Ig[treated_idx]),
                  Z[treated_idx, :])
    fit_pi1 = glm(term(:_Ig) ~ rhs_terms, pi1_df, Binomial(), LogitLink())
    pi1g_hat = predict(fit_pi1, hcat(DataFrame(_Ig = Ig), Z))

    pi0_df = hcat(DataFrame(_Ig = Ig[control_idx]),
                  Z[control_idx, :])
    fit_pi0 = glm(term(:_Ig) ~ rhs_terms, pi0_df, Binomial(), LogitLink())
    pi0g_hat = predict(fit_pi0, hcat(DataFrame(_Ig = Ig), Z))

    # --- optional propensity-score trim (Xu 2026 uses 0.01) -----------------
    keep_idx = collect(1:n0)
    n_dropped = 0
    if trim !== nothing
        keep = (p_hat .> trim) .& (p_hat .< 1 - trim) .&
               (pi1g_hat .> trim) .& (pi1g_hat .< 1 - trim) .&
               (pi0g_hat .> trim) .& (pi0g_hat .< 1 - trim)
        n_dropped = sum(.!keep)
        if n_dropped > 0
            keep_idx = keep_idx[keep]
            W = W[keep]; Ig = Ig[keep]; dY = dY[keep]
            Z = Z[keep, :]
            p_hat = p_hat[keep]
            pi1g_hat = pi1g_hat[keep]; pi0g_hat = pi0g_hat[keep]
        end
    end

    # Post-trim non-empty checks (trim can empty the outcome regressions)
    any(i -> W[i] == 1 && Ig[i] == 1, eachindex(W)) ||
        throw(ArgumentError("_dr_atte: trim emptied the (W=1, G=g) subset"))
    any(i -> W[i] == 0 && Ig[i] == 1, eachindex(W)) ||
        throw(ArgumentError("_dr_atte: trim emptied the (W=0, G=g) subset"))

    # --- outcome-change regressions on (W=1, G=g) and (W=0, G=g) ------------
    mask1 = (W .== 1) .& (Ig .== 1)
    mask0 = (W .== 0) .& (Ig .== 1)
    m1_df = hcat(DataFrame(_dY = dY[mask1]), Z[mask1, :])
    m0_df = hcat(DataFrame(_dY = dY[mask0]), Z[mask0, :])
    fit_m1 = lm(term(:_dY) ~ rhs_terms, m1_df)
    fit_m0 = lm(term(:_dY) ~ rhs_terms, m0_df)
    m1_hat = predict(fit_m1, hcat(DataFrame(_dY = dY), Z))
    m0_hat = predict(fit_m0, hcat(DataFrame(_dY = dY), Z))

    # --- DR signal (Xu 2026 eq. 5) ------------------------------------------
    if_treated = W .* Ig ./ (p_hat .* pi1g_hat) .* (dY .- m1_hat)
    if_control = (1 .- W) .* Ig ./ ((1 .- p_hat) .* pi0g_hat) .* (dY .- m0_hat)
    if_reg     = m1_hat .- m0_hat
    psi = if_treated .- if_control .+ if_reg

    n   = length(W)
    est = sum(psi) / n
    if_emp = psi .- est
    se = sqrt(sum(if_emp .^ 2) / n^2)
    z = quantile(Normal(), 1 - alpha / 2)

    return (estimate  = est,
            se        = se,
            ci        = (est - z*se, est + z*se),
            influence = if_emp,
            keep_idx  = keep_idx,
            n_total   = n,
            n_treated = sum(W .== 1),
            n_control = sum(W .== 0),
            n_at_g    = sum(Ig .== 1),
            n_dropped = n_dropped)
end
