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
