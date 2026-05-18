"""
    did_int_2x2(data; yname, yname_pre, treat, exposure, g, covariates,
                trim=nothing, alpha=0.05)

Doubly robust direct ATT at exposure level `g` in the two-period
common-adoption-timing case (Xu 2023). Mirrors the R `did_int_2x2()`.

# Arguments
- `data::DataFrame` — wide-format data with pre- and post-period outcomes.
- `yname::Symbol` — post-period outcome column.
- `yname_pre::Symbol` — pre-period outcome column.
- `treat::Symbol` — binary treatment indicator column (0/1).
- `exposure::Symbol` — exposure column (integer; we estimate at `G == g`).
- `g` — target exposure level.
- `covariates::Vector{Symbol}` — covariate columns.
- `trim`, `alpha` — see `_dr_atte`.

# Returns
NamedTuple with `estimate`, `se`, `ci`, `n_treated`, `n_control`,
`n_total`, `n_at_g`, `n_dropped`, `exposure_g`, `influence`.
"""
function did_int_2x2(data::DataFrame;
                     yname::Symbol,
                     yname_pre::Symbol,
                     treat::Symbol,
                     exposure::Symbol,
                     g,
                     covariates::Vector{Symbol},
                     trim::Union{Nothing,Real} = nothing,
                     alpha::Real = 0.05)

    W  = Int.(data[!, treat])
    Gv = data[!, exposure]
    dY = data[!, yname] .- data[!, yname_pre]
    Z  = select(data, covariates)
    Ig = Int.(Gv .== g)

    any(ismissing, W) || any(ismissing, dY) || any(isnan, dY) &&
        throw(ArgumentError("did_int_2x2: missing/NaN in inputs"))
    all(w -> w == 0 || w == 1, W) ||
        throw(ArgumentError("did_int_2x2: treat must be 0/1"))

    out = _dr_atte(W, Ig, Z, dY; trim = trim, alpha = alpha)
    return merge(out, (exposure_g = g,))
end
