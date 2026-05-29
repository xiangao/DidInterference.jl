# Decomposition: catching a composition-driven spillover

A difference-in-differences "effect" is only as clean as the treatment that defines
it. When a single treatment label pools two different kinds of event, the pooled
estimate can be positive and significant even when the effect of interest is exactly
zero — because one sub-population carries a confound. The interference-DiD machinery
here makes the treatment an explicit, decomposable estimand, so you can split the
pooled effect by sub-population and *see* the problem instead of publishing it.

This vignette shows the failure and the fix on a synthetic data set, so it runs
without any external data. (It mirrors a real case: estimating crime spillovers
around cannabis-dispensary openings, where "openings" silently mixed genuinely new
storefronts with pre-existing medical shops that merely added recreational sales on
the legalization launch day. The pooled spillover looked like displacement; it was
entirely the launch-day conversions.)

## A two-type DGP with zero true effect

We simulate units of two treated types plus never-treated controls:

- **"new"** units adopt at staggered later times; their **true treatment effect is
  zero**.
- **"conversion"** units all adopt at one early date (a "launch"), and they sit in
  higher-baseline areas that receive a **post-launch shock unrelated to treatment** —
  a confound. Their true treatment effect is *also* zero; the post-launch jump is not
  caused by the treatment.

```@example decomp
using DidInterference, DataFrames, Random, Statistics
Random.seed!(1)
N, T = 3000, 6
u = rand(N)
ctype  = [x < 0.45 ? :control : x < 0.72 ? :conv : :new for x in u]
cohort = [c == :control ? Inf : c == :conv ? 2.0 : rand([3.0, 4.0, 5.0]) for c in ctype]
z      = randn(N)                                  # observed covariate
base   = [c == :conv ? 1.5 : 0.0 for c in ctype] .+ 0.3 .* z   # conversions in higher-baseline areas

rows = NamedTuple[]
for i in 1:N, t in 1:T
    Wt    = Int(cohort[i] <= t)
    shock = (ctype[i] == :conv && t >= 2) ? 0.8 : 0.0   # confound: conv areas jump at the launch
    Y     = base[i] + 0.1t + shock + 0.0 * Wt + randn() # TRUE treatment effect = 0 for everyone
    push!(rows, (id = i, time = t, cohort = cohort[i], z = z[i], Y = Y, G = 0))
end
df = DataFrame(rows)
nothing # hide
```

## The pooled estimate is spuriously positive

Pool every adopter together and estimate one effect:

```@example decomp
est(d) = did_int_staggered(d; yname = :Y, time = :time, id = :id,
                           cohort = :cohort, exposure = :G, g = 0,
                           covariates = [:z]).agg.simple
pooled = est(df)
round.((pooled.estimate, pooled.ci[1], pooled.ci[2]), digits = 3)
```

The pooled ATT is positive and its 95 % interval excludes zero — even though we built
the data with **no treatment effect at all**. A naive reader would report "treatment
raises `Y`." The doubly-robust adjustment for the covariate `z` does not save us,
because the confound is a post-launch shock, not something `z` predicts.

## Decompose by sub-population

Re-estimate on each treated type separately (dropping the other type, keeping the
never-treated controls). New adopters are cohorts 3–5; conversions are cohort 2.

```@example decomp
new_only  = est(df[df.cohort .!= 2.0, :])                              # drop conversions
conv_only = est(df[(df.cohort .== 2.0) .| .!isfinite.(df.cohort), :])  # keep conv + controls
(new = round(new_only.estimate, digits = 3),
 conversion = round(conv_only.estimate, digits = 3))
```

The decomposition exposes the truth:

- **new-only ≈ 0** — correct; new adopters have no effect;
- **conversion-only ≈ 0.8** — the planted confound, recovered;
- the **pooled** estimate was just an `n`-weighted blend of the two, landing positive
  and significant for the wrong reason.

## The lesson

A pooled DiD/spillover estimate can be **composition-driven**: positive and
significant because one sub-population's treatment timing is confounded, not because
the treatment does anything. Whenever a treatment label aggregates heterogeneous
events — new versus converted, voluntary versus mandated, early versus late — estimate
the arms separately. If the pooled effect lives in one arm and vanishes in the other,
the pooled number is telling you about *who is in the group*, not about the treatment.

The interference-DiD framework helps here precisely because it forces the treatment
and exposure to be explicit objects you can subset and re-estimate — turning a hidden
composition artifact into a visible, reportable one.
