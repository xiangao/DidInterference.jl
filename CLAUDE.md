# DidInterference.jl — project notes for Claude

Julia port of the R package
[`didint`](https://github.com/xiangao/didint). Implements the doubly
robust DiD with spatial interference of Xu (2023, 2026).

The two packages are designed to produce **matching estimates up to MC
noise on the same DGP**. If you change the DR core in one, mirror it
in the other.

## What's where

- `src/dr_atte.jl` — internal `_dr_atte()` helper (DR core).
- `src/did_int_2x2.jl` — Xu (2023) 2×2.
- `src/did_int_dynamic.jl` — Xu (2026) §I event study.
- `src/did_int_staggered.jl` — Xu (2026) §II staggered, with joint-IF
  aggregation across cells. Shares the same subtlety as R: per-cell
  IFs must be stacked to unit IDs across cells, not treated as
  independent.
- `test/runtests.jl` — smoke tests on simulated lattice DGPs.

## Docs

Documenter.jl-based. `docs/make.jl` builds, `.github/workflows/docs.yml`
deploys to `gh-pages` on push. Live at
<https://xiangao.github.io/DidInterference.jl/dev/>.

**Reference page pattern**: every function in `docs/src/reference.md`
gets a `@docs` block immediately followed by an `@example` block (so
the page shows API + live output with real numbers). Docstring
`# Examples` blocks render as text only — they don't execute.

## Pages config

GH Pages source = `gh-pages` branch root (set once via `gh api`;
already done).

## Mirroring R changes

When `didint` (R) changes:
1. Replicate the equivalent fix in the matching `src/*.jl` file.
2. Re-run `julia --project=. test/runtests.jl`.
3. Update any docstring examples that depend on the changed behaviour.
