# Testing

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Fuzz design

The factorisation tests use a **fuzz design**: each scenario in the
catalogue is a structural family (space dict, key pattern, hermitian
flag) and is exercised by `N_REPS = 40` repetitions with fresh random
`A`, perturbation `Δ`, weight `M`. Per-decomposition outcomes are
classified as `pass / excused_ill_cond / excused_near_boundary /
real_fail`; a scenario passes iff there are *no* real failures and at
least one rep was not excused. Reported as one `@testset` per scenario
with `@info` breakdown. Plan:
`~/.claude/plans/factorizations-test-plan.md`.

## Gauge-invariant losses

The AD comparison uses **gauge-invariant losses** (reconstruction of
`A` via the decomposition, dotted with a random fixed `M`) so the
sort-ordered eigenvalue function is irrelevant; the math is smooth
everywhere the decomposition exists.

## Excuse policies

Excuse policies fire only on genuine numerical degeneracy:

- `eig_full` / `eig_trunc`: `cond(U) > 1e6` (the only places `inv(U)`
  is used).
- `eigh_trunc` / `svd_trunc`: truncation-boundary gap
  `min |kept| − atol < 1e-6`.

## Varying-rank fragment coverage

Covered in two places (regression guard for the rank-pinning bugs in
[`factorizations.md`](factorizations.md#rank-agnostic-container-types)):

- *Fuzz* "Group F" — two scenarios (`dense herm`, `dense non-herm`)
  over a `SpaceID{5}` whose `"r"` legs don't materialise (rank 5/3/1
  fragments), mirroring `failing_example.data` at reduced dims. These
  also exercise the **AD** path, which is what catches the
  `dot`-pullback variant. The fuzz helper convention: **a label
  absent from a scenario's `space_dict` is a non-materialising leg**
  (`get_product_space` skips it); `build_random_A` therefore uses a
  rank-free `where {N₁,N₂}` dict value type. Heavy scenarios can set
  a per-scenario `reps` field to override `N_REPS`.
- *Explicit* "Multi-Space" testset §8 — a fast, no-AD reconstruction
  guard for the same structure (hermitian `eigh_full`; non-hermitian
  `eig_full`/`inv`/`svd_full`).
