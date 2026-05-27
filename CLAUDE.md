# FragmentedTensors.jl — package guide

`FragmentedTensors.jl` is a standalone Julia package providing block-sparse,
heterogeneous tensor structures keyed by labeled boundary spaces. It is consumed
by the web GUI's generated Julia code (`web/src/transforms/deckCodeGenerationCore.js`,
`diagramCodeGeneration.js`) and is also used directly for matrix factorizations
and AD-aware decompositions.

This file is the high-level orientation. For the fast-operation reference and
hand-off-style summaries see [`ANTIGRAVITY.md`](ANTIGRAVITY.md).

## Layout

- `src/FragmentedTensors.jl` — module definition, `SpaceID{N}`,
  `FragmentedTensor{N_out, N_in, TensorType}`, arithmetic, indexing,
  `VectorInterface.jl` overloads.
- `src/factorizations.jl` — `eigh_full` / `eigh_trunc` / `eig_full` /
  `eig_trunc` / `svd_full` / `svd_compact` / `svd_trunc` for
  `FragmentedTensor`, plus the assemble / disassemble helpers and
  `ChainRulesCore` pullbacks for Zygote AD.
- `test/runtests.jl` — assembled test suite (SpaceID arithmetic,
  VectorInterface, TensorKit interop, factorizations, AD).

## Core data structures

### `SpaceID{N}`

```julia
struct SpaceID{N}
    labels::NTuple{N,String}
    adjoint::NTuple{N,Bool}
end
```

Represents a labeled boundary space with `N` legs.

- **Adjoint (`S'`)** — out-of-place bit-flip of every `adjoint` boolean.
- **Multiplication (`S1 * S2`)** — concatenates `labels` and `adjoint`,
  returning `SpaceID{N1+N2}`.
- **`NullSpaceID`** — constant for empty boundaries, `SpaceID{0}((), ())`.
- **Macro `sid"..."`** — parses comma-separated literals
  (`sid"A, B', C"`). Empty literal `sid""` yields `NullSpaceID`.

### `FragmentedTensor{N_out, N_in, TensorType}`

```julia
struct FragmentedTensor{N_out, N_in, TensorType}
    data::Dictionary{Tuple{SpaceID{N_out},SpaceID{N_in}},TensorType}
end
```

- **`TensorType` is unconstrained.** Can be a concrete `Matrix` /
  `TensorMap`, a union (`Union{DiagonalTensorMap, TensorMap}`), or an
  abstract supertype (`Any`, `AbstractTensorMap`). This is the price for
  supporting **inhomogeneous storage** — fragments at different keys may
  carry different leg counts or space dimensions.
- **A fragment's tensor rank can be *less than* `N`.** A `SpaceID{N}` may
  carry labels that don't materialise as tensor legs (e.g. an `"r"`
  placeholder leg). So two keys sharing the same `N_out`/`N_in` can map to
  `TensorMap`s of *different* rank (5, 3, 1, …). `failing_example.data` (a
  self-adjoint `FragmentedTensor{5,5}`) is the canonical case. **Consequence:**
  any container keyed inside the factorizations must leave the tensor /
  `ProductSpace` rank free — never pin it to `N`. See *Rank-agnostic
  container types* below.
- **N-leg verification.** Inner constructor checks every key's two
  `SpaceID`s match `N_out` and `N_in` respectively (it checks the *key*
  arity, not the stored tensor's rank — those may legitimately differ, per
  the previous bullet).

## Critical implementation rules

### 1. `Dictionaries.jl` mutation: use `set!`, not `[]=`

`Dictionaries.jl` differs from `Base.Dict`: `dict[key] = val` is an
**update-only** operation that throws `IndexError` if the key is
absent. Always go through `Dictionaries.set!`:

```julia
Base.setindex!(ft::FragmentedTensor, val, key) = set!(ft.data, key, val)
```

### 2. `VectorInterface.jl` in-place vs out-of-place

- **Mutable values only.** `scale!(va, α)` mutates and is therefore
  valid only for mutable `TensorType` (e.g. `Matrix`, `TensorMap`). For
  immutable types (`Float64`, …) use out-of-place `scale` or
  `scale!!`.
- **`scale!(y, x, α)` must never mutate `x`.** Pre-allocated scaling
  (called inside Krylov solvers in `KrylovKit.jl`) needs three-argument
  `scale!` / `scale!!` overloads that copy out of `x` — use
  `VectorInterface.scale(vx, α)` per element, not `scale!(vx, α)`.

### 3. Type-stable multiplication via `Base.promote_op`

For inhomogeneous storage, the result element type of `*` is computed
at the type level:

```julia
T_res = Base.promote_op(*, T1, T2)
res_data = Dictionary{Tuple{SpaceID{N_out1}, SpaceID{N_in2}}, T_res}()
```

This gives zero overhead for homogeneous storage and safely promotes to
`Any` / `AbstractTensorMap` when fragment types differ.

### 4. `dot` correctness on `Dictionary` — sum over intersection

`LinearAlgebra.dot(d1, d2)` falls back to iterating values **in
insertion order, ignoring keys** — wrong for fragmented tensors.

**Philosophy.** Two `FragmentedTensor`s with the same `(N_out, N_in)`
live in the same direct-sum space `⊕_S V_S`. A missing key is a *true
zero* in that space, not a shape mismatch. This is the same convention
behind `+` using `mergewith` (union of keys) and matrix `*` accumulating
over matching middle SpaceIDs. The right `dot` therefore sums over the
**intersection** of the two keysets — terms outside have at least one
zero factor:

```julia
res = zero(promote_type(scalartype(TA), scalartype(TB)))
for k in keys(a.data)
    if haskey(b.data, k)
        res += dot(a.data[k], b.data[k])
    end
end
```

**Anti-pattern.** Don't `issetequal`-check and error / return zero on
mismatch. That kills AD silently — Zygote pullbacks legitimately produce
cotangents whose keyset is larger than the primal (e.g. the cotangent
of `assemble_global(A, OUT, IN)` spans `OUT × IN`, which can exceed
`keys(A.data)`). With the buggy strict-equality dot, those gradients
would collapse to zero.

### 5. Composite index lookup `FT[S]`

Look up a composite `SpaceID` key `S` of length `N_out + N_in`:

1. Check `N == N_out + N_in` (throw `DimensionMismatch` otherwise).
2. Decompose: `S_out` = first `N_out` legs, `S_in_prime` = remaining `N_in` legs.
3. Apply adjoint: `S_in = S_in_prime'`.
4. Return `FT[(S_out, S_in)]`.

## Factorizations (`src/factorizations.jl`)

Provides decompositions over `FragmentedTensor` by assembling fragments
into a single global `TensorMap`, running TensorKit / MatrixAlgebraKit's
factorization, and disassembling the results back into fragmented form.

### Wrapper design (Zygote-friendly)

`eigh_full`, `eigh_trunc`, `eig_full`, `eig_trunc`, `svd_full`,
`svd_compact`, `svd_trunc` are plain Julia functions. There are *no*
custom rrules on them — Zygote traces through their bodies natively.
The custom rrules live only at the assemble/disassemble boundary:

- `rrule(::typeof(assemble_global), A, OUT, IN)` — cotangent of
  `H_global` flows back to a `FragmentedTensor` cotangent via
  `disassemble_global`.
- `rrule(::typeof(disassemble_global_U), …)` and
  `rrule(::typeof(disassemble_global_V), …)` — inverse of the above
  for the U / V factors.
- `rrule(::Type{FragmentedTensor{…}}, ::Dictionary)` — Zygote can't
  derive constructor adjoints when the cotangent is itself a
  `FragmentedTensor`; this rule extracts the inner `data` field.
- `rrule(::typeof(wrap_singleton_frag), tensor, S_eigen)` — the
  singleton boxing of `D` / `S` inside the decomposition wrappers
  goes through `Dictionary([…], […])` and `FragmentedTensor{…}(data)`
  in sequence; Zygote can't trace that chain naturally, so we expose
  the whole boxing as one named function with one rrule that pulls
  the inner tensor back out.

Inside the decomposition body the inner call `eigh_full(H_global)`
(now on a TensorMap) uses MatrixAlgebraKit's own pullback, which is
gauge-invariant by construction and handles degeneracies via
`inv_safe` (zeros out `1/(d_i - d_j)` when the gap is below
`degeneracy_atol`).

### Assemble / disassemble

`assemble_global(A, OUT, IN)` returns `(H_global, metadata)`. The metadata
tuple is `(offsets_out, offsets_in, V_global_out, V_global_in, active_sectors,
space_dict_out, space_dict_in)`. The `disassemble_*` helpers and their
ChainRules pullbacks `assemble_global_*` invert the operation column-wise
(for `V_global` factors) and row-wise (for `U_global` factors).

### Self-adjoint space fallback

In Hermitian / square decompositions (`eigh_full`, `eig_full`) the
package builds `SPACES = collect(union(OUT, IN))` and calls
`assemble_global(A, SPACES, SPACES)`. A space `S` may only appear as a
domain entry (in `IN`) or only as a codomain entry (in `OUT`).

**Fallback uses the *same* `W`, not its dual.** Per this codebase's
convention (test helpers build hermitian fragments as `TensorMap(W ← W)`
with codomain and domain the *same* `W`, not duals), the missing entry
inherits `W` directly:

```julia
W_out = haskey(space_dict_out, S) ? space_dict_out[S] : space_dict_in[S]
W_in  = haskey(space_dict_in,  S) ? space_dict_in[S]  : space_dict_out[S]
```

Using `space_dict_in[S]'` (the dual) would shift `U_g`'s codomain into
the dual space and break the downstream `A * U_g` reconstruction with
a `SpaceMismatch`. Don't re-introduce the prime.

### Rank-agnostic container types

Because a `SpaceID{N}` may have legs that don't materialise (see the data-model
note), fragments sharing one `N` can have tensor rank `< N`. **Every dict /
`TensorMap` value type built inside the assemble/disassemble machinery must
leave the materialised rank free**, otherwise the first non-full-rank fragment
throws a `convert` `MethodError`:

```julia
CodomType = ProductSpace{S_type}                              # NOT {S_type, N_out}
T_val     = TensorMap{T, S_type, N₁, N₂, Vector{T}} where {N₁, N₂}  # U: {…,N₁,1,…}; V: {…,1,N₂,…}
space_dict = Dict{SpaceID{N_out}, ProductSpace{S_type}}()     # in assemble_U_frag
```

This rule covers `assemble_global`, `disassemble_global`,
`disassemble_global_U/_V`, and `assemble_U_frag` (the last feeds `inv`, hence
`eig_full` reconstruction). The **AD path has the same trap once more**: the
`dot` rrule must type its cotangent `FragmentedTensor` from the mapped dict's
own `eltype(...)` — *never* `typeof(first(da_data))`, which pins the first
fragment's concrete rank and forces the inner-constructor `convert` to throw.
General lesson: don't derive a result `FragmentedTensor`'s `TensorType` from one
fragment; use the dict `eltype`, `Base.promote_op`, or a free `where` UnionAll.

### Non-differentiable structural helpers

`extract_out_in_spaces`, `extract_out_in`, `make_S_eigen`,
`combined_codom_dict` are pure structural extraction over `SpaceID`s
and are registered `@non_differentiable`. Without this, Zygote tries
to trace through `Set`, `union(...)`, and `push!`, complains about
mutation, and breaks AD.

### Matrix inverse on the eig-shape FragmentedTensor

`Base.inv(U::FragmentedTensor{N_out, 1, T})` is defined: it assembles
`U_global`, calls TensorKit's `inv`, and disassembles the result into a
V-shape `FragmentedTensor{1, N_out, T_inv}` keyed by `(S_eigen, S_k)`.
The rrule uses the matrix-inverse adjoint identity `dX = -Y' · dY · Y'`.

This is what makes the gauge-invariant `eig_full` loss
`real(dot(U * D * inv(U), M))` differentiable. For non-square
`U_kept` from `eig_trunc` the analogous loss requires a pseudoinverse,
which is not yet implemented — `eig_trunc` AD comparison is therefore
skipped in the test suite (see `~/.claude/plans/factorizations-test-plan.md`).

### Pair-based user-facing constructor

```julia
FragmentedTensor((S_out, S_in) => t1, (S_out', S_in') => t2, …)
```

is supported (with `ChainRulesCore.rrule`) so users can construct
fragmented tensors fluently in Zygote-differentiable code.

## Testing

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The factorisation tests use a **fuzz design**: each scenario in the
catalogue is a structural family (space dict, key pattern, hermitian
flag) and is exercised by `N_REPS = 40` repetitions with fresh random
`A`, perturbation `Δ`, weight `M`. Per-decomposition outcomes are
classified as `pass / excused_ill_cond / excused_near_boundary /
real_fail`; a scenario passes iff there are *no* real failures and at
least one rep was not excused. Reported as one `@testset` per scenario
with `@info` breakdown. Plan: `~/.claude/plans/factorizations-test-plan.md`.

The AD comparison uses **gauge-invariant losses** (reconstruction of
`A` via the decomposition, dotted with a random fixed `M`) so the
sort-ordered eigenvalue function is irrelevant; the math is smooth
everywhere the decomposition exists.

Excuse policies fire only on genuine numerical degeneracy:
- `eig_full` / `eig_trunc`: `cond(U) > 1e6` (the only places `inv(U)` is used).
- `eigh_trunc` / `svd_trunc`: truncation-boundary gap
  `min |kept| − atol < 1e-6`.

**Varying-rank fragments are covered in two places** (regression guard for the
rank-pinning bugs above):
- *Fuzz* "Group F" — two scenarios (`dense herm`, `dense non-herm`) over a
  `SpaceID{5}` whose `"r"` legs don't materialise (rank 5/3/1 fragments),
  mirroring `failing_example.data` at reduced dims. These also exercise the
  **AD** path, which is what catches the `dot`-pullback variant. The fuzz
  helper convention: **a label absent from a scenario's `space_dict` is a
  non-materialising leg** (`get_product_space` skips it); `build_random_A`
  therefore uses a rank-free `where {N₁,N₂}` dict value type. Heavy scenarios
  can set a per-scenario `reps` field to override `N_REPS`.
- *Explicit* "Multi-Space" testset §8 — a fast, no-AD reconstruction guard for
  the same structure (hermitian `eigh_full`; non-hermitian `eig_full`/`inv`/
  `svd_full`).

## Consumer interface (web GUI)

Generated code from the web GUI imports `FragmentedTensors`:

- `eval_<deck>(; tensor1=tensor1, ...)` returns a `FragmentedTensor`
  keyed by `(sid"out", sid"in")` tuples — one entry per
  (out-space, in-space) pair occurring across the deck's diagrams.
- `apply_<deck>(v; ...)` and `apply_adjoint_<deck>(v; ...)` act on
  vectors / co-vectors. Their results are also `FragmentedTensor`s
  keyed by `(sid"...", NullSpaceID)` pairs.
- Random test vectors are generated as `FragmentedTensor`s with the
  same `(sid"...", NullSpaceID)` keying.

`NullSpaceID` is the marker for the empty boundary side of an
apply/random-vector result; the non-empty side uses ordinary
`sid"..."` literals.
