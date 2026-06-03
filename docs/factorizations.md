# Factorizations

Decompositions over `FragmentedTensor`. The pattern is the same for
all of them: assemble fragments into a single global `TensorMap`, run
TensorKit / MatrixAlgebraKit's factorization, disassemble the results
back into fragmented form. Lives in `src/factorizations.jl`.

Provided: `eigh_full`, `eigh_trunc`, `eig_full`, `eig_trunc`,
`svd_full`, `svd_compact`, `svd_trunc`, plus `Base.inv` on the
eig-shape result.

## Wrapper design (Zygote-friendly)

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

## Assemble / disassemble

`assemble_global(A, OUT, IN)` returns `(H_global, metadata)`. The
metadata tuple is `(offsets_out, offsets_in, V_global_out,
V_global_in, active_sectors, space_dict_out, space_dict_in)`. The
`disassemble_*` helpers and their ChainRules pullbacks
`assemble_global_*` invert the operation column-wise (for `V_global`
factors) and row-wise (for `U_global` factors).

## Self-adjoint space fallback

In Hermitian / square decompositions (`eigh_full`, `eig_full`) the
package builds `SPACES = collect(union(OUT, IN))` and calls
`assemble_global(A, SPACES, SPACES)`. A space `S` may only appear as a
domain entry (in `IN`) or only as a codomain entry (in `OUT`).

**Fallback uses the *same* `W`, not its dual.** Per this codebase's
convention (test helpers build hermitian fragments as
`TensorMap(W ← W)` with codomain and domain the *same* `W`, not
duals), the missing entry inherits `W` directly:

```julia
W_out = haskey(space_dict_out, S) ? space_dict_out[S] : space_dict_in[S]
W_in  = haskey(space_dict_in,  S) ? space_dict_in[S]  : space_dict_out[S]
```

Using `space_dict_in[S]'` (the dual) would shift `U_g`'s codomain into
the dual space and break the downstream `A * U_g` reconstruction with
a `SpaceMismatch`. Don't re-introduce the prime.

## Rank-agnostic container types

Because a `SpaceID{N}` may have legs that don't materialise (see
[`data-model.md`](data-model.md)), fragments sharing one `N` can have
tensor rank `< N`. **Every dict / `TensorMap` value type built inside
the assemble/disassemble machinery must leave the materialised rank
free**, otherwise the first non-full-rank fragment throws a `convert`
`MethodError`:

```julia
CodomType = ProductSpace{S_type}                              # NOT {S_type, N_out}
T_val     = TensorMap{T, S_type, N₁, N₂, Vector{T}} where {N₁, N₂}  # U: {…,N₁,1,…}; V: {…,1,N₂,…}
space_dict = Dict{SpaceID{N_out}, ProductSpace{S_type}}()     # in assemble_U_frag
```

This rule covers `assemble_global`, `disassemble_global`,
`disassemble_global_U/_V`, and `assemble_U_frag` (the last feeds
`inv`, hence `eig_full` reconstruction). The **AD path has the same
trap once more**: the `dot` rrule must type its cotangent
`FragmentedTensor` from the mapped dict's own `eltype(...)` — *never*
`typeof(first(da_data))`, which pins the first fragment's concrete
rank and forces the inner-constructor `convert` to throw. General
lesson: don't derive a result `FragmentedTensor`'s `TensorType` from
one fragment; use the dict `eltype`, `Base.promote_op`, or a free
`where` UnionAll.

## Non-differentiable structural helpers

`extract_out_in_spaces`, `extract_out_in`, `make_S_eigen`,
`combined_codom_dict` are pure structural extraction over `SpaceID`s
and are registered `@non_differentiable`. Without this, Zygote tries
to trace through `Set`, `union(...)`, and `push!`, complains about
mutation, and breaks AD.

## Matrix inverse on the eig-shape FragmentedTensor

`Base.inv(U::FragmentedTensor{N_out, 1, T})` is defined: it assembles
`U_global`, calls TensorKit's `inv`, and disassembles the result into
a V-shape `FragmentedTensor{1, N_out, T_inv}` keyed by
`(S_eigen, S_k)`. The rrule uses the matrix-inverse adjoint identity
`dX = -Y' · dY · Y'`.

This is what makes the gauge-invariant `eig_full` loss
`real(dot(U * D * inv(U), M))` differentiable. For non-square
`U_kept` from `eig_trunc` the analogous loss requires a pseudoinverse,
which is not yet implemented — `eig_trunc` AD comparison is therefore
skipped in the test suite (see
`~/.claude/plans/factorizations-test-plan.md`).

## Pair-based user-facing constructor

```julia
FragmentedTensor((S_out, S_in) => t1, (S_out', S_in') => t2, …)
```

is supported (with `ChainRulesCore.rrule`) so users can construct
fragmented tensors fluently in Zygote-differentiable code.
