# FragmentedTensors data model

The two core types and the rule for reading a fragment's leg structure.

## `SpaceID{N}`

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

## `FragmentedTensor{N_out, N_in, TensorType}`

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
  self-adjoint `FragmentedTensor{5,5}`) is the canonical case.
  **Consequence:** any container keyed inside the factorizations must
  leave the tensor / `ProductSpace` rank free — never pin it to `N`. See
  *Rank-agnostic container types* in
  [`factorizations.md`](factorizations.md).
- **N-leg verification.** Inner constructor checks every key's two
  `SpaceID`s match `N_out` and `N_in` respectively (it checks the *key*
  arity, not the stored tensor's rank — those may legitimately differ,
  per the previous bullet).

## Tensor-leg ↔ key-leg mapping

A `SpaceID{N}` may carry labels that don't materialise as tensor legs
(e.g. an `"r"` placeholder, or any infinite-dimension label). The caller
passes the set of such labels as `non_material_labels` when an operation
needs to navigate from key positions to tensor legs — it is **not**
stored on the `FragmentedTensor` itself, so the same fragment can be
interpreted with different non-material sets in different contexts.

**Rule.** Walk the combined key SpaceID `S_out * S_in'` left-to-right.
Skip every position whose label string is in `non_material_labels`. The
remaining positions — the *material* labels, in that same order —
correspond one-to-one with the stored tensor's legs (codomain first,
then domain).

The implementation helper is:

```julia
labels_full     = (S_out * S_in').labels         # length N_out + N_in
leg_to_key_leg  = findall(x -> !(x in nonmat), labels_full)
# tensor leg i  ↔  key position leg_to_key_leg[i]
```

The order guarantee here relies on `findall` returning matching
positions in `eachindex` order, which it does for `AbstractArray`
inputs (and `labels_full` is a `Vector`). The inverse map

```julia
key_leg_to_leg = Dict(key_leg => leg for (leg, key_leg) in enumerate(leg_to_key_leg))
```

takes you from a key position to its tensor leg (it has no entry for
positions that are non-material).

**Example.** Key `(sid"A, r, B", sid"C, r")` with
`non_material_labels = ["r"]`:

- Combined SpaceID `S_out * S_in'` has labels `(A, r, B, C', r')` —
  5 key-leg positions.
- Material positions: 1 (`A`), 3 (`B`), 4 (`C'`). The two `r`s drop out.
- The stored tensor has rank 3:
  - tensor leg 1 ↔ key leg 1 (`A`, codomain side)
  - tensor leg 2 ↔ key leg 3 (`B`, codomain side)
  - tensor leg 3 ↔ key leg 4 (`C`, domain side — the prime came from
    taking `S_in'`)
- TensorMap shape: 2 codomain legs (material legs falling inside
  `S_out`) and 1 domain leg (material leg inside `S_in`).

So `N_out` / `N_in` in the type signature count *key* legs, not tensor
legs — the tensor's `(M_out, M_in)` ≤ `(N_out, N_in)`, with the
inequality strict whenever a non-material label sits in that side of
the key.
