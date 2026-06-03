# Critical implementation rules

Load-bearing invariants of `FragmentedTensors.jl`. Each rule has a
concrete failure mode behind it — don't relax without re-reading why.

## 1. `Dictionaries.jl` mutation: use `set!`, not `[]=`

`Dictionaries.jl` differs from `Base.Dict`: `dict[key] = val` is an
**update-only** operation that throws `IndexError` if the key is
absent. Always go through `Dictionaries.set!`:

```julia
Base.setindex!(ft::FragmentedTensor, val, key) = set!(ft.data, key, val)
```

## 2. `VectorInterface.jl` in-place vs out-of-place

- **Mutable values only.** `scale!(va, α)` mutates and is therefore
  valid only for mutable `TensorType` (e.g. `Matrix`, `TensorMap`). For
  immutable types (`Float64`, …) use out-of-place `scale` or
  `scale!!`.
- **`scale!(y, x, α)` must never mutate `x`.** Pre-allocated scaling
  (called inside Krylov solvers in `KrylovKit.jl`) needs three-argument
  `scale!` / `scale!!` overloads that copy out of `x` — use
  `VectorInterface.scale(vx, α)` per element, not `scale!(vx, α)`.

## 3. Type-stable multiplication via `Base.promote_op`

For inhomogeneous storage, the result element type of `*` is computed
at the type level:

```julia
T_res = Base.promote_op(*, T1, T2)
res_data = Dictionary{Tuple{SpaceID{N_out1}, SpaceID{N_in2}}, T_res}()
```

This gives zero overhead for homogeneous storage and safely promotes to
`Any` / `AbstractTensorMap` when fragment types differ.

## 4. `dot` correctness on `Dictionary` — sum over intersection

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

## 5. Composite index lookup `FT[S]`

Look up a composite `SpaceID` key `S` of length `N_out + N_in`:

1. Check `N == N_out + N_in` (throw `DimensionMismatch` otherwise).
2. Decompose: `S_out` = first `N_out` legs, `S_in_prime` = remaining `N_in` legs.
3. Apply adjoint: `S_in = S_in_prime'`.
4. Return `FT[(S_out, S_in)]`.

## 6. `permute` overload (two index levels)

`TensorKit.permute(FT, ((p1...), (p2...)); non_material_labels=String[])`
(and the single-tuple form `permute(FT, (p1...))` ≡
`permute(FT, (p1, ()))`) permutes a `FragmentedTensor`. `p1` / `p2` are
1-based indices into the **combined** key leg list `S_out * S_in'` and
become the new out / in legs.

Two distinct levels, both required:

1. **SpaceID level.** Per key, reindex the combined SpaceID: new out =
   `S_full[p1]`, new in-prime = `S_full[p2]`, then
   `S_in_new = (S_full[p2])'`. Indexing the SpaceID carries the
   per-label adjoint flags along.
2. **TensorKit level.** A stored tensor may have **fewer legs than the
   key has labels**. Use the tensor-leg ↔ key-leg correspondence
   described in [`data-model.md`](data-model.md) to translate `p1` /
   `p2` (positions in the *key*) into the permutation that goes into
   `permute(tensor, (p1_tens, p2_tens))`. Concretely:

   ```julia
   p1_tens = Tuple(key_leg_to_leg[kl] for kl in p1 if haskey(key_leg_to_leg, kl))
   p2_tens = Tuple(key_leg_to_leg[kl] for kl in p2 if haskey(key_leg_to_leg, kl))
   ```

   Non-material key positions are simply dropped — they have no
   counterpart in the tensor.

The two index spaces (key legs vs. tensor legs) must not be conflated —
that's the whole point of the `non_material_labels` mapping. Empty
`FragmentedTensor` short-circuits to an empty result of the new
`(length(p1), length(p2))` shape. Tests: the "Permutation" testset in
`test/runtests.jl`.
