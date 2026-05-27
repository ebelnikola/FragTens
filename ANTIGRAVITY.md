# Antigravity Package Guide: Fragmented Tensors

This guide serves as the specialized technical reference for the `FragmentedTensors` package. It documents core architectural decisions, data structures, and mathematical conventions to maintain type-stability and correctness in future development.

## Core Package Architecture & Structures

### 1. `SpaceID{N}`
Represents a labeled boundary space with `N` legs.
```julia
struct SpaceID{N}
    labels::NTuple{N,String}
    adjoint::NTuple{N,Bool}
end
```
- **Adjoint (`S'`)**: Performs an out-of-place bit-flip of all `adjoint` booleans.
- **Multiplication (`S1 * S2`)**: Concatenates `labels` and `adjoint` tuples, returning a `SpaceID{N1+N2}`.
- **`NullSpaceID`**: A special constant for empty boundaries, represented as `SpaceID{0}((), ())`.
- **Macro (`sid"..."`)**: Parses comma-separated string literals (e.g., `sid"A, B', C"`) into `SpaceID`s. An empty string (`sid""`) evaluates directly to `NullSpaceID`.

### 2. `FragmentedTensor{N_out, N_in, TensorType}`
Underlying storage maps boundary pairs to tensor structures:
```julia
struct FragmentedTensor{N_out, N_in, TensorType}
    data::Dictionary{Tuple{SpaceID{N_out},SpaceID{N_in}},TensorType}
end
```
- **Heterogeneity (`TensorType`)**: The `TensorType` parameter is completely unrestricted (can be a `Matrix`, a concrete `TensorMap`, or abstract types like `Any` and `AbstractTensorMap`).
- **N-Leg Verification**: Automatically verifies that keys match `N_out` and `N_in` dimensions at the type level.

---

## Critical Implementation Rules

### 1. Dictionaries.jl Mutation via `set!`
`Dictionaries.jl` differs significantly from standard Julia `Dict`s. Calling `dict[key] = val` is strictly an *update-only* operation and throws an `IndexError` if the key is not already present.
- **Rule**: Always use `Dictionaries.set!(dict, key, val)` for standard setting/insertion:
  ```julia
  Base.setindex!(ft::FragmentedTensor, val, key) = set!(ft.data, key, val)
  ```

### 2. In-Place vs Out-of-Place Vector Space Math
When overloading `VectorInterface.jl` methods:
- **Mutable vs Immutable**: In-place mutation of the underlying values (e.g. `scale!(va, α)`) **only works for mutable types** (like `Matrix` or `TensorMap`). For immutable value types (like `Float64`), use out-of-place `scale` or reassigning `scale!!`.
- **3-Argument Scale (`scale!(y, x, α)`)**: Pre-allocated scaling (like those called in Krylov subspace solvers in `KrylovKit.jl`) must **never mutate the source tensor `x`**. Thus, the 3-argument overloads `scale!(y, x, α)` and `scale!!(y, x, α)` must use the out-of-place `VectorInterface.scale(vx, α)` for their elements rather than `scale!(vx, α)` / `scale!!(vx, α)` to keep `x` completely untouched.

### 3. Type-Stable Matrix Multiplication via `Base.promote_op`
To correctly support **inhomogeneous tensors** (where different fragments inside the dictionary have varying leg counts or space dimensions), we determine the value type of the result dictionary dynamically using `Base.promote_op`:
```julia
T_res = Base.promote_op(*, T1, T2)
res_data = Dictionary{Tuple{SpaceID{N_out1}, SpaceID{N_in2}}, T_res}()
```
This guarantees:
- Zero overhead and perfect type-stability for homogeneous storage.
- Safe type promotion to abstract types (like `Any` or `AbstractTensorMap`) for heterogeneous/inhomogeneous storage, permitting blocks of different sizes and legs to coexist in the result.

### 4. Dictionary `dot` (Inner Product) Correctness — sum over intersection
Generic `dot(d1, d2)` on `Dictionary` in `LinearAlgebra` falls back to iterating over values in insertion order, **ignoring the dictionary keys**.
- **Philosophy**: Two `FragmentedTensor`s with the same `(N_out, N_in)` live in the same direct-sum space `⊕_S V_S`; a key that's missing from the dictionary is a *true zero* in that direct sum, not a "shape mismatch". This is the same reason `+` uses `mergewith` (the union of keysets) and `*` (FT × FT) accumulates over matching middle SpaceIDs.
- **Rule**: Compute `dot` by iterating over the *intersection* of the two keysets — terms outside the intersection have at least one zero factor and contribute zero:
  ```julia
  res = zero(promote_type(scalartype(TA), scalartype(TB)))
  for k in keys(a.data)
      if haskey(b.data, k)
          res += dot(a.data[k], b.data[k])
      end
  end
  ```
- **Anti-pattern**: do *not* `issetequal`-check and either error or return zero on mismatch. That contradicts the direct-sum interpretation and silently kills AD pullbacks where a cotangent legitimately has a larger keyset than the primal (e.g. cotangents from `assemble_global` span `OUT × IN`).

### 5. Composite Index Lookup (`FT[S]`)
To look up a composite `SpaceID` key `S` of length `N_out + N_in`:
1. Check `N == N_out + N_in` (throw `DimensionMismatch` otherwise).
2. Decompose `S` into `S_out` (first `N_out` legs) and `S_in_prime` (remaining `N_in` legs).
3. Apply `'` to `S_in_prime` to get `S_in`.
4. Return `FT[(S_out, S_in)]`.

### 6. Self-adjoint space fallback in `assemble_global`
When `assemble_global(A, OUT, IN)` is called with `OUT = IN = SPACES = union(...)`
(the eigh / eig path), a `SpaceID` `S` may appear on one side but not the other
in `A.data`. The missing-side entry inherits the *same* `W` from the present side
— **not** `W'`:
```julia
W_out = haskey(space_dict_out, S) ? space_dict_out[S] : space_dict_in[S]
W_in  = haskey(space_dict_in,  S) ? space_dict_in[S]  : space_dict_out[S]
```
The test convention builds hermitian fragments as `TensorMap(W ← W)` (codom and
dom are the *same* `W`, not duals). Using `space_dict_in[S]'` would break
downstream `A * U_g` with a `SpaceMismatch`.

### 7. Decomposition wrappers are plain functions; no custom decomposition rrules
`eigh_full` / `eig_full` / `svd_full` etc. on `FragmentedTensor` are plain
Julia functions traced by Zygote. The only ChainRules rrules at this layer
are at the **assemble/disassemble boundary**:
- `assemble_global`, `disassemble_global_U`, `disassemble_global_V`
- `FragmentedTensor{N,M,T}(::Dictionary)` constructor and
  `wrap_singleton_frag(tensor, S_eigen)` (the singleton-boxing helper)
- `Base.adjoint`, `Base.:*(::FT, ::FT)`, `Base.inv(::FT{N,1,T})`
- arithmetic (`+`, `-`, `*` scalar, `dot`)

Pure structural helpers `extract_out_in_spaces`, `extract_out_in`,
`make_S_eigen`, `combined_codom_dict` are registered
`@non_differentiable` (they consume `SpaceID`s, which carry no gradient).

The inner `eigh_full(H_global)` call uses MatrixAlgebraKit's own pullback,
which is gauge-invariant by construction (`inv_safe(d_i - d_j, ε)` zeros out
the would-be-singular gauge-fragile contributions when eigenvalues are within
`ε`). For non-degenerate spectra MAKit's AD is correct exactly; for
degenerate cases its `@warn` flags that the user's cotangent has
gauge-sensitive components.

### 8. Matrix inverse on eig-shape FragmentedTensor
`Base.inv(U::FragmentedTensor{N_out, 1, T})` is the inverse of the
"eigenvector" shape (keys `(S_k, S_eigen)`). Implementation: assemble
`U_global`, call TensorKit's `inv`, disassemble the result into the
V-shape (keys `(S_eigen, S_k)`). The rrule uses the matrix-inverse
adjoint identity `dX = -Y' * dY * Y'`. Required for the gauge-invariant
`eig` AD loss `real(dot(U * D * inv(U), M))`.

For non-square `U_kept` from `eig_trunc` this generalises to a
pseudoinverse — *not yet implemented*; `eig_trunc` AD is skipped in the
current test suite. See `~/.claude/plans/factorizations-test-plan.md`.

### 9. Test suite is fuzz-based
The factorisation tests are organised as a small structural catalogue of
scenarios; each scenario runs `N_REPS = 40` random repetitions and reports
the per-decomposition `pass / excused_ill_cond / excused_near_boundary /
real_fail` breakdown via `@info`. AD-vs-FD failures are excused only by
two specific structural conditions:
- `cond(U) > 1e6` for the `inv(U)`-using losses (`eig_full`, `eig_trunc`);
- `min |kept eigenvalue| − atol < 1e-6` for truncated decompositions.
Anything else is a real failure.

The losses are gauge-invariant by design (each is a reconstruction of `A`
through the decomposition, dotted with a fixed random `M`), so sort-order
fragility — the dominant cause of historical false failures — cannot occur.

**Varying-rank coverage**: "Group F" adds a `SpaceID{5}` whose `"r"` legs don't
materialise (rank 5/3/1 fragments), hermitian and non-hermitian, mirroring
`failing_example.data` at reduced dims. Convention: a label absent from a
scenario's `space_dict` is a non-materialising leg. There's also an explicit
no-AD reconstruction guard in the "Multi-Space" testset (§8).

### 10. Rank-agnostic container types in the factorizations
A `SpaceID{N}` can have legs that don't materialise as tensor legs, so fragments
sharing one `N` may have tensor rank `< N`. **Never pin the materialised rank**
when building the assemble/disassemble containers:
```julia
ProductSpace{S_type}                              # NOT ProductSpace{S_type, N_out}
TensorMap{T, S_type, N₁, N₂, Vector{T}} where {N₁, N₂}   # NOT {…, N_out, N_in, …}
```
Pinning to `N` throws a `convert` `MethodError` on the first non-full-rank
fragment. This bit `assemble_global`, `disassemble_global`,
`disassemble_global_U/_V`, `assemble_U_frag`, **and** the `dot` rrule — the
last typed its cotangent from `typeof(first(da_data))` (the first fragment's
concrete rank); use the dict's own `eltype(...)` instead. Rule of thumb: derive
a result `FragmentedTensor`'s `TensorType` from the dict `eltype`,
`Base.promote_op`, or a free `where` UnionAll — never from a single fragment.
