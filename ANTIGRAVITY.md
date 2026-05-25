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

### 4. Dictionary `dot` (Inner Product) Correctness
Generic `dot(d1, d2)` on `Dictionary` in `LinearAlgebra` falls back to iterating over values in insertion order, **ignoring the dictionary keys**.
- **Rule**: To compute the exact dot product between two fragmented tensors, first verify that their key sets are equal via `issetequal(keys(a.data), keys(b.data))`, and then compute the dot product key-by-key:
  ```julia
  res = zero(promote_type(scalartype(TA), scalartype(TB)))
  for k in keys(a.data)
      res += dot(a.data[k], b.data[k])
  end
  ```

### 5. Composite Index Lookup (`FT[S]`)
To look up a composite `SpaceID` key `S` of length `N_out + N_in`:
1. Check `N == N_out + N_in` (throw `DimensionMismatch` otherwise).
2. Decompose `S` into `S_out` (first `N_out` legs) and `S_in_prime` (remaining `N_in` legs).
3. Apply `'` to `S_in_prime` to get `S_in`.
4. Return `FT[(S_out, S_in)]`.
