# FragmentedTensors.jl

A standalone, type-stable Julia package for working with **fragmented tensor networks** — collections of tensors indexed by pairs of boundary space identifiers (`SpaceID`).

## Overview

A `FragmentedTensor` is a dictionary mapping `Tuple{SpaceID{N_out}, SpaceID{N_in}}` keys to `TensorType` values. The type of the underlying tensors (`TensorType`) is completely unrestricted.

```julia
struct FragmentedTensor{N_out, N_in, TensorType}
    data::Dictionary{Tuple{SpaceID{N_out}, SpaceID{N_in}}, TensorType}
end
```

**Type parameters:**
- `N_out` — Number of legs in the out (codomain) `SpaceID`
- `N_in` — Number of legs in the in (domain) `SpaceID`
- `TensorType` — The type of the underlying tensors (unrestricted: can be `TensorMap`, `Matrix`, etc.)

## SpaceID

A `SpaceID{N}` labels a boundary space with `N` named legs, each optionally marked as adjoint:

```julia
struct SpaceID{N}
    labels::NTuple{N, String}
    adjoint::NTuple{N, Bool}
end
```

Create `SpaceID` values using the `sid"..."` string macro:

```julia
using FragmentedTensors

s1 = sid"A, B', C"    # 3 legs: A (normal), B (adjoint), C (normal)
s2 = sid"X, Y"        # 2 legs: X, Y (both normal)
s_empty = sid""       # empty SpaceID (NullSpaceID)
```

### SpaceID Operators
- **Adjoint (`'`)**: Flips all adjoint booleans. `(sid"A, B'")' == sid"A', B"`.
- **Multiplication (`*`)**: Concatenates SpaceIDs. `sid"A, B" * sid"C" == sid"A, B, C"`.

---

## Quick Start

```julia
using FragmentedTensors, TensorKit

# Create random FragmentedTensor with 1 out-leg, 1 in-leg per fragment
function make_op(k::Tuple{SpaceID{1}, SpaceID{1}})
    V = ℂ^2
    return randn(ComplexF64, V, V)
```

### Constructors & Conversion

- **Dictionary Conversion**:
  If you construct a `FragmentedTensor` from a `Dictionary{SpaceID{N}, TensorType}` (where keys are single `SpaceID`s instead of pairs), it automatically converts the keys to `(key, NullSpaceID)` and returns a `FragmentedTensor{N, 0, TensorType}`:
  ```julia
  d_single = Dictionary([sid"A", sid"B"], [t1, t2])
  FT = FragmentedTensor(d_single) # FragmentedTensor{1, 0, TensorType}
  ```

### Lookups & Indexing

- **Pair-Key Lookup**: `FT[(S_out, S_in)]` returns the tensor associated with that pair.
- **Single-Key Lookup**: `FT[S]` where `S` is a single `SpaceID` of size `N_out + N_in`. It checks that `length(S) == N_out + N_in`, extracts `S_out` and `S_in'`, applies `'` to `S_in'` to get `S_in`, and looks up `(S_out, S_in)`.

---

## Supported Operations

### Basic Linear Algebra
- **Addition (`+`)**: `mergewith(+, a.data, b.data)` (sums overlapping keys, copies others)
- **Subtraction (`-`)**: `mergewith(+, a.data, map(-, b.data))`
- **Scalar operations (`*`, `/`)**: Uses native order-preserving broadcasting `a.data .* c` and `a.data ./ c`.
- **Dot product (`dot`)**: If two fragmented tensors have different sets of keys, the product is zero. If their key sets are identical, it is the sum of products per key.
- **Norm (`norm`)**: Evaluates `norm(FT.data)`.

### Matrix Multiplication (`FT1 * FT2`)
Performs block-like multiplication of fragmented tensors by matching second `SpaceID` in key of `FT1` with first `SpaceID` in key of `FT2`:
- Verifies `N_in` of `FT1` matches `N_out` of `FT2` (throws `DimensionMismatch` otherwise).
- All entries with matching keys `(S1, S2)` and `(S2, S3)` are multiplied (`va * vb`) and summed under the key `(S1, S3)`.

### VectorInterface.jl Compatibility
`FragmentedTensor` implements the complete `VectorInterface.jl` protocol:
- `scalartype`, `zerovector`, `zerovector!`, `zerovector!!`
- `scale`, `scale!`, `scale!!`
- `add`, `add!`, `add!!`
- `inner` (delegates to `dot`)

---

## Installation

From the `FragTens/` directory:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Dependencies

- [Dictionaries.jl](https://github.com/andyferris/Dictionaries.jl)
- [VectorInterface.jl](https://github.com/Jutho/VectorInterface.jl)
- [LinearAlgebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/)
- [TensorKit.jl](https://github.com/Jutho/TensorKit.jl) (for tests and space support)
