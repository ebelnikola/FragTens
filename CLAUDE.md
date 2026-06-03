# FragmentedTensors.jl — package guide

`FragmentedTensors.jl` is a standalone Julia package providing block-sparse,
heterogeneous tensor structures keyed by labeled boundary spaces. It is
consumed by the web GUI's generated Julia code
(`web/src/transforms/deckCodeGenerationCore.js`,
`diagramCodeGeneration.js`) and is also used directly for matrix
factorizations and AD-aware decompositions.

This file is the high-level orientation. Topic-specific detail lives
under `docs/` and is loaded only when you actually need it. For the
fast-operation reference and hand-off-style summaries see
[`ANTIGRAVITY.md`](ANTIGRAVITY.md).

## Layout

- `src/FragmentedTensors.jl` — module definition, `SpaceID{N}`,
  `FragmentedTensor{N_out, N_in, TensorType}`, arithmetic, indexing,
  `VectorInterface.jl` overloads, `permute` overload.
- `src/factorizations.jl` — `eigh_full` / `eigh_trunc` / `eig_full` /
  `eig_trunc` / `svd_full` / `svd_compact` / `svd_trunc` for
  `FragmentedTensor`, plus the assemble / disassemble helpers,
  `Base.inv` on the eig-shape result, and `ChainRulesCore` pullbacks
  for Zygote AD.
- `test/runtests.jl` — assembled test suite (SpaceID arithmetic,
  VectorInterface, TensorKit interop, permutation, factorizations,
  AD).

## Deeper-dive docs

- [`docs/data-model.md`](docs/data-model.md) — `SpaceID{N}` and
  `FragmentedTensor{N_out, N_in, TensorType}` structure, plus the
  **tensor-leg ↔ key-leg mapping rule**: how the legs of the stored
  tensor correspond to positions in the combined key SpaceID
  `S_out * S_in'` when some labels are non-material (material labels
  in order, codomain first then domain). Includes worked example.
- [`docs/implementation-rules.md`](docs/implementation-rules.md) —
  the six load-bearing invariants: `Dictionaries.jl` `set!` instead
  of `[]=`; `VectorInterface.jl` in-place vs out-of-place;
  type-stable multiplication via `Base.promote_op`; `dot` summed
  over the keyset *intersection* (direct-sum philosophy); composite
  `FT[S]` lookup; the two-level `permute` overload.
- [`docs/factorizations.md`](docs/factorizations.md) — wrapper /
  assemble / disassemble design (Zygote-friendly), self-adjoint
  space fallback, **rank-agnostic container types** (the
  `where {N₁,N₂}` rule that keeps non-full-rank fragments from
  throwing `convert` errors), non-differentiable structural
  helpers, the matrix-inverse rrule, and the pair-based constructor.
- [`docs/testing.md`](docs/testing.md) — fuzz design (`N_REPS=40`
  per scenario), gauge-invariant losses, excuse policies,
  varying-rank coverage (Group F + Multi-Space §8). Test-plan
  rationale: `~/.claude/plans/factorizations-test-plan.md`.
- [`docs/consumer-interface.md`](docs/consumer-interface.md) — what
  the web GUI's generated Julia code expects: `eval_<deck>`,
  `apply_<deck>`, `apply_adjoint_<deck>`, and `NullSpaceID` semantics.

When working on a topic, open the matching doc; don't reconstruct the
rules from the code each time.

## Running tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```
