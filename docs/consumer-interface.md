# Consumer interface (web GUI)

Generated code from the web GUI (`web/src/transforms/deckCodeGenerationCore.js`,
`diagramCodeGeneration.js`) imports `FragmentedTensors` and produces three
kinds of entry points per deck.

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

Operand-name formation rules on the *web* side (fragmented `SpaceID`
block extraction, adjoint-by-swap, flips, indices) live in the
top-level `web/docs/code-generation.md` — that's the source of truth
for how a deck becomes Julia source.
