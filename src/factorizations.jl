using Dictionaries
using LinearAlgebra
using TensorKit
using MatrixAlgebraKit
import TensorKit: eigh_full, eigh_trunc, eig_full, eig_trunc, svd_full, svd_compact, svd_trunc
using ChainRulesCore

export eigh_full, eigh_trunc, eig_full, eig_trunc, svd_full, svd_compact, svd_trunc

# -----------------------------------------------------------------------------
# Modular Helper Routines
# -----------------------------------------------------------------------------

"""
    assemble_global(A::FragmentedTensor, OUT, IN)

Assembles block-sparse fragmented tensor `A` into a single global `TensorMap`.
Returns `(H_global, metadata)` where `metadata` contains sector offsets and global spaces.
"""
function assemble_global(A::FragmentedTensor{N_out, N_in, TensorType}, OUT, IN) where {N_out, N_in, TensorType}
    if !isempty(A.data)
        first_val = first(values(A.data))
        S_type = spacetype(first_val)
        I_type = sectortype(first_val)
        T_scalar = scalartype(first_val)
    else
        T_concrete = (TensorType isa Union) ? TensorType.a : TensorType
        S_type = try
            spacetype(T_concrete)
        catch
            ComplexSpace
        end
        I_type = try
            sectortype(T_concrete)
        catch
            Trivial
        end
        T_scalar = try
            scalartype(T_concrete)
        catch
            ComplexF64
        end
    end
    # NB: the ProductSpace rank is intentionally left free (not pinned to
    # N_out / N_in). A fragmented SpaceID{N} may carry labels that don't
    # materialise as tensor legs (e.g. trivial "r" legs), so different keys
    # sharing the same N can map to ProductSpaces of *different* rank. Pinning
    # the rank to N_out / N_in throws a convert error on such tensors.
    CodomType = ProductSpace{S_type}
    DomType = ProductSpace{S_type}

    # 0. Extract spaces from the blocks
    space_dict_out = Dict{SpaceID{N_out}, CodomType}()
    space_dict_in  = Dict{SpaceID{N_in}, DomType}()
    for (k, val) in pairs(A.data)
        S_out, S_in = k
        if !haskey(space_dict_out, S_out)
            space_dict_out[S_out] = codomain(val)
        end
        if !haskey(space_dict_in, S_in)
            space_dict_in[S_in] = domain(val)
        end
    end

    # 1. Collect all coupled sectors (charges) across all spaces in OUT and IN.
    #
    # Self-adjoint fallback: in square decompositions (eigh / eig) OUT and IN are
    # both passed `SPACES = union(OUT, IN)`, so a SpaceID S may appear in OUT
    # without being present in `space_dict_out` (and vice-versa). Fragments at the
    # missing side are zero; we still need a space of the right dimension to lay
    # out blocks. Per the codebase convention (TensorMap(W ← W) for diagonal
    # fragments — see test `build_random_tensor`), codomain and domain at the
    # same SpaceID share the same W, so the fallback uses the same W (not its
    # dual). Otherwise downstream `A * U` would hit a SpaceMismatch between
    # `domain(A[_, S])` and `codomain(U[S, _])`.
    sectors_set = Set{I_type}()
    for S in OUT
        W = haskey(space_dict_out, S) ? space_dict_out[S] : space_dict_in[S]
        for c in blocksectors(W)
            push!(sectors_set, c)
        end
    end
    for S in IN
        W = haskey(space_dict_in, S) ? space_dict_in[S] : space_dict_out[S]
        for c in blocksectors(W)
            push!(sectors_set, c)
        end
    end
    active_sectors = collect(sectors_set)

    # 2. Build offsets for each sector
    offsets_out = Dict{SpaceID{N_out}, Dict{I_type, UnitRange{Int}}}()
    offsets_in  = Dict{SpaceID{N_in}, Dict{I_type, UnitRange{Int}}}()

    D_out = Dict{I_type, Int}()
    D_in  = Dict{I_type, Int}()

    for c in active_sectors
        D_out[c] = 0
        D_in[c] = 0
    end

    for S in OUT
        W = haskey(space_dict_out, S) ? space_dict_out[S] : space_dict_in[S]
        offsets_out[S] = Dict{I_type, UnitRange{Int}}()
        for c in active_sectors
            d = c ∈ blocksectors(W) ? blockdim(W, c) : 0
            offsets_out[S][c] = (D_out[c] + 1):(D_out[c] + d)
            D_out[c] += d
        end
    end

    for S in IN
        W = haskey(space_dict_in, S) ? space_dict_in[S] : space_dict_out[S]
        offsets_in[S] = Dict{I_type, UnitRange{Int}}()
        for c in active_sectors
            d = c ∈ blocksectors(W) ? blockdim(W, c) : 0
            offsets_in[S][c] = (D_in[c] + 1):(D_in[c] + d)
            D_in[c] += d
        end
    end

    # 3. Create global spaces (checking if we are in GradedSpace or Cartesian/ComplexSpace)
    first_space = if !isempty(OUT)
        haskey(space_dict_out, first(OUT)) ? space_dict_out[first(OUT)] : space_dict_in[first(OUT)]
    else
        haskey(space_dict_in, first(IN)) ? space_dict_in[first(IN)] : space_dict_out[first(IN)]
    end
    if I_type == Trivial
        tot_d_out = sum(values(D_out))
        tot_d_in  = sum(values(D_in))
        V_global_out = S_type(tot_d_out)
        V_global_in  = S_type(tot_d_in)
    else
        V_global_out = S_type(D_out...)
        V_global_in  = S_type(D_in...)
    end

    # 4. Allocate H_global
    H_global = zeros(T_scalar, V_global_out ← V_global_in)

    # 5. Populate H_global
    for (k, val) in pairs(A.data)
        S_out, S_in = k
        if !(S_out in OUT) || !(S_in in IN)
            continue
        end
        for c in active_sectors
            W_out = space_dict_out[S_out]
            W_in  = space_dict_in[S_in]
            if (c ∈ blocksectors(W_out)) && (c ∈ blocksectors(W_in))
                block_mat = block(val, c)
                range_out = offsets_out[S_out][c]
                range_in  = offsets_in[S_in][c]
                block(H_global, c)[range_out, range_in] .= block_mat
            end
        end
    end

    metadata = (offsets_out, offsets_in, V_global_out, V_global_in, active_sectors, space_dict_out, space_dict_in)
    return H_global, metadata
end

"""
    disassemble_global(H_global, OUT, IN, metadata)

Reconstructs the fragmented dictionary of block matrices from a global `H_global`.
"""
function disassemble_global(H_global, OUT::AbstractVector{SpaceID{N_out}}, IN::AbstractVector{SpaceID{N_in}}, metadata) where {N_out, N_in}
    offsets_out, offsets_in, V_global_out, V_global_in, active_sectors, space_dict_out, space_dict_in = metadata
    T_res = eltype(H_global)
    S_type = spacetype(typeof(V_global_out))

    # Leave the materialised-leg ranks free: a fragmented SpaceID{N} may carry
    # non-materialising labels, so the actual TensorMap rank can be < N_out / N_in.
    T_val = TensorMap{T_res, S_type, N₁, N₂, Vector{T_res}} where {N₁, N₂}
    res_data = Dictionary{Tuple{SpaceID{N_out}, SpaceID{N_in}}, T_val}()

    for S_out in OUT
        for S_in in IN
            W_out = haskey(space_dict_out, S_out) ? space_dict_out[S_out] : space_dict_in[S_out]
            W_in  = haskey(space_dict_in, S_in)  ? space_dict_in[S_in]   : space_dict_out[S_in]
            val = zeros(T_res, W_out ← W_in)

            has_nonzero = false
            for c in active_sectors
                if (c ∈ blocksectors(W_out)) && (c ∈ blocksectors(W_in))
                    range_out = offsets_out[S_out][c]
                    range_in  = offsets_in[S_in][c]
                    block_slice = block(H_global, c)[range_out, range_in]
                    if !isempty(block_slice)
                        block(val, c) .= block_slice
                        has_nonzero = true
                    end
                end
            end

            if has_nonzero
                insert!(res_data, (S_out, S_in), val)
            end
        end
    end
    return res_data
end

"""
    disassemble_global_U(U_global, OUT, V_virt, offsets_out, eigen_space_name, space_dict)

Slices U_global row-wise to re-construct FragmentedTensor U.
"""
function disassemble_global_U(U_global, OUT, V_virt, offsets_out, eigen_space_name, space_dict)
    T_res = eltype(U_global)
    N_out = length(first(OUT).labels)
    S_eigen = SpaceID{1}((string(eigen_space_name),), (false,))

    S_type = spacetype(typeof(V_virt))
    # Codomain rank is free (fragmented SpaceIDs may have non-materialising legs);
    # domain is always the 1-leg eigen/SV space.
    T_val = TensorMap{T_res, S_type, N₁, 1, Vector{T_res}} where {N₁}

    U_data = Dictionary{Tuple{SpaceID{N_out}, SpaceID{1}}, T_val}()

    for S_k in OUT
        W_k = space_dict[S_k]
        U_k = zeros(T_res, W_k ← V_virt)

        has_nonzero = false
        for c in blocksectors(V_virt)
            if c ∈ blocksectors(W_k)
                range_k = offsets_out[S_k][c]
                block_slice = block(U_global, c)[range_k, :]
                if !isempty(block_slice)
                    block(U_k, c) .= block_slice
                    has_nonzero = true
                end
            end
        end
        if has_nonzero
            insert!(U_data, (S_k, S_eigen), U_k)
        end
    end

    return FragmentedTensor{N_out, 1, T_val}(U_data)
end

"""
    assemble_global_U(dU_fragmented, U_global, OUT, V_virt, offsets_out, space_dict)

Assembles tangent dU_fragmented back into a global dU_global.
"""
function assemble_global_U(dU_fragmented, U_global, OUT, V_virt, offsets_out, space_dict)
    T_res = eltype(U_global)
    dU_global = zeros(T_res, codomain(U_global) ← domain(U_global))

    for (k, val) in pairs(dU_fragmented.data)
        S_k, S_eigen = k
        if !(S_k in OUT)
            continue
        end
        for c in blocksectors(V_virt)
            if c ∈ blocksectors(space_dict[S_k])
                range_k = offsets_out[S_k][c]
                block(dU_global, c)[range_k, :] .= block(val, c)
            end
        end
    end
    return dU_global
end

"""
    disassemble_global_V(V_global, IN, V_virt, offsets_in, eigen_space_name, space_dict)

Slices V_global column-wise to re-construct FragmentedTensor V.
"""
function disassemble_global_V(V_global, IN, V_virt, offsets_in, eigen_space_name, space_dict)
    T_res = eltype(V_global)
    N_in = length(first(IN).labels)
    S_eigen = SpaceID{1}((string(eigen_space_name),), (false,))

    S_type = spacetype(typeof(V_virt))
    # Domain rank is free (fragmented SpaceIDs may have non-materialising legs);
    # codomain is always the 1-leg eigen/SV space.
    T_val = TensorMap{T_res, S_type, 1, N₂, Vector{T_res}} where {N₂}

    V_data = Dictionary{Tuple{SpaceID{1}, SpaceID{N_in}}, T_val}()

    for S_j in IN
        W_j = space_dict[S_j]
        V_k = zeros(T_res, V_virt ← W_j)

        has_nonzero = false
        for c in blocksectors(V_virt)
            if c ∈ blocksectors(W_j)
                range_in = offsets_in[S_j][c]
                block_slice = block(V_global, c)[:, range_in]
                if !isempty(block_slice)
                    block(V_k, c) .= block_slice
                    has_nonzero = true
                end
            end
        end
        if has_nonzero
            insert!(V_data, (S_eigen, S_j), V_k)
        end
    end

    return FragmentedTensor{1, N_in, T_val}(V_data)
end

"""
    assemble_global_V(dV_fragmented, V_global, IN, V_virt, offsets_in, space_dict)

Assembles tangent dV_fragmented back into a global dV_global.
"""
function assemble_global_V(dV_fragmented, V_global, IN, V_virt, offsets_in, space_dict)
    T_res = eltype(V_global)
    dV_global = zeros(T_res, codomain(V_global) ← domain(V_global))

    for (k, val) in pairs(dV_fragmented.data)
        S_eigen, S_j = k
        if !(S_j in IN)
            continue
        end
        for c in blocksectors(V_virt)
            if c ∈ blocksectors(space_dict[S_j])
                range_in = offsets_in[S_j][c]
                block(dV_global, c)[:, range_in] .= block(val, c)
            end
        end
    end
    return dV_global
end

# -----------------------------------------------------------------------------
# Matrix inverse on eigenvector FragmentedTensor
# -----------------------------------------------------------------------------

"""
    assemble_U_frag(U_frag::FragmentedTensor{N_out, 1, T})

Assemble an eig-shape FragmentedTensor U (keys `(S_k, S_eigen)`) into a global
square TensorMap `U_global : V_virt → ⊕_k W_k`, plus metadata sufficient to
disassemble its inverse back into a V-shape FragmentedTensor.
"""
function assemble_U_frag(U_frag::FragmentedTensor{N_out, 1, T}) where {N_out, T}
    isempty(U_frag.data) && error("assemble_U_frag: cannot assemble an empty FragmentedTensor")
    first_val = first(values(U_frag.data))
    S_type = spacetype(first_val)
    I_type = sectortype(first_val)
    T_scalar = scalartype(first_val)

    # All entries must share the same S_eigen on the second key slot.
    S_eigen = first(keys(U_frag.data))[2]
    for k in keys(U_frag.data)
        k[2] == S_eigen || error("assemble_U_frag: inconsistent S_eigen across keys")
    end
    V_virt = domain(first_val)[1]

    # Collect codomain spaces per S_k from the data. Rank left free: a fragmented
    # SpaceID{N_out} may carry non-materialising legs, so codomain rank can be < N_out.
    space_dict = Dict{SpaceID{N_out}, ProductSpace{S_type}}()
    for (k, val) in pairs(U_frag.data)
        S_k, _ = k
        if !haskey(space_dict, S_k)
            space_dict[S_k] = codomain(val)
        end
    end
    OUT = collect(keys(space_dict))

    # Active sectors = union of V_virt's and all W_k's blocksectors.
    sectors_set = Set{I_type}()
    for c in blocksectors(V_virt)
        push!(sectors_set, c)
    end
    for S in OUT
        for c in blocksectors(space_dict[S])
            push!(sectors_set, c)
        end
    end
    active_sectors = collect(sectors_set)

    # Per-sector codomain-row offsets, one block of rows per S_k.
    offsets_out = Dict{SpaceID{N_out}, Dict{I_type, UnitRange{Int}}}()
    D_out = Dict{I_type, Int}()
    for c in active_sectors
        D_out[c] = 0
    end
    for S in OUT
        W = space_dict[S]
        offsets_out[S] = Dict{I_type, UnitRange{Int}}()
        for c in active_sectors
            d = c ∈ blocksectors(W) ? blockdim(W, c) : 0
            offsets_out[S][c] = (D_out[c] + 1):(D_out[c] + d)
            D_out[c] += d
        end
    end

    # Global codomain space.
    if I_type == Trivial
        V_global_out = S_type(sum(values(D_out)))
    else
        V_global_out = S_type(D_out...)
    end

    # For inv to make sense we need a square TensorMap, i.e. V_global_out and
    # V_virt must have matching total dimensions per sector.
    U_global = zeros(T_scalar, V_global_out ← V_virt)
    for (k, val) in pairs(U_frag.data)
        S_k, _ = k
        W_k = space_dict[S_k]
        for c in active_sectors
            if c ∈ blocksectors(W_k) && c ∈ blocksectors(V_virt)
                range_out = offsets_out[S_k][c]
                block(U_global, c)[range_out, :] .= block(val, c)
            end
        end
    end

    metadata = (offsets_out, V_virt, V_global_out, OUT, S_eigen, space_dict, active_sectors)
    return U_global, metadata
end

"""
    Base.inv(U_frag::FragmentedTensor{N_out, 1, T})

Matrix inverse of an eig-shape `FragmentedTensor` (keys `(S_k, S_eigen)`).
Returns the V-shape inverse with keys `(S_eigen, S_k)`. Requires `U_frag` to
be square in the global sense — typically satisfied when it comes from
`eig_full(A::FragmentedTensor)` of a square `A`.
"""
function Base.inv(U_frag::FragmentedTensor{N_out, 1, T}) where {N_out, T}
    U_global, metadata = assemble_U_frag(U_frag)
    offsets_out, V_virt, V_global_out, OUT, S_eigen, space_dict, _ = metadata
    Uinv_global = inv(U_global)
    eigen_name = S_eigen.labels[1]
    return disassemble_global_V(Uinv_global, OUT, V_virt, offsets_out, eigen_name, space_dict)
end

function ChainRulesCore.rrule(::typeof(Base.inv), U_frag::FragmentedTensor{N_out, 1, T}) where {N_out, T}
    U_global, metadata = assemble_U_frag(U_frag)
    offsets_out, V_virt, V_global_out, OUT, S_eigen, space_dict, active_sectors = metadata
    Uinv_global = inv(U_global)
    eigen_name = S_eigen.labels[1]
    Uinv_frag = disassemble_global_V(Uinv_global, OUT, V_virt, offsets_out, eigen_name, space_dict)

    function inv_frag_pullback(dUinv_frag)
        d_un = unthunk(dUinv_frag)
        if d_un isa Union{ZeroTangent, NoTangent, Nothing}
            return (NoTangent(), ZeroTangent())
        end
        # Promote tangent-typed input into a proper FragmentedTensor for assembly.
        d_frag = d_un isa FragmentedTensor ? d_un : begin
            ddata = _extract_ft_cotangent_data(d_un)
            ddata === nothing ?
                FragmentedTensor{1, N_out, T}() :
                FragmentedTensor{1, N_out, eltype(ddata)}(ddata)
        end
        dUinv_global = assemble_global_V(d_frag, Uinv_global, OUT, V_virt, offsets_out, space_dict)
        # Matrix-inverse adjoint identity: dX = -Y' * dY * Y'   (Y = inv(X))
        dU_global = -(Uinv_global' * dUinv_global * Uinv_global')
        dU_frag = disassemble_global_U(dU_global, OUT, V_virt, offsets_out, eigen_name, space_dict)
        return (NoTangent(), dU_frag)
    end

    return Uinv_frag, inv_frag_pullback
end

# Helper used by both the FT-constructor pullback above and the inv pullback.
function _extract_ft_cotangent_data(d_un)
    d_un isa Union{ZeroTangent, NoTangent, Nothing} && return nothing
    if d_un isa FragmentedTensor
        return d_un.data
    end
    if d_un isa Tangent
        ddata = try d_un.data catch; nothing; end
        return ddata isa Dictionary ? ddata : nothing
    end
    return nothing
end

# -----------------------------------------------------------------------------
# Constructor adjoint
# -----------------------------------------------------------------------------
#
# Zygote can't differentiate through `FragmentedTensor{N,M,T}(data)` natively
# because the gradient flowing back is itself a `FragmentedTensor` (produced by
# the `dot` / arithmetic pullbacks), not the `NamedTuple{(:data,)}` that
# Zygote's `Jnew` expects. Give it an explicit rule.

function ChainRulesCore.rrule(::Type{FragmentedTensor{N_out, N_in, T}}, data::Dictionary) where {N_out, N_in, T}
    res = FragmentedTensor{N_out, N_in, T}(data)
    function ft_constructor_pullback(dFT)
        d_un = unthunk(dFT)
        if d_un isa Union{ZeroTangent, NoTangent, Nothing}
            return (NoTangent(), NoTangent())
        elseif d_un isa FragmentedTensor
            return (NoTangent(), d_un.data)
        elseif d_un isa Tangent
            backing = d_un.backing
            if backing isa NamedTuple && haskey(backing, :data)
                return (NoTangent(), backing.data)
            end
        end
        try
            return (NoTangent(), d_un.data)
        catch
            return (NoTangent(), NoTangent())
        end
    end
    return res, ft_constructor_pullback
end

# Look up the cotangent of the i-th inserted value inside a FragmentedTensor
# cotangent. Zygote may present the cotangent as:
#   * a real `FragmentedTensor` — pull the entry at `key` from its data;
#   * a `Tangent{<:FragmentedTensor, ...}` with `.data` resolving to a `Tangent`
#     of a Dictionary, whose `.values` field is the Vector aligned with
#     insertion order; or
#   * a real `Tangent`-wrapped Dictionary, with `.values` similarly aligned.
# `Tangent.getproperty` papers over the "logical all-zero" backing case, so we
# access fields via getproperty rather than touching `.backing` directly.
function _lookup_ft_value_cotangent(d_un, key, key_index)
    d_un isa Union{ZeroTangent, NoTangent, Nothing} && return ZeroTangent()
    if d_un isa FragmentedTensor
        return haskey(d_un.data, key) ? d_un.data[key] : ZeroTangent()
    end
    if d_un isa Tangent
        ddata = try d_un.data catch; nothing; end
        ddata === nothing && return ZeroTangent()
        return _lookup_dict_value_cotangent(ddata, key, key_index)
    end
    return ZeroTangent()
end

function _lookup_dict_value_cotangent(ddata, key, key_index)
    ddata isa Union{ZeroTangent, NoTangent, Nothing} && return ZeroTangent()
    if ddata isa Dictionary
        return haskey(ddata, key) ? ddata[key] : ZeroTangent()
    end
    if ddata isa Tangent
        vals = try ddata.values catch; nothing; end
        vals isa AbstractVector || return ZeroTangent()
        return key_index <= length(vals) ? vals[key_index] : ZeroTangent()
    end
    return ZeroTangent()
end

# rrule for the user-facing pair-based constructor `FragmentedTensor((S_out,S_in) => t, ...)`.
# The cotangent of `FragmentedTensor(pair_1, ..., pair_n)` w.r.t. `pair_i` lives
# at key `pair_i.first` of the result FragmentedTensor's data — the i-th slot
# under the Dictionary's insertion order, which matches the argument order.
function ChainRulesCore.rrule(::Type{FragmentedTensor}, pair1::Pair{<:Tuple{SpaceID{N_out}, SpaceID{N_in}}}, more::Pair{<:Tuple{SpaceID{N_out}, SpaceID{N_in}}}...) where {N_out, N_in}
    res = FragmentedTensor(pair1, more...)
    all_pairs = (pair1, more...)
    function ft_pairs_pullback(dFT)
        d_un = unthunk(dFT)
        d_pairs = map(enumerate(all_pairs)) do (i, p)
            dv = _lookup_ft_value_cotangent(d_un, p.first, i)
            Tangent{typeof(p)}(first = NoTangent(), second = dv)
        end
        return (NoTangent(), d_pairs...)
    end
    return res, ft_pairs_pullback
end

"""
    wrap_singleton_frag(tensor, S_eigen)

Wraps a single `TensorMap` (or `DiagonalTensorMap`) into a `FragmentedTensor{1,1}`
keyed by `(S_eigen, S_eigen)`. Used inside every decomposition wrapper to box
the global `D` or `S` factor. Its hand-written rrule pulls the inner tensor
back out, sidestepping Zygote's lack of an adjoint for the `Dictionary(...)`
constructor along the chain.
"""
function wrap_singleton_frag(tensor, S_eigen::SpaceID{1})
    data = Dictionary([(S_eigen, S_eigen)], [tensor])
    return FragmentedTensor{1, 1, typeof(tensor)}(data)
end

function ChainRulesCore.rrule(::typeof(wrap_singleton_frag), tensor, S_eigen::SpaceID{1})
    res = wrap_singleton_frag(tensor, S_eigen)
    function wrap_singleton_pullback(dFT)
        d_un = unthunk(dFT)
        if d_un isa Union{ZeroTangent, NoTangent, Nothing}
            return (NoTangent(), zero(tensor), NoTangent())
        end
        ddata = if d_un isa FragmentedTensor
            d_un.data
        elseif d_un isa Tangent && d_un.backing isa NamedTuple && haskey(d_un.backing, :data)
            d_un.backing.data
        else
            return (NoTangent(), zero(tensor), NoTangent())
        end
        # ddata is a Dictionary (or a Tangent wrapping one)
        key = (S_eigen, S_eigen)
        if ddata isa Dictionary && haskey(ddata, key)
            return (NoTangent(), ddata[key], NoTangent())
        elseif ddata isa Tangent
            inner = ddata.backing
            if inner isa NamedTuple
                indices = haskey(inner, :indices) ? inner.indices : nothing
                vals = haskey(inner, :values) ? inner.values : nothing
                if vals !== nothing
                    return (NoTangent(), first(vals), NoTangent())
                end
            end
        end
        return (NoTangent(), zero(tensor), NoTangent())
    end
    return res, wrap_singleton_pullback
end

# Treat metadata-returning TensorKit accessors as non-differentiable. They return
# space / dimension info, not numerical data; ChainRules / TensorKit may already
# cover some, but covering them here avoids any silent Jnew fallback.
ChainRulesCore.@non_differentiable space(::Any)
ChainRulesCore.@non_differentiable space(::Any, ::Any)
ChainRulesCore.@non_differentiable codomain(::Any)
ChainRulesCore.@non_differentiable domain(::Any)
ChainRulesCore.@non_differentiable blocksectors(::Any)
ChainRulesCore.@non_differentiable spacetype(::Any)
ChainRulesCore.@non_differentiable sectortype(::Any)

# -----------------------------------------------------------------------------
# FragmentedTensor basic arithmetic pullbacks for Zygote compatibility
# -----------------------------------------------------------------------------

function ChainRulesCore.rrule(::typeof(Base.:+), a::FragmentedTensor{N_out, N_in, TA}, b::FragmentedTensor{N_out, N_in, TB}) where {N_out, N_in, TA, TB}
    res = a + b
    function add_pullback(dres)
        d_unthunked = unthunk(dres)
        return (NoTangent(), d_unthunked, d_unthunked)
    end
    return res, add_pullback
end

function ChainRulesCore.rrule(::typeof(Base.:-), a::FragmentedTensor{N_out, N_in, TA}, b::FragmentedTensor{N_out, N_in, TB}) where {N_out, N_in, TA, TB}
    res = a - b
    function sub_pullback(dres)
        d_unthunked = unthunk(dres)
        return (NoTangent(), d_unthunked, -d_unthunked)
    end
    return res, sub_pullback
end

function ChainRulesCore.rrule(::typeof(Base.:*), c::Number, a::FragmentedTensor{N_out, N_in, TensorType}) where {N_out, N_in, TensorType}
    res = c * a
    function mult_pullback(dres)
        dres_un = unthunk(dres)
        dc = dot(a, dres_un)
        da = conj(c) * dres_un
        return (NoTangent(), dc, da)
    end
    return res, mult_pullback
end

function ChainRulesCore.rrule(::typeof(Base.:*), a::FragmentedTensor{N_out, N_in, TensorType}, c::Number) where {N_out, N_in, TensorType}
    res = a * c
    function mult_pullback_rev(dres)
        dres_un = unthunk(dres)
        da = dres_un * conj(c)
        dc = dot(a, dres_un)
        return (NoTangent(), da, dc)
    end
    return res, mult_pullback_rev
end

function ChainRulesCore.rrule(::typeof(Base.adjoint), a::FragmentedTensor)
    res = adjoint(a)
    function ft_adjoint_pullback(dres)
        d_un = unthunk(dres)
        return (NoTangent(), d_un isa Union{ZeroTangent, NoTangent, Nothing} ? ZeroTangent() : adjoint(d_un))
    end
    return res, ft_adjoint_pullback
end

function ChainRulesCore.rrule(::typeof(Base.:*), a::FragmentedTensor{N_out1, N_mid, TA}, b::FragmentedTensor{N_mid, N_in2, TB}) where {N_out1, N_mid, N_in2, TA, TB}
    res = a * b
    function ft_mul_pullback(dres)
        d_un = unthunk(dres)
        # For matrix product C = A * B with cotangent dC,
        # dA = dC * B'  and  dB = A' * dC.
        dA = d_un * adjoint(b)
        dB = adjoint(a) * d_un
        return (NoTangent(), dA, dB)
    end
    return res, ft_mul_pullback
end

function ChainRulesCore.rrule(::typeof(LinearAlgebra.dot), a::FragmentedTensor{N_out, N_in, TA}, b::FragmentedTensor{N_out, N_in, TB}) where {N_out, N_in, TA, TB}
    val = dot(a, b)
    function dot_pullback(dval)
        if isempty(a.data) || isempty(b.data)
            da = FragmentedTensor{N_out, N_in, TA}()
            db = FragmentedTensor{N_out, N_in, TB}()
            return (NoTangent(), da, db)
        end
        # ChainRules convention: for c = dot(a,b) = sum conj(a)*b,
        #   da = conj(dval) * b  ;  db = dval * a
        da_data = map(x -> conj(dval) * x, b.data)
        db_data = map(x -> dval * x, a.data)
        # Use the mapped dict's own (rank-agnostic) value type, not the concrete
        # rank of its first fragment: fragmented tensors store varying-rank
        # tensors under one key type, so `typeof(first(...))` would force a
        # convert that throws on any fragment of a different rank.
        da = FragmentedTensor{N_out, N_in, eltype(da_data)}(da_data)
        db = FragmentedTensor{N_out, N_in, eltype(db_data)}(db_data)
        return (NoTangent(), da, db)
    end
    return val, dot_pullback
end

# -----------------------------------------------------------------------------
# ChainRulesCore.jl Custom Pullback Definitions
# -----------------------------------------------------------------------------
#
# These three (assemble_global, disassemble_global_U, disassemble_global_V) are
# the ONLY decomposition-related rrules we define. Everything else — eigh_full,
# eig_full, svd_full, etc. — is an ordinary Julia function that Zygote traces
# through. The hand-off to MatrixAlgebraKit's AD happens at the inner
# `eigh_full(H_global)` call which operates on a TensorMap.

function ChainRulesCore.rrule(::typeof(assemble_global), A::FragmentedTensor{N_out, N_in, TensorType}, OUT, IN) where {N_out, N_in, TensorType}
    res = assemble_global(A, OUT, IN)

    function assemble_global_pullback(d_res)
        d_un = unthunk(d_res)
        # d_un is the cotangent of the (H_global, metadata) tuple. metadata is
        # not differentiable, so we accept either a plain Tuple or a Tangent.
        dH_global = if d_un isa Tuple
            unthunk(d_un[1])
        elseif d_un isa Tangent
            unthunk(d_un.backing[1])
        else
            d_un
        end
        dA_data = disassemble_global(dH_global, OUT, IN, res[2])
        dA = FragmentedTensor{N_out, N_in, eltype(dA_data)}(dA_data)
        return (NoTangent(), dA, NoTangent(), NoTangent())
    end

    return res, assemble_global_pullback
end

function ChainRulesCore.rrule(::typeof(disassemble_global_U), U_global, OUT, V_virt, offsets_out, eigen_space_name, space_dict)
    U_fragmented = disassemble_global_U(U_global, OUT, V_virt, offsets_out, eigen_space_name, space_dict)

    function disassemble_global_U_pullback(dU_fragmented)
        d_un = unthunk(dU_fragmented)
        if d_un isa Union{ZeroTangent, NoTangent, Nothing}
            return (NoTangent(), zero(U_global), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
        end
        dU_global = assemble_global_U(d_un, U_global, OUT, V_virt, offsets_out, space_dict)
        return (NoTangent(), dU_global, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    return U_fragmented, disassemble_global_U_pullback
end

function ChainRulesCore.rrule(::typeof(disassemble_global_V), V_global, IN, V_virt, offsets_in, eigen_space_name, space_dict)
    V_fragmented = disassemble_global_V(V_global, IN, V_virt, offsets_in, eigen_space_name, space_dict)

    function disassemble_global_V_pullback(dV_fragmented)
        d_un = unthunk(dV_fragmented)
        if d_un isa Union{ZeroTangent, NoTangent, Nothing}
            return (NoTangent(), zero(V_global), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
        end
        dV_global = assemble_global_V(d_un, V_global, IN, V_virt, offsets_in, space_dict)
        return (NoTangent(), dV_global, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    return V_fragmented, disassemble_global_V_pullback
end

# -----------------------------------------------------------------------------
# Structural helpers (non-differentiable)
# -----------------------------------------------------------------------------

"""
    extract_out_in_spaces(A::FragmentedTensor)

Returns `(OUT, IN, SPACES)` — the unique out- and in-side SpaceIDs of `A`, plus
their union. Marked non-differentiable so Zygote doesn't try to trace through
`Set` / `union(...)` / `push!`, which are all `Δ-mutating` from its perspective
and unrelated to numerical gradients (these depend only on the keyset, not the
tensor values).
"""
function extract_out_in_spaces(A::FragmentedTensor)
    OUT = collect(Set(first(k) for k in keys(A.data)))
    IN  = collect(Set(last(k) for k in keys(A.data)))
    SPACES = collect(union(OUT, IN))
    return OUT, IN, SPACES
end
ChainRulesCore.@non_differentiable extract_out_in_spaces(::Any)

"""
    extract_out_in(A::FragmentedTensor)

Returns `(OUT, IN)` — unique out- and in-side SpaceIDs of `A`. Used by SVD
where OUT and IN are kept separate. Non-differentiable for the same reason as
`extract_out_in_spaces`.
"""
function extract_out_in(A::FragmentedTensor)
    OUT = collect(Set(first(k) for k in keys(A.data)))
    IN  = collect(Set(last(k) for k in keys(A.data)))
    return OUT, IN
end
ChainRulesCore.@non_differentiable extract_out_in(::Any)

"""
    make_S_eigen(name)

Builds the synthetic 1-leg `SpaceID` for the eigen/SV space. Non-differentiable.
"""
make_S_eigen(name) = SpaceID{1}((string(name),), (false,))
ChainRulesCore.@non_differentiable make_S_eigen(::Any)

# -----------------------------------------------------------------------------
# Combined codomain dict for square decompositions
# -----------------------------------------------------------------------------

"""
    combined_codom_dict(SPACES, space_dict_out, space_dict_in)

For square decompositions (eigh / eig) where `SPACES = union(OUT, IN)` may
contain entries that only appear in `space_dict_in`. Returns a dictionary
mapping every `S ∈ SPACES` to its codomain-form `ProductSpace` — falling back
to `space_dict_in[S]` (same W; see note in `assemble_global`) when `S` is
missing from `space_dict_out`.
"""
function combined_codom_dict(SPACES, space_dict_out::Dict{SpaceID{N}, PS}, space_dict_in::Dict{SpaceID{N}, PS}) where {N, PS}
    result = Dict{SpaceID{N}, PS}()
    for S in SPACES
        result[S] = haskey(space_dict_out, S) ? space_dict_out[S] : space_dict_in[S]
    end
    return result
end
ChainRulesCore.@non_differentiable combined_codom_dict(::Any, ::Any, ::Any)

# -----------------------------------------------------------------------------
# eigh (Hermitian Eigenvalue Decomposition)
# -----------------------------------------------------------------------------

function eigh_full(A::FragmentedTensor{N, N, TensorType}; eigen_space_name::Union{String,Symbol}=string("eigen_", hash(A))) where {N, TensorType}
    _, _, SPACES = extract_out_in_spaces(A)
    H_global, metadata = assemble_global(A, SPACES, SPACES)
    D_global, U_global = eigh_full(H_global)

    V_virt = space(D_global, 1)
    S_eigen = make_S_eigen(eigen_space_name)
    D_frag = wrap_singleton_frag(D_global, S_eigen)

    space_dict_U = combined_codom_dict(SPACES, metadata[6], metadata[7])
    U_frag = disassemble_global_U(U_global, SPACES, V_virt, metadata[1], eigen_space_name, space_dict_U)
    return D_frag, U_frag
end

function eigh_trunc(A::FragmentedTensor{N, N, TensorType}, args...; eigen_space_name::Union{String,Symbol}=string("eigen_", hash(A)), kwargs...) where {N, TensorType}
    _, _, SPACES = extract_out_in_spaces(A)
    H_global, metadata = assemble_global(A, SPACES, SPACES)
    if length(args) > 0 && args[1] isa MatrixAlgebraKit.TruncationStrategy
        D_global, U_global, _ = eigh_trunc(H_global; trunc=args[1], kwargs...)
    else
        D_global, U_global, _ = eigh_trunc(H_global, args...; kwargs...)
    end

    V_virt = space(D_global, 1)
    S_eigen = make_S_eigen(eigen_space_name)
    D_frag = wrap_singleton_frag(D_global, S_eigen)

    space_dict_U = combined_codom_dict(SPACES, metadata[6], metadata[7])
    U_frag = disassemble_global_U(U_global, SPACES, V_virt, metadata[1], eigen_space_name, space_dict_U)
    return D_frag, U_frag
end

# -----------------------------------------------------------------------------
# eig (General Eigenvalue Decomposition)
# -----------------------------------------------------------------------------

function eig_full(A::FragmentedTensor{N, N, TensorType}; eigen_space_name::Union{String,Symbol}=string("eigen_", hash(A))) where {N, TensorType}
    _, _, SPACES = extract_out_in_spaces(A)
    H_global, metadata = assemble_global(A, SPACES, SPACES)
    D_global, U_global = eig_full(H_global)

    V_virt = space(D_global, 1)
    S_eigen = make_S_eigen(eigen_space_name)
    D_frag = wrap_singleton_frag(D_global, S_eigen)

    space_dict_U = combined_codom_dict(SPACES, metadata[6], metadata[7])
    U_frag = disassemble_global_U(U_global, SPACES, V_virt, metadata[1], eigen_space_name, space_dict_U)
    return D_frag, U_frag
end

function eig_trunc(A::FragmentedTensor{N, N, TensorType}, args...; eigen_space_name::Union{String,Symbol}=string("eigen_", hash(A)), kwargs...) where {N, TensorType}
    _, _, SPACES = extract_out_in_spaces(A)
    H_global, metadata = assemble_global(A, SPACES, SPACES)
    if length(args) > 0 && args[1] isa MatrixAlgebraKit.TruncationStrategy
        D_global, U_global, _ = eig_trunc(H_global; trunc=args[1], kwargs...)
    else
        D_global, U_global, _ = eig_trunc(H_global, args...; kwargs...)
    end

    V_virt = space(D_global, 1)
    S_eigen = make_S_eigen(eigen_space_name)
    D_frag = wrap_singleton_frag(D_global, S_eigen)

    space_dict_U = combined_codom_dict(SPACES, metadata[6], metadata[7])
    U_frag = disassemble_global_U(U_global, SPACES, V_virt, metadata[1], eigen_space_name, space_dict_U)
    return D_frag, U_frag
end

# -----------------------------------------------------------------------------
# svd (Singular Value Decomposition)
# -----------------------------------------------------------------------------

function svd_full(A::FragmentedTensor{N_out, N_in, TensorType}; eigen_space_name::Union{String,Symbol}=string("svd_", hash(A))) where {N_out, N_in, TensorType}
    OUT, IN = extract_out_in(A)
    H_global, metadata = assemble_global(A, OUT, IN)
    U_global, S_global, V_global = svd_full(H_global)

    V_virt_U = codomain(S_global)
    V_virt_V = domain(S_global)
    S_eigen = make_S_eigen(eigen_space_name)
    S_frag = wrap_singleton_frag(S_global, S_eigen)

    U_frag = disassemble_global_U(U_global, OUT, V_virt_U, metadata[1], eigen_space_name, metadata[6])
    V_frag = disassemble_global_V(V_global, IN, V_virt_V, metadata[2], eigen_space_name, metadata[7])
    return U_frag, S_frag, V_frag
end

function svd_compact(A::FragmentedTensor{N_out, N_in, TensorType}; eigen_space_name::Union{String,Symbol}=string("svd_", hash(A))) where {N_out, N_in, TensorType}
    OUT, IN = extract_out_in(A)
    H_global, metadata = assemble_global(A, OUT, IN)
    U_global, S_global, V_global = svd_compact(H_global)

    V_virt_U = codomain(S_global)
    V_virt_V = domain(S_global)
    S_eigen = make_S_eigen(eigen_space_name)
    S_frag = wrap_singleton_frag(S_global, S_eigen)

    U_frag = disassemble_global_U(U_global, OUT, V_virt_U, metadata[1], eigen_space_name, metadata[6])
    V_frag = disassemble_global_V(V_global, IN, V_virt_V, metadata[2], eigen_space_name, metadata[7])
    return U_frag, S_frag, V_frag
end

function svd_trunc(A::FragmentedTensor{N_out, N_in, TensorType}, args...; eigen_space_name::Union{String,Symbol}=string("svd_", hash(A)), kwargs...) where {N_out, N_in, TensorType}
    OUT, IN = extract_out_in(A)
    H_global, metadata = assemble_global(A, OUT, IN)
    if length(args) > 0 && args[1] isa MatrixAlgebraKit.TruncationStrategy
        U_global, S_global, V_global, _ = svd_trunc(H_global; trunc=args[1], kwargs...)
    else
        U_global, S_global, V_global, _ = svd_trunc(H_global, args...; kwargs...)
    end

    V_virt_U = codomain(S_global)
    V_virt_V = domain(S_global)
    S_eigen = make_S_eigen(eigen_space_name)
    S_frag = wrap_singleton_frag(S_global, S_eigen)

    U_frag = disassemble_global_U(U_global, OUT, V_virt_U, metadata[1], eigen_space_name, metadata[6])
    V_frag = disassemble_global_V(V_global, IN, V_virt_V, metadata[2], eigen_space_name, metadata[7])
    return U_frag, S_frag, V_frag
end
