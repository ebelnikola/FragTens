module FragmentedTensors

using Dictionaries
using LinearAlgebra
using VectorInterface
using TensorOperations
using TensorKit

export SpaceID, FragmentedTensor, @sid_str, string_to_space_id, NullSpaceID

# ---------------------------------------------------------
# SpaceID
# ---------------------------------------------------------

struct SpaceID{N}
    labels::NTuple{N,String}
    adjoint::NTuple{N,Bool}
end

Base.:(==)(a::SpaceID, b::SpaceID) = false
Base.:(==)(a::SpaceID{N}, b::SpaceID{N}) where N = a.labels == b.labels && a.adjoint == b.adjoint
Base.hash(s::SpaceID, h::UInt) = hash(s.adjoint, hash(s.labels, h))

function string_to_space_id(s::AbstractString)
    parts = split(s, ',')
    labels = String[]
    adjoint = Bool[]
    for p in parts
        p = strip(p)
        if isempty(p)
            continue
        end
        if endswith(p, "'")
            push!(labels, p[1:end-1])
            push!(adjoint, true)
        else
            push!(labels, p)
            push!(adjoint, false)
        end
    end
    N = length(labels)
    return SpaceID{N}(NTuple{N,String}(labels), NTuple{N,Bool}(adjoint))
end

macro sid_str(s)
    return string_to_space_id(s)
end

function Base.string(s::SpaceID)
    return join([lbl * (adj ? "'" : "") for (lbl, adj) in zip(s.labels, s.adjoint)], ", ")
end
Base.show(io::IO, s::SpaceID) = print(io, "sid\"", string(s), "\"")

# Adjoint and multiplication
Base.adjoint(S::SpaceID{N}) where N = SpaceID{N}(S.labels, map(!, S.adjoint))
Base.:*(S1::SpaceID{N1}, S2::SpaceID{N2}) where {N1,N2} = SpaceID{N1+N2}((S1.labels..., S2.labels...), (S1.adjoint..., S2.adjoint...))

const NullSpaceID = SpaceID{0}((), ())

# ---------------------------------------------------------
# FragmentedTensor
# ---------------------------------------------------------

struct FragmentedTensor{N_out, N_in, TensorType}
    data::Dictionary{Tuple{SpaceID{N_out},SpaceID{N_in}},TensorType}
end

# Default empty constructor
FragmentedTensor{N_out, N_in, TensorType}() where {N_out, N_in, TensorType} =
    FragmentedTensor{N_out, N_in, TensorType}(Dictionary{Tuple{SpaceID{N_out},SpaceID{N_in}},TensorType}())


function FragmentedTensor(dict::Dictionary{SpaceID{N}, TensorType}) where {N, TensorType}
    new_keys = (x -> (x, NullSpaceID)).(keys(dict))
    return FragmentedTensor{N, 0, TensorType}(Dictionary(new_keys, dict))
end

# ---------------------------------------------------------
# Dictionary Interface
# ---------------------------------------------------------

Base.getindex(ft::FragmentedTensor, key::Tuple{SpaceID, SpaceID}) = ft.data[key]
Base.setindex!(ft::FragmentedTensor, val, key::Tuple{SpaceID, SpaceID}) = set!(ft.data, key, val)
Base.haskey(ft::FragmentedTensor, key::Tuple{SpaceID, SpaceID}) = haskey(ft.data, key)
Base.keys(ft::FragmentedTensor) = keys(ft.data)
Base.values(ft::FragmentedTensor) = values(ft.data)
Base.pairs(ft::FragmentedTensor) = pairs(ft.data)

function Base.getindex(ft::FragmentedTensor{N_out, N_in}, S::SpaceID{N}) where {N_out, N_in, N}
    if N != N_out + N_in
        throw(DimensionMismatch("SpaceID length ($N) does not match total expected length ($(N_out + N_in))"))
    end
    S_out = SpaceID{N_out}(S.labels[1:N_out], S.adjoint[1:N_out])
    S_in_prime = SpaceID{N_in}(S.labels[N_out+1:end], S.adjoint[N_out+1:end])
    S_in = S_in_prime'
    return ft[(S_out, S_in)]
end

function Base.haskey(ft::FragmentedTensor{N_out, N_in}, S::SpaceID{N}) where {N_out, N_in, N}
    if N != N_out + N_in
        return false
    end
    S_out = SpaceID{N_out}(S.labels[1:N_out], S.adjoint[1:N_out])
    S_in_prime = SpaceID{N_in}(S.labels[N_out+1:end], S.adjoint[N_out+1:end])
    S_in = S_in_prime'
    return haskey(ft.data, (S_out, S_in))
end

function Base.setindex!(ft::FragmentedTensor{N_out, N_in}, val, S::SpaceID{N}) where {N_out, N_in, N}
    if N != N_out + N_in
        throw(DimensionMismatch("SpaceID length ($N) does not match total expected length ($(N_out + N_in))"))
    end
    S_out = SpaceID{N_out}(S.labels[1:N_out], S.adjoint[1:N_out])
    S_in_prime = SpaceID{N_in}(S.labels[N_out+1:end], S.adjoint[N_out+1:end])
    S_in = S_in_prime'
    ft[(S_out, S_in)] = val
end

# ---------------------------------------------------------
# Random Generation
# ---------------------------------------------------------

function Base.randn(::Type{FragmentedTensor{N_out, N_in, TensorType}}, keys::AbstractVector{<:Tuple{SpaceID{N_out}, SpaceID{N_in}}}, tensor_constructor::Function) where {N_out, N_in, TensorType}
    dict = Dictionary{Tuple{SpaceID{N_out}, SpaceID{N_in}}, TensorType}()
    for k in keys
        t = tensor_constructor(k)
        insert!(dict, k, t)
    end
    return FragmentedTensor{N_out, N_in, TensorType}(dict)
end

function Base.randn(::Type{FragmentedTensor{N_out, 0, TensorType}}, keys::AbstractVector{SpaceID{N_out}}, tensor_constructor::Function) where {N_out, TensorType}
    dict = Dictionary{Tuple{SpaceID{N_out}, SpaceID{0}}, TensorType}()
    for k in keys
        t = tensor_constructor(k)
        insert!(dict, (k, NullSpaceID), t)
    end
    return FragmentedTensor{N_out, 0, TensorType}(dict)
end

# ---------------------------------------------------------
# Base / LinearAlgebra Math
# ---------------------------------------------------------

Base.:+(a::FragmentedTensor{N_out, N_in, TA}, b::FragmentedTensor{N_out, N_in, TB}) where {N_out, N_in, TA, TB} =
    FragmentedTensor(mergewith(+, a.data, b.data))

Base.:-(a::FragmentedTensor{N_out, N_in, TA}, b::FragmentedTensor{N_out, N_in, TB}) where {N_out, N_in, TA, TB} =
    FragmentedTensor(mergewith(+, a.data, map(-, b.data)))

Base.:*(a::FragmentedTensor{N_out, N_in, TensorType}, c::Number) where {N_out, N_in, TensorType} =
    FragmentedTensor(a.data .* c)

Base.:*(c::Number, a::FragmentedTensor{N_out, N_in, TensorType}) where {N_out, N_in, TensorType} =
    FragmentedTensor(c .* a.data)

Base.:/(a::FragmentedTensor{N_out, N_in, TensorType}, c::Number) where {N_out, N_in, TensorType} =
    FragmentedTensor(a.data ./ c)

function LinearAlgebra.dot(a::FragmentedTensor{N_out, N_in, TA}, b::FragmentedTensor{N_out, N_in, TB}) where {N_out, N_in, TA, TB}
    if !issetequal(keys(a.data), keys(b.data))
        return zero(promote_type(scalartype(TA), scalartype(TB)))
    end
    if isempty(a.data)
        return zero(promote_type(scalartype(TA), scalartype(TB)))
    end
    res = zero(promote_type(scalartype(TA), scalartype(TB)))
    for k in keys(a.data)
        res += dot(a.data[k], b.data[k])
    end
    return res
end

LinearAlgebra.norm(ft::FragmentedTensor) = norm(ft.data)

function Base.:*(a::FragmentedTensor{N_out1, N_in1, T1}, b::FragmentedTensor{N_out2, N_in2, T2}) where {N_out1, N_in1, T1, N_out2, N_in2, T2}
    if N_in1 != N_out2
        throw(DimensionMismatch("N_in ($N_in1) of first tensor does not match N_out ($N_out2) of second tensor"))
    end
    if isempty(a.data) || isempty(b.data)
        T_res = promote_type(T1, T2)
        return FragmentedTensor{N_out1, N_in2, T_res}()
    end

    T_res = Base.promote_op(*, T1, T2)
    res_data = Dictionary{Tuple{SpaceID{N_out1}, SpaceID{N_in2}}, T_res}()

    for (ka, va) in pairs(a.data)
        S1, S2 = ka
        for (kb, vb) in pairs(b.data)
            S2_prime, S3 = kb
            if S2 == S2_prime
                k_res = (S1, S3)
                prod_val = va * vb
                if haskey(res_data, k_res)
                    res_data[k_res] = res_data[k_res] + prod_val
                else
                    insert!(res_data, k_res, prod_val)
                end
            end
        end
    end
    return FragmentedTensor{N_out1, N_in2, T_res}(res_data)
end

# ---------------------------------------------------------
# VectorInterface.jl Overloads
# ---------------------------------------------------------

VectorInterface.scalartype(::Type{FragmentedTensor{N_out, N_in, TensorType}}) where {N_out, N_in, TensorType} = VectorInterface.scalartype(TensorType)
VectorInterface.scalartype(a::FragmentedTensor) = VectorInterface.scalartype(typeof(a))

function VectorInterface.zerovector(ft::FragmentedTensor{N_out, N_in, TensorType}, ::Type{S}) where {N_out, N_in, TensorType, S<:Number}
    if !isempty(ft.data)
        zero_type = typeof(VectorInterface.zerovector(first(ft.data), S))
        return FragmentedTensor{N_out, N_in, zero_type}()
    else
        return FragmentedTensor{N_out, N_in, TensorType}()
    end
end
VectorInterface.zerovector!(ft::FragmentedTensor) = (empty!(ft.data); ft)
VectorInterface.zerovector!!(ft::FragmentedTensor) = VectorInterface.zerovector!(ft)

VectorInterface.scale(ft::FragmentedTensor, α::Number) = FragmentedTensor(ft.data .* α)
function VectorInterface.scale!(ft::FragmentedTensor, α::Number)
    for k in keys(ft.data)
        VectorInterface.scale!(ft.data[k], α)
    end
    return ft
end
function VectorInterface.scale!!(ft::FragmentedTensor, α::Number)
    for k in keys(ft.data)
        ft.data[k] = VectorInterface.scale!!(ft.data[k], α)
    end
    return ft
end

VectorInterface.scale(y::FragmentedTensor, x::FragmentedTensor, α::Number) = VectorInterface.scale(x, α)
function VectorInterface.scale!(y::FragmentedTensor, x::FragmentedTensor, α::Number)
    empty!(y.data)
    for (k, vx) in pairs(x.data)
        insert!(y.data, k, VectorInterface.scale(vx, α))
    end
    return y
end
function VectorInterface.scale!!(y::FragmentedTensor, x::FragmentedTensor, α::Number)
    empty!(y.data)
    for (k, vx) in pairs(x.data)
        insert!(y.data, k, VectorInterface.scale(vx, α))
    end
    return y
end


function VectorInterface.add(y::FragmentedTensor{N_out, N_in, TY}, x::FragmentedTensor{N_out, N_in, TX}, α::Number=1, β::Number=1) where {N_out, N_in, TY, TX}
    T_res = promote_type(TY, TX)
    res = Dictionary{Tuple{SpaceID{N_out}, SpaceID{N_in}}, T_res}()
    for k in union(keys(y.data), keys(x.data))
        if haskey(y.data, k) && haskey(x.data, k)
            insert!(res, k, VectorInterface.add(y.data[k], x.data[k], α, β))
        elseif haskey(y.data, k)
            insert!(res, k, VectorInterface.scale(y.data[k], β))
        else
            insert!(res, k, VectorInterface.scale(x.data[k], α))
        end
    end
    return FragmentedTensor(res)
end

function VectorInterface.add!(y::FragmentedTensor, x::FragmentedTensor, α::Number=1, β::Number=1)
    for k in keys(y.data)
        if haskey(x.data, k)
            VectorInterface.add!(y.data[k], x.data[k], α, β)
        else
            VectorInterface.scale!(y.data[k], β)
        end
    end
    for k in keys(x.data)
        if !haskey(y.data, k)
            insert!(y.data, k, VectorInterface.scale(x.data[k], α))
        end
    end
    return y
end

function VectorInterface.add!!(y::FragmentedTensor, x::FragmentedTensor, α::Number=1, β::Number=1)
    for k in keys(y.data)
        if haskey(x.data, k)
            y.data[k] = VectorInterface.add!!(y.data[k], x.data[k], α, β)
        else
            y.data[k] = VectorInterface.scale!!(y.data[k], β)
        end
    end
    for k in keys(x.data)
        if !haskey(y.data, k)
            insert!(y.data, k, VectorInterface.scale(x.data[k], α))
        end
    end
    return y
end

VectorInterface.inner(a::FragmentedTensor, b::FragmentedTensor) = dot(a, b)

# ---------------------------------------------------------
# TensorOperations.jl Interface (Commented out for now)
# ---------------------------------------------------------

# const IndexTuple = Tuple{Vararg{Int}}

# function TensorOperations.tensoradd!(
#     C::FragmentedTensor,
#     A::FragmentedTensor, pA::Tuple{IndexTuple,IndexTuple}, conjA::Bool,
#     α::Number, β::Number
# )
#     ...
# end

# function TensorOperations.tensortrace!(
#     ...
# )
#     ...
# end

# function TensorOperations.tensorcontract!(
#     ...
# )
#     ...
# end

# function TensorOperations.tensorcontract_type(...)
#     ...
# end

# function TensorOperations.tensoralloc_contract(...)
#     ...
# end

end # module
