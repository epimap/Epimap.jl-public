# We call `NNlib.pad_constant(u, Δsu, 0)` in our convolution methods to avoid
# the `NNlib.gen_pad` since it introduces type-instabilities. The two lines below are
# necessary to make the ChainRules-pullback for `NNlib.pad_constant` work with the
# direct `Δsu` we are using.
# TODO: Raise this isse over at NNlib and make PR with "fix"
# (a bit unclear if this is the below is the way to go though).
NNlib.gen_pad(pad::NTuple{N,Tuple{Int,Int}}, dims, _) where {N} = pad
NNlib.gen_pad(pad::NTuple{N,Tuple{Int,Int}}, dims::Colon, _) where {N} = pad

# Utility overloads
PDMats.PDMat(P::PDMats.PDMat) = P

# DynamicPPL.jl-related
"""
    evaluatortype(f)
    evaluatortype(f, nargs)
    evaluatortype(f, argtypes)
    evaluatortype(m::DynamicPPL.Model)

Returns the evaluator-type for model `m` or a model-constructor `f`.
"""
function evaluatortype(f, argtypes)
    rets = Core.Compiler.return_types(f, argtypes)
    if (length(rets) != 1) || !(first(rets) <: DynamicPPL.Model)
        error("inferred return-type of $(f) using $(argtypes) is not `Model`; please specify argument types")
    end
    # Extract the anonymous evaluator.
    return first(rets).parameters[1]
end
evaluatortype(f, nargs::Int) = evaluatortype(f, ntuple(_ -> Missing, nargs))
function evaluatortype(f)
    m = first(methods(f))
    # Extract the arguments (first element is the method itself).
    nargs = length(m.sig.parameters) - 1

    return evaluatortype(f, nargs)
end
evaluatortype(::DynamicPPL.Model{F}) where {F} = F

evaluator(m::DynamicPPL.Model) = m.f

"""
    precompute(m::DynamicPPL.Model)

Returns precomputed quantities for model `m` used in its `DynamicPPL.logdensity` implementation.
"""
precompute(::DynamicPPL.Model) = ()

function DynamicPPL.logjoint(model::DynamicPPL.Model)
    precomputed = precompute(model)
    logjoint(θ) = DynamicPPL.logjoint(model, precomputed, θ)
    return logjoint
end

# Turing.jl-related
# Makes it so we can use the samples from AHMC as we would a chain obtained from Turing.jl.
struct SimpleTransition{T, L, S}
    θ::T
    logp::L
    stat::S
end

function Turing.Inference.metadata(t::SimpleTransition)
    lp = t.z.ℓπ.value
    return merge(t.stat, (lp = lp, ))
end

DynamicPPL.getlogp(t::SimpleTransition) = t.z.ℓπ.value

function AbstractMCMC.bundle_samples(
    ts::Vector{<:SimpleTransition},
    var_info::DynamicPPL.VarInfo,
    chain_type::Type{MCMCChains.Chains};
    save_state = false,
    kwargs...
)

    # Convert transitions to array format.
    # Also retrieve the variable names.
    @info "Getting the param names"
    nms, _ = Turing.Inference._params_to_array([var_info]);

    # Get the values of the extra parameters in each transition.
    @info "Transition extras"
    extra_params, extra_values = Turing.Inference.get_transition_extras(ts)

    # We make our own `vals`
    @info "Getting the values"
    vals = map(ts) do t
        Matrix(ComponentArrays.getdata(t.z.θ)')
    end;
    vals = vcat(vals...)

    # Extract names & construct param array.
    nms = [nms; extra_params]
    parray = hcat(vals, extra_values)

    info = NamedTuple()

#     # Conretize the array before giving it to MCMCChains.
#     parray = MCMCChains.concretize(parray)
    @info "Converting to Array{Float64}"
    parray = convert(Array{Float64}, parray)

    # Chain construction.
    @info "Constructing `Chains`"
    return MCMCChains.Chains(
        parray,
        nms,
        (internals = extra_params,);
        info=info,
    )
end

#############
### Rules ###
#############
# Temporary fix for https://github.com/JuliaDiff/ChainRules.jl/issues/402.
ChainRulesCore.@scalar_rule(SpecialFunctions.erfc(x), -(2 * exp(-x * x)) / StatsFuns.sqrtπ)

##############
### Repeat ###
##############
# Adapted from https://gist.github.com/mcabbott/80ac43cca3bee8f57809155a5240519f.
function _repeat(x::AbstractArray, counts::Integer...)
    N = max(ndims(x), length(counts))
    size_y = ntuple(d -> size(x,d) * _get(counts, d, 1), N)
    size_x2 = ntuple(d -> isodd(d) ? size(x, 1+d÷2) : 1, 2*N)

    ## version without mutation
    # ignores = ntuple(d -> reshape(Base.OneTo(counts[d]), ntuple(_->1, 2d-1)..., :), length(counts))
    # y = reshape(broadcast(first∘tuple, reshape(x, size_x2), ignores...), size_y)

    # ## version with mutation
    size_y2 = ntuple(d -> isodd(d) ? size(x, 1+d÷2) : _get(counts, d÷2, 1), 2*N)
    y = similar(x, size_y)
    reshape(y, size_y2) .= reshape(x, size_x2)
    y
end

function _repeat(x::AbstractArray; inner=1, outer=1)
    N = max(ndims(x), length(inner), length(outer))
    size_y = ntuple(d -> size(x, d) * _get(inner, d, 1) * _get(outer, d, 1), N)
    size_y3 = ntuple(3*N) do d3
        dim, class = divrem(d3+2, 3)  # e.g. for x::Matrix, [divrem(n+2,3) for n in 1:3*2] 
        class == 0 && return _get(inner, dim, 1)
        class == 1 && return size(x, dim)
        class == 2 && return _get(outer, dim,1)
    end
    size_x3 = ntuple(3*N) do d3
        dim, class = divrem(d3+2, 3)
        class == 1 ? size(x, dim) : 1
    end
    y = similar(x, size_y)
    reshape(y, size_y3) .= reshape(x, size_x3)
    y
end

_get(t::Tuple, i::Int, default) = i in 1:length(t) ? t[i] : default
_get(x::Number, i::Int, default) = i==1 ? x : default

function ChainRulesCore.rrule(::typeof(_repeat), x::AbstractArray, counts::Integer...)
    size_x = size(x)
    function repeat_pullback_1(dy)
        size2ndims = ntuple(d -> isodd(d) ? size_x[1+d÷2] : _get(counts, d÷2, 1), 2*ndims(dy))
        reduced = sum(reshape(dy, size2ndims); dims = ntuple(d -> 2d, ndims(dy)))
        return (ChainRulesCore.NoTangent(), reshape(reduced, size_x), map(_->ChainRulesCore.NoTangent(), counts)...)
    end
    return repeat(x, counts...), repeat_pullback_1
end

function ChainRulesCore.rrule(::typeof(_repeat), x::AbstractArray; inner=1, outer=1)
    size_x = size(x)
    function repeat_pullback_2(dy)
        size3ndims = ntuple(3*ndims(dy)) do d3
            dim, class = divrem(d3+2, 3)  # e.g. for x::Matrix, [divrem(n+2,3) for n in 1:3*2] 
            class == 0 && return _get(inner, dim, 1)
            class == 1 && return size(x, dim)
            class == 2 && return _get(outer, dim,1)
        end
        reduced = sum(reshape(dy, size3ndims); dims = ntuple(d -> isodd(d) ? 3(d÷2)+1 : 3(d÷2), 2*ndims(dy)))
        return (ChainRulesCore.NoTangent(), reshape(reduced, size_x))
    end
    return repeat(x; inner=inner, outer=outer), repeat_pullback_2
end

# using KernelAbstractions
# import Zygote
# import Base: _RepeatInnerOuter

# @kernel function repeat_inner_kernel!(a::AbstractArray{<:Any, N}, inner::NTuple{N}, out) where {N}
#     inds = @index(Global, NTuple)
#     inds_a = ntuple(i -> (inds[i] - 1) ÷ inner[i] + 1, N)

#     @inbounds out[inds...] = a[inds_a...]
# end

# function repeat_inner(xs::AbstractArray, inner)
#     out = similar(xs, eltype(xs), inner .* size(xs))
#     any(==(0), size(out)) && return out # consistent with `Base.repeat`

#     # This is threadsafe, so we run with as many threads as possible.
#     kernel! = repeat_inner_kernel!(CPU(), Threads.nthreads())

#     ev = kernel!(xs, inner, out, ndrange=size(out))
#     wait(ev)
#     return out
# end
# # Non-cached coherent loads
# @kernel function repeat_outer_kernel!(a::AbstractArray{<:Any, N}, sa::NTuple{N}, outer::NTuple{N}, out) where {N}
#     inds = @index(Global, NTuple)
#     inds_a = ntuple(i -> (inds[i] - 1) % sa[i] + 1, N)

#     @inbounds out[inds...] = a[inds_a...]
# end

# function repeat_outer(xs::AbstractArray, outer)
#     out = similar(xs, eltype(xs), outer .* size(xs))
#     any(==(0), size(out)) && return out # consistent with `Base.repeat`

#     # This is threadsafe, so we run with as many threads as possible.
#     kernel! = repeat_outer_kernel!(CPU(), Threads.nthreads())

#     ev = kernel!(xs, size(xs), outer, out, ndrange=size(out))
#     wait(ev)
#     return out
# end


# # Overload methods used by `Base.repeat`.
# # No need to implement `repeat_inner_outer` since this is implemented in `Base` as
# # `repeat_outer(repeat_inner(arr, inner), outer)`.
# function _RepeatInnerOuter.repeat_inner(a::AbstractArray{<:Any, N}, dims::NTuple{N}) where {N}
#     return repeat_inner(a, dims)
# end

# function _RepeatInnerOuter.repeat_outer(a::AbstractArray{<:Any, N}, dims::NTuple{N}) where {N}
#     return repeat_outer(a, dims)
# end

# function _RepeatInnerOuter.repeat_outer(a::AbstractVector, dims::Tuple{Any})
#     return repeat_outer(a, dims)
# end

# function _RepeatInnerOuter.repeat_outer(a::AbstractMatrix, dims::NTuple{2, Any})
#     return repeat_outer(a, dims)
# end

# ### Adjoint implementation for `repeat`
# # FIXME: not threadsafe atm!!! And therefore not used anywhere (we only overload) the
# # adjoint for `CuArray`. But if we have  something like `@atomic_addindex!`
# # from https://github.com/JuliaLang/julia/pull/37683, we'd be golden.
# @kernel function repeat_adjoint_cpu_kernel!(
#     Δ::AbstractArray,
#     inner::NTuple,
#     outer::NTuple,
#     out::AbstractArray,
#     outsize::NTuple{N}
# ) where {N}
#     dest_inds = @index(Global, NTuple)
#     src_inds = ntuple(i -> mod1((dest_inds[i] - 1) ÷ inner[i] + 1, outsize[i]), N)
#     # FIXME: make threadsafe
#     out[src_inds...] += Δ[dest_inds...]
# end;

# function repeat_adjoint(
#     x,
#     Δ::AbstractArray{<:Any, N},
#     inner::NTuple{N},
#     outer::NTuple{N}
# ) where {N}
#     out = zero(x)

#     # TODO: Make kernel threadsafe. Until then we run using only 1 thread.
#     kernel! = repeat_adjoint_cpu_kernel!(CPU(), 1)

#     ev = kernel!(Δ, inner, outer, out, size(out), ndrange=size(Δ))
#     wait(ev)

#     return out
# end;


# Zygote.@adjoint function _RepeatInnerOuter.repeat_inner_outer(xs::AbstractArray, inner::Nothing, outer::Nothing)
#     return xs, Δ -> (Δ, )
# end
# Zygote.@adjoint function _RepeatInnerOuter.repeat_inner_outer(
#     xs::AbstractArray,
#     inner::Nothing,
#     outer::NTuple{N}
# ) where {N}
#     inner_new = ntuple(_ -> 1, N)
#     return (
#         _RepeatInnerOuter.repeat_outer(xs, outer),
#         Δ -> (repeat_adjoint(xs, Δ, inner_new, outer), )
#     )
# end
# Zygote.@adjoint function _RepeatInnerOuter.repeat_inner_outer(
#     xs::AbstractArray,
#     inner::NTuple{N},
#     outer::Nothing
# ) where {N}
#     outer_new = ntuple(_ -> 1, N)
#     return (
#         _RepeatInnerOuter.repeat_inner(xs, inner),
#         Δ -> (repeat_adjoint(xs, Δ, inner, outer_new), )
#     )
# end
# Zygote.@adjoint function _RepeatInnerOuter.repeat_inner_outer(
#     xs::AbstractArray,
#     inner::NTuple{N},
#     outer::NTuple{N}
# ) where {N}
#     return (
#         _RepeatInnerOuter.repeat_outer(_RepeatInnerOuter.repeat_inner(xs, inner), outer),
#         Δ -> (repeat_adjoint(xs, Δ, inner, outer), )
#     )
# end

# # We need to stop Zygote from using the rule implemented in
# # https://github.com/FluxML/Zygote.jl/blob/d5be4d5ca80e79278d714eaac15ca71904a262e3/src/lib/array.jl#L149-L163
# # We stop this adjoint from triggering, and then call the underlying `repeat_inner_outer`.
# # Unfortunately we need to replicate the body of `_RepeatInnerOuter.repeat` since we can't do `Zygote.pullback`
# # for kwargs.
# Zygote.@adjoint function Base.repeat(arr::AbstractArray; inner = nothing, outer = nothing)
#     _RepeatInnerOuter.check(arr, inner, outer)
#     arr, inner, outer = _RepeatInnerOuter.resolve(arr, inner, outer)
#     return Zygote.pullback(_RepeatInnerOuter.repeat_inner_outer, arr, inner, outer)
# end
