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
    precompute(m::DynamicPPL.Model)

Returns precomputed quantities for model `m` used in its `DynamicPPL.logdensity` implementation.
"""
precompute(::DynamicPPL.Model) = ()

function DynamicPPL.logjoint(model::DynamicPPL.Model)
    precomputed = precompute(model)
    logjoint(θ) = DynamicPPL.logjoint(model, precomputed, θ)
    return logjoint
end

function DynamicPPL.logjoint(model::DynamicPPL.Model, θ::Union{NamedTuple,ComponentArray})
    return DynamicPPL.logjoint(model, SimpleVarInfo(θ))
end

function DynamicPPL._getvalue(nt::ComponentArrays.ComponentArray, sym::Val, inds=())
    # Use `getproperty` instead of `getfield`
    value = getproperty(nt, sym)
    return DynamicPPL._getindex(value, inds)
end

function DynamicPPL.getval(vi::DynamicPPL.SimpleVarInfo{<:ComponentArray}, vn::DynamicPPL.VarName{sym}) where {sym}
    return DynamicPPL._getvalue(vi.θ, Val{sym}(), vn.indexing)
end

# Turing.jl-related
# Makes it so we can use the samples from AHMC as we would a chain obtained from Turing.jl.
struct SimpleTransition{T, L, S}
    θ::T
    logp::L
    stat::S
end

function Turing.Inference.metadata(t::SimpleTransition)
    lp = DynamicPPL.getlogp(t)
    return merge(t.stat, (lp = lp, ))
end

DynamicPPL.getlogp(t::SimpleTransition) = t.logp

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
        Matrix(ComponentArrays.getdata(t.θ)')
    end;
    vals = vcat(vals...)

    # Extract names & construct param array.
    nms = [nms; extra_params]
    parray = hcat(vals, extra_values)

    info = NamedTuple()

    # Conretize the array before giving it to MCMCChains.
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
