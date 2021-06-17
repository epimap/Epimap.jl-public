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

#############
### Rules ###
#############
# Temporary fix for https://github.com/JuliaDiff/ChainRules.jl/issues/402.
ChainRulesCore.@scalar_rule(SpecialFunctions.erfc(x), -(2 * exp(-x * x)) / StatsFuns.sqrtπ)
