module Epimap

import Random
import ChainRulesCore, SpecialFunctions, StatsFuns
import NNlib, PDMats

using ComponentArrays
using Bijectors.Functors
using TuringUtils: TuringUtils

using Tullio

using Reexport
@reexport using Turing, Distributions
using DocStringExtensions

export Rmap,
    NegativeBinomial2,
    NegativeBinomial3,
    AR1,
    SimpleTransition

"""
    make_logdensity(model_def, args...)

Makes a method which computes the log density for the given model.

The method should at least be able to handle the arguments passed as a flattened vector.

## Arguments
- `model_def`: a model *definition*, not an instance of a model.
- `args`: args usually passed to the model definition to create the model instance.
"""
function make_logjoint(
    model::DynamicPPL.Model, ::Type{T}=Float64;
    bijector_options=TuringUtils.BijectorStructureOptions(unvectorize_univariates=true)
) where {T}
    # Sample using `SimpleVarInfo` once to get the parameter space.
    svi = SimpleVarInfo(model)

    # Adapt parameters to use desired `eltype`.
    adaptor = FloatMaybeAdaptor{T}()
    θ = adapt(adaptor, ComponentArray(svi.θ))

    # Construct an example `VarInfo`.
    vi = Turing.VarInfo(model)

    # Construct the corresponding bijector.
    b_orig = TuringUtils.optimize_bijector_structure(
        Bijectors.bijector(vi, bijector_options);
        options=bijector_options
    )
    # Adapt bijector parameters to use desired `eltype`.
    b = fmap(b_orig) do x
        adapt(adaptor, x)
    end
    binv = inv(b)

    # Converter used for standard arrays.
    axis = first(ComponentArrays.getaxes(θ))
    nt(x) = Epimap.tonamedtuple(x, axis)

    function logjoint_unconstrained(args_unconstrained::AbstractVector)
        return logjoint_unconstrained(nt(args_unconstrained))
    end
    function logjoint_unconstrained(args_unconstrained::Union{NamedTuple, ComponentArray})
        args, logjac = forward(binv, args_unconstrained)
        return logjoint(args) + logjac
    end

    logjoint(args::AbstractVector) = logjoint(nt(args))
    function logjoint(args::Union{NamedTuple, ComponentArray})
        return DynamicPPL.logjoint(model, args)
    end

    return (logjoint, logjoint_unconstrained, b, θ)
end

include("temporary_hacks.jl")
include("adapt.jl")
include("utils.jl")
include("conv.jl")
include("distributions.jl")

include("models/rmap/Rmap.jl")

end
