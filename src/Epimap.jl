module Epimap

import Random
import StatsFuns, SpecialFunctions
import NNlib

using Turing, Distributions
using DocStringExtensions

export Rmap,
    NegativeBinomial2,
    NegativeBinomial3,
    AR1

"""
    make_logdensity(model_def, args...)

Makes a method which computes the log density for the given model.

The method should at least be able to handle the arguments passed as a flattened vector.

## Arguments
- `model_def`: a model *definition*, not an instance of a model.
- `args`: args usually passed to the model definition to create the model instance.
"""
make_logjoint(model_def, args...) = Turing.Variational.make_logjoint(model_def(args...))


include("adapt.jl")
include("utils.jl")
include("conv.jl")
include("distributions.jl")

include("models/rmap/Rmap.jl")

end
