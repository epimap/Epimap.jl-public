module Epimap

import Random

using Turing, Distributions

import StatsFuns

"""
    make_logdensity(model_def, args...)

Makes a method which computes the log density for the given model.

The method should at least be able to handle the arguments passed as a flattened vector.

## Arguments
- `model_def`: a model *definition*, not an instance of a model.
- `args`: args usually passed to the model definition to create the model instance.
"""
make_logdensity(model_def, args...) = Turing.Variational.make_logjoint(model_def(args...))

include("utils.jl")
include("distributions.jl")

end
