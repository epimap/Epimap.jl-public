module Rmap

import ..Epimap
import ..Epimap: AR1, NegativeBinomial2, NegativeBinomial3

import Random
import StatsFuns

using LinearAlgebra
using Turing
using Distributions
using KernelFunctions
using PDMats
using UnPack
using DocStringExtensions

include("models.jl")
include("data.jl")

end
