module Rmap

import ..Epimap
import ..Epimap: AR1, NegativeBinomial2, NegativeBinomial3

import Random
import StatsFuns
import NNlib

using LinearAlgebra
using Turing
using Distributions
using KernelFunctions
using PDMats
using UnPack
using DocStringExtensions
using TensorOperations

include("models.jl")
include("data.jl")

end
