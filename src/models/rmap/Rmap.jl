module Rmap

import ..Epimap
import ..Epimap:
    AR1,
    NegativeBinomial2,
    NegativeBinomial3,
    𝒩₊,
    lowerboundednormlogpdf,
    truncatednormlogpdf,
    halfnormlogpdf,
    nbinomlogpdf3

import Random
import StatsFuns
import NNlib
import Zygote: FillArrays

using LinearAlgebra
using Turing
using DynamicPPL
using Distributions
using KernelFunctions
using PDMats
using UnPack
using DocStringExtensions
using TensorOperations
using Tullio

using TuringUtils, ComponentArrays

import Turing.DynamicPPL: @submodel

include("models.jl")
include("data.jl")

end
