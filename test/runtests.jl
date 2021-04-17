using Epimap
using Epimap.Turing

using StableRNGs
using Dates
using ComponentArrays
using Adapt
using Test

@testset "Epimap.jl" begin
    include("test_data.jl")

    include("conv.jl")
    include("distributions.jl")
    include("dataprocessing.jl")

    include("rmap.jl")
end
