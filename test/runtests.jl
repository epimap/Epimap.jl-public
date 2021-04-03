using Epimap
using StableRNGs
using Dates
using Test

@testset "Epimap.jl" begin
    include("conv.jl")
    include("distributions.jl")
    include("dataprocessing.jl")

    if !isempty(get(ENV, "EPIMAP_TEST_ALL", ""))
        include("rmap.jl")
    end
end
