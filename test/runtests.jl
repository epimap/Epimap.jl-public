using Epimap
using StableRNGs
using Test

@testset "Epimap.jl" begin
    include("conv.jl")
    include("distributions.jl")

    if !isempty(get(ENV, "EPIMAP_TEST_ALL", ""))
        include("rmap.jl")
    end
end
