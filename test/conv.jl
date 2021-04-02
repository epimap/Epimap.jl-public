@testset "conv.jl" begin
    # Vector × Vector
    x = randn(10); w = randn(2);
    @test Epimap.conv_dsp(x, w) ≈ Epimap.conv_fft(x, w)
    @test Epimap.conv_fft(x, w) ≈ Epimap.conv_nnlib(x, w)

    # Matrix × Vector
    x = randn(10, 2); w = randn(1, 2);
    @test Epimap.conv_dsp(x, w) ≈ Epimap.conv_fft(x, w)
    @test Epimap.conv_fft(x, w) ≈ Epimap.conv_nnlib(x, w)

    # Matrix × Matrix
    x = randn(10, 2); w = randn(2, 2);
    @test Epimap.conv_dsp(x, w) ≈ Epimap.conv_fft(x, w)
    @test Epimap.conv_fft(x, w) ≈ Epimap.conv_nnlib(x, w)
end
