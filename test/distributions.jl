using Distributions, StatsBase
import Epimap: NegativeBinomial2, NegativeBinomial3, GammaMeanCv, AR1

@testset "NegativeBinomial2" begin
    μ = 1.
    ϕ = 2.
    dist = NegativeBinomial2(μ, ϕ)

    @test mean(dist) ≈ μ
    @test var(dist) ≈ μ + μ^2 / ϕ
end

@testset "GammaMeanCv" begin
    μ = 1.
    cv = √2
    dist = GammaMeanCv(μ, cv)

    @test mean(dist) ≈ μ
    @test var(dist) ≈ cv^2
end

@testset "AR1" begin
    # Zero auto-correlation, zero mean, unit variance
    α = 0.0; μ = 0.0; σ = 1.0;
    ar = AR1(1000, α, μ, σ)
    @test mean([mean(rand(ar)) for i = 1:100]) ≈ μ atol=1e-2

    # Zero auto-correlation, non-zero mean, unit variance
    α = 0.0; μ = 1.0; σ = 1.0;
    ar = AR1(1000, α, μ, σ)
    @test mean([mean(rand(ar)) for i = 1:100]) ≈ μ atol=1e-2

    # Zero auto-correlation, non-zero mean, non-unit variance
    α = 0.0; μ = 1.0; σ = 5.0;
    ar = AR1(1000, α, μ, σ)
    @test mean([std(rand(ar)) for i = 1:100]) ≈ σ atol=1e-1

    # Non-zero auto-correlation, zero mean, unit variance
    α = 0.5; μ = 0.0; σ = 1.0;
    ar = AR1(1000, α, μ, σ)
    @test mean([first(autocov(rand(ar), [1])) for i = 1:100]) ≈ α atol=1e-1

    # TODO: test `logpdf` somehow
end

@testset "truncatednormlogpdf" begin
    rng = StableRNG(42);

    μs = [0.0, 1.0]
    σs = [0.5, 10.0]
    lbs = [-Inf, -5.0, 0.0, 5.0]
    ubs = [-5.0, 0.0, 5.0, Inf]

    for μ in μs, σ in σs, lb in lbs, ub in ubs
        if lb ≥ ub
            continue
        end
        target = truncated(Normal(μ, σ), lb, ub)
        x = rand(target)

        @test logpdf(target, x) ≈ Rmap.truncatednormlogpdf(μ, σ, x, lb, ub)
    end
end
