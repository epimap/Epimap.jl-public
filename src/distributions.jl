"""
    NegativeBinomial2(μ, ϕ)

Mean-variance parameterization of `NegativeBinomial`.

## Derivation
`NegativeBinomial` from `Distributions.jl` is parameterized following [1]. With the parameterization in [2], we can solve
for `r` (`n` in [1]) and `p` by matching the mean and the variance given in `μ` and `ϕ`.
We have the following two equations
(1) μ = r (1 - p) / p
(2) μ + μ^2 / ϕ = r (1 - p) / p^2
Substituting (1) into the RHS of (2):
  μ + (μ^2 / ϕ) = μ / p
⟹ 1 + (μ / ϕ) = 1 / p
⟹ p = 1 / (1 + μ / ϕ)
⟹ p = (1 / (1 + μ / ϕ)
Then in (1) we have
  μ = r (1 - (1 / 1 + μ / ϕ)) * (1 + μ / ϕ)
⟹ μ = r ((1 + μ / ϕ) - 1)
⟹ r = ϕ
Hence, the resulting map is `(μ, ϕ) ↦ NegativeBinomial(ϕ, 1 / (1 + μ / ϕ))`.

## References
[1] https://reference.wolfram.com/language/ref/NegativeBinomialDistribution.html
[2] https://mc-stan.org/docs/2_20/functions-reference/nbalt.html
"""
function NegativeBinomial2(μ, ϕ)
    p = 1 / (1 + μ / ϕ)
    r = ϕ

    return NegativeBinomial(r, p)
end

NegativeBinomial3(μ, ϕ) = NegativeBinomial2(μ, μ / ϕ)

@inline nbinomlogpdf3(μ, ϕ, k) = nbinomlogpdf2(μ, μ / ϕ, k)
@inline function nbinomlogpdf2(μ, ϕ, k)
    p = 1 / (1 + μ / ϕ)
    r = ϕ

    return nbinomlogpdf(p, r, k)
end

@inline function nbinomlogpdf(p, r, k)
    r_ = r * log(p) + k * log1p(-p)
    return r_ - log(k + r) - SpecialFunctions.logbeta(r, k + 1)
end

"""
    GammaMeanCv(mean, cv)

Mean-variance-coefficient parameterization of `Gamma`.

## References
- https://www.rdocumentation.org/packages/EnvStats/versions/2.3.1/topics/GammaAlt
"""
function GammaMeanCv(mean, cv)
    k = cv^(-2)
    θ = mean / k
    return Gamma(k, θ)
end


@doc raw"""
    AR1(num_times, α, μ, σ)

1-th order Autoregressive model for `num_times` steps and autocorrelation `α`.

Specifically, it defines the process
```math
\begin{align*}
\xi_t & \sim \mathcal{N}(0, 1) & \quad \forall t = 1, \dots, T \\
X_1 &= \xi_1 \\
X_t &= \alpha X_{t - 1} + \sqrt{1 - \alpha^2} \xi_t & \quad \forall t = 2, \dots, T
\end{align*}
```

## Arguments
- `num_times::Int`: number of time steps for the model
- `α`: autocorrelation for.
"""
struct AR1{T1, T2, T3} <: ContinuousMultivariateDistribution
    num_times::Int
    alpha::T1
    mu::T2
    sigma::T3
end

Base.size(ar::AR1) = (ar.num_times, )
Base.length(ar::AR1) = ar.num_times
Base.eltype(::AR1{T1, T2, T3}) where {T1, T2, T3} = promote_type(
    eltype(T1), eltype(T2), eltype(T3)
)

function Distributions.rand(rng::Random.AbstractRNG, ar::AR1)    
    # Sample
    ξ = randn(rng, length(ar))
    
    # Pre-compute the scaling factor
    δ = sqrt(1 - ar.alpha^2)
    
    # Pre-allocate
    x = similar(ξ)
    
    # Compute
    x[1] = ξ[1]
    for t = 2:length(ar)
        x[t] = ar.alpha * x[t - 1] + δ * ξ[t]
    end
    
    # μ (mean of the process) + σ * xₜ (scale everything)
    return @. ar.mu + ar.sigma * x
end

function Distributions.logpdf(ar::AR1, y::AbstractVector{<:Real})
    δ = sqrt(1 - ar.alpha^2)
    
    # Recover `x`
    x = @. (y / ar.sigma) - ar.mu
    
    # Compute the mean for x[2:end]
    μ = ar.alpha .* x[1:end - 1]
    
    return StatsFuns.normlogpdf(x[1]) + sum(StatsFuns.normlogpdf.((x[2:end] .- μ) ./ δ))
end
