import StatsFuns: normlogpdf
using Bijectors.Functors

### Convenience methods ###

# TODO: Currently the trick we do with `^inv(ρ)` means that this is only
# valid for Matern(1/2) kernel. We should just pass in the distance-matrices instead.
function spatial_L(K_spatial_nonscaled, K_local, σ_spatial, σ_local)
    # Use `PDMats.ScalMat` to ensure that positive-definiteness is preserved
    # K_spatial = ScalMat(size(K_spatial_nonscaled, 1), σ_spatial^2) * K_spatial_nonscaled
    # K_local = ScalMat(size(K_local, 1), σ_local^2) * K_local
    # HACK: use this until we have an adjoint for `ScalMat` constructor in ChainRulesCore.jl
    K_spatial = σ_spatial^2 .* K_spatial_nonscaled
    K_local = σ_local^2 .* K_local

    K_space = K_local + K_spatial # `PDMat` is a no-op if the input is already a `PDMat`
    L_space = cholesky(K_space).L

    return L_space
end

function spatial_L(K_spatial_nonscaled, K_local, σ_spatial, σ_local, ρ_spatial)
    return spatial_L(K_spatial_nonscaled .^ inv.(ρ_spatial), K_local, σ_spatial, σ_local)
end

time_U(K_time) = cholesky(K_time).U
time_U(K_time, ρ_time) = time_U(K_time .^ inv.(ρ_time))

@doc raw"""
    rmap_naive(args...; kwargs...)

Naive implementation of full Rmap model.

# Arguments
- `C::AbstractMatrix`: cases in a matrix of size `(num_regions, num_times)`.
- `D::AbstractVector`: testing delay distribution of length `< num_times`, i.e. `D[t]` is
    the probability that an infected person tests positive after `t` steps.
- `W::AbstractVector`: generation distribution/infection profile of length `< num_times`, i.e.
    `W[t]` is the probability of a secondary infection after `t` steps.
- `F_id::AbstractMatrix`: diagonal flux matrix representing local infection.
- `F_out::AbstractMatrix`: flux matrix representing outgoing infections.
- `F_in::AbstractMatrix`: flux matrix representing incoming infections.
- `K_time::AbstractPDMat`: (positive-definite) kernel matrix for the time kernel.
- `K_spatial::AbstractPDMat`: (positive-definite) kernel matrix for the spatial kernel.
- `K_local::AbstractPDMat`: (positive-definite) kernel matrix for the local spatial kernel.
- `days_per_step = 1`: specifies how many days of data each time step corresponds to.
- `X_cond::AbstractMatrix=nothing`: Precomputed Xt before the start of the modelling period to condition on.

# Keyword arguments
- `ρ_spatial = missing`: length scale / "stretch" applied to `K_spatial`.
- `ρ_time = missing`: length scale / "stretch" applied to `K_time`.
- `σ_spatial = missing`: scale applied to `K_spatial`.
- `σ_local = missing`: scale applied to `K_local`.
- `σ_ξ = missing`: square root of variance of the "global" infection pressure `ξ`.


Note that those with default value `missing` will be sampled if not specified.

# Examples
```julia
julia> # Load data.
       data = Rmap.load_data();

julia> # Convert `data` into something compatible with `Rmap.rmap_naive`
       args, dates = Rmap.setup_args(Rmap.rmap_naive, data; num_steps=15, timestep=Week(1), include_dates=true);

julia> # Instantiate the model.
       m = Rmap.rmap_naive(args...; ρ_spatial=0.1, ρ_time=100);
```

# Mathematical description
```math
\begin{align*}
  \psi & \sim \mathcal{N}_{ + }(0, 5) \\
  \phi_i & \sim \mathcal{N}_{ + }(0, 5) & \quad \forall i = 1, \dots, n \\
  \underline{\text{Time:}} \\
  \rho_{\mathrm{time}} & \sim \mathcal{N}_{ + }(0, 5) \\
  \sigma_{\mathrm{time}} & \sim \mathcal{N}_{ + }(0, 5) \\
  % \big( K_{\mathrm{time}} \big)_{t, t'} & := \sigma_{\mathrm{time}}^2 k_{\mathrm{time}}(t, t')^{1 / \rho_{\mathrm{time}}} & \quad \forall t, t' = 1, \dots, T \\
  % L_{\mathrm{time}} & := \mathrm{cholesky}(K_{\mathrm{time}}) \\
  f_{\mathrm{time}} & \sim \mathrm{GP} \Big( 0, \sigma_{\mathrm{time}}^2 k_{\mathrm{time}}^{1 / \rho_{\mathrm{time}}} \Big) \\
  \underline{\text{Space:}} \\
  \rho_{\mathrm{spatial}} & \sim \mathcal{N}_{ + }(0, 5) \\
  \sigma_{\mathrm{spatial}} & \sim \mathcal{N}_{ + }(0, 5) \\
  \sigma_{\mathrm{local}} & \sim \mathcal{N}_{ + }(0, 5) \\
  % \big( k_{\mathrm{spatial}} \big)_{i, j} & := \sigma_{\mathrm{local}}^2 \delta_{i, j} + \sigma_{\mathrm{spatial}}^2 k_{\mathrm{spatial}}(i, j)^{1 / \rho_{\mathrm{spatial}}} & \quad \forall i, j = 1, \dots, n \\
  % L_{\mathrm{space}} & := \mathrm{cholesky}(k_{\mathrm{spatial}}) \\
  f_{\mathrm{space}} & \sim \mathrm{GP} \Big( 0, \sigma_{\mathrm{local}}^2 \delta_{i, j} + \sigma_{\mathrm{spatial}}^2 k_{\mathrm{spatial}}^{1 / \rho_{\mathrm{spatial}}} \Big) \\
  \underline{\text{R-value:}} \\
  % E_{i, t} & \sim \mathcal{N}(0, 1) & \quad \forall i = 1, \dots, n, \quad t = 1, \dots, T \\
  % f & := L_{\mathrm{space}} \ E \ L_{\mathrm{time}}^T \\
  f & := f_{\mathrm{time}}(1, \dots, T) + f_{\mathrm{space}} \big( (x_1, y_1), \dots, (x_n, y_n) \big) \\
  R & := \exp(f) \\
  \underline{\text{AR-process:}} \\
  \mu_{\mathrm{AR}} & \sim \mathcal{N}(-2.19, 0.25) \\
  \sigma_{\mathrm{AR}} & \sim \mathcal{N}_{ + }(0, 0.25) \\
  \tilde{\alpha} & \sim \mathcal{N}\big(0, 1 - e^{- \Delta t / 28} \big) \\
  \alpha & := 1 - \mathrm{constrain}(\tilde{\alpha}, 0, 1) \\
  \tilde{\rho} & \sim \mathrm{AR}_1(\alpha, \mu_{\mathrm{AR}}, \sigma_{\mathrm{AR}}) \\
  \rho_t &:= \mathrm{constrain}(\tilde{\rho}_t, 0, 1) & \quad \forall t = 1, \dots, T \\
  \underline{\text{Flux matrix:}} \\
  \beta & \sim \mathrm{Uniform}(0, 1) \\
  F_{t} & := (1 - \rho_t) F_{\mathrm{id}} + \rho_t \big(\beta F_{\mathrm{fwd}} + (1 - \beta) F_{\mathrm{rev}} \big) & \quad \forall t = 1, \dots, T \\
  \underline{\text{Latent process:}} \\
  \xi & \sim \mathcal{N}_{ + }(0, \sigma_{\xi}^2) \\
  Z_{i, t} & := \sum_{\tau = 1}^{t} I(\tau < T_{\mathrm{flux}}) X_{i, t - \tau} W_{\tau} & \quad \forall i = 1, \dots, n, \quad t = 1, \dots, T \\
  \tilde{Z}_{i, t} & := \sum_{i = 1}^{n} F_{i, t} Z_{i, t} & \quad \forall  i = 1, \dots, n, \quad t = 1, \dots, T \\
  X_{i, t} & \sim \mathrm{NegativeBinomial3}\big(R_{i, t} \tilde{Z}_{i, t} + \xi, \psi\big) & \quad \forall  i = 1, \dots, n, \quad t = 1, \dots, T \\
  \underline{\text{Observe:}} \\
  C_{i, t} & \sim \mathrm{NegativeBinomial3}\bigg( \sum_{\tau = 1}^{t} I(\tau < T_{\mathrm{test}}) X_{i, t - \tau} D_{\tau}, \ \phi_i \bigg) & \quad \forall  i = 1, \dots, n, \quad t = 1, \dots, T
\end{align*}
```
"""
@model function rmap_naive(
    C, D, W,
    F_id, F_out, F_in,
    K_time, K_spatial, K_local,
    days_per_step = 1,
    X_cond = nothing,
    ::Type{T} = Float64;
    ρ_spatial = missing, ρ_time = missing,
    σ_spatial = missing, σ_local = missing,
    σ_ξ = missing
) where {T}
    num_regions = size(C, 1)
    num_infer = size(C, 2)
    num_cond = X_cond === nothing ? 0 : size(X_cond, 2)
    num_times = num_infer + num_cond

    @assert num_infer % days_per_step == 0
    num_steps = num_infer ÷ days_per_step

    ### GP prior ###
    @submodel R = SpatioTemporalGP(K_spatial, K_local, K_time, T; σ_spatial, σ_local, ρ_spatial, ρ_time)

    ### Flux ###
    @submodel (X, Z) = RegionalFluxNaive(F_id, F_in, F_out, W, R, X_cond; days_per_step, σ_ξ)

    # Observe (if we're done imputing)
    @submodel (C, B) = NegBinomialWeeklyAdjustedTestingNaive(C, X, D, num_cond)

    return (;
        R = repeat(R, inner=(1, days_per_step)),
        X = X[:, (num_cond + 1):end],
        B = B,
        C = C,
        Z
    )
end

@doc raw"""
    SpatioTemporalGP(K_spatial, K_local, K_time[, ::Type{T}]; kwargs...)

Model of Rt for each region and time using a spatio-temporal Gaussian process.

# Arguments
- `K_time::AbstractPDMat`: (positive-definite) kernel matrix for the time kernel.
- `K_spatial::AbstractPDMat`: (positive-definite) kernel matrix for the spatial kernel.
- `K_local::AbstractPDMat`: (positive-definite) kernel matrix for the local spatial kernel.

# Keyword arguments
- `σ_spatial = missing`: scale used for the cross-region spatial component of the GP.
- `σ_local = missing`: scale used for the local region component of the GP.
- `ρ_spatial = missing`: length-scale used for the cross-region spatial component of the GP.
- `ρ_time = missing`: length-scale used for the temporal component of the GP.

# Mathematical description
```math
\begin{align*}
  \underline{\text{Time:}} \\
  \rho_{\mathrm{time}} & \sim \mathcal{N}_{ + }(0, 5) \\
  \sigma_{\mathrm{time}} & \sim \mathcal{N}_{ + }(0, 5) \\
  % \big( K_{\mathrm{time}} \big)_{t, t'} & := \sigma_{\mathrm{time}}^2 k_{\mathrm{time}}(t, t')^{1 / \rho_{\mathrm{time}}} & \quad \forall t, t' = 1, \dots, T \\
  % L_{\mathrm{time}} & := \mathrm{cholesky}(K_{\mathrm{time}}) \\
  f_{\mathrm{time}} & \sim \mathrm{GP} \Big( 0, \sigma_{\mathrm{time}}^2 k_{\mathrm{time}}^{1 / \rho_{\mathrm{time}}} \Big) \\
  \underline{\text{Space:}} \\
  \rho_{\mathrm{spatial}} & \sim \mathcal{N}_{ + }(0, 5) \\
  \sigma_{\mathrm{spatial}} & \sim \mathcal{N}_{ + }(0, 0.5) \\
  \sigma_{\mathrm{local}} & \sim \mathcal{N}_{ + }(0, 0.5) \\
  % \big( k_{\mathrm{spatial}} \big)_{i, j} & := \sigma_{\mathrm{local}}^2 \delta_{i, j} + \sigma_{\mathrm{spatial}}^2 k_{\mathrm{spatial}}(i, j)^{1 / \rho_{\mathrm{spatial}}} & \quad \forall i, j = 1, \dots, n \\
  % L_{\mathrm{space}} & := \mathrm{cholesky}(k_{\mathrm{spatial}}) \\
  f_{\mathrm{space}} & \sim \mathrm{GP} \Big( 0, \sigma_{\mathrm{local}}^2 \delta_{i, j} + \sigma_{\mathrm{spatial}}^2 k_{\mathrm{spatial}}^{1 / \rho_{\mathrm{spatial}}} \Big) \\
  \underline{\text{R-value:}} \\
  % E_{i, t} & \sim \mathcal{N}(0, 1) & \quad \forall i = 1, \dots, n, \quad t = 1, \dots, T \\
  % f & := L_{\mathrm{space}} \ E \ L_{\mathrm{time}}^T \\
  f & := f_{\mathrm{time}}(1, \dots, T) + f_{\mathrm{space}} \big( (x_1, y_1), \dots, (x_n, y_n) \big) \\
  R & := \exp(f)
\end{align*}
```
"""
@model function SpatioTemporalGP(
    K_spatial, K_local, K_time,
    ::Type{T} = Float64;
    σ_spatial = missing,
    σ_local = missing,
    ρ_spatial = missing,
    ρ_time = missing,
    shift = 0.0
) where {T}
    num_steps = size(K_time, 1)
    num_regions = size(K_spatial, 1)

    # Length scales
    ρ_spatial ~ 𝒩₊(T(0), T(5))
    ρ_time ~ 𝒩₊(T(0), T(5))

    # Scales
    σ_spatial ~ 𝒩₊(T(0), T(0.5))
    σ_local ~ 𝒩₊(T(0), T(0.5))

    # GP
    E_vec ~ MvNormal(num_regions * num_steps, one(T))
    E = reshape(E_vec, (num_regions, num_steps))

    # Get cholesky decomps using precomputed kernel matrices
    L_space = spatial_L(K_spatial, K_local, σ_spatial, σ_local, ρ_spatial)
    U_time = time_U(K_time, ρ_time)

    # Obtain realization of log-R.
    f = L_space * E * U_time

    # Compute R.
    R = exp.(f .- T(shift))
    return R
end

@model function LogisticAR1(
    num_steps, ::Type{T} = Float64;
    days_per_step = 1, α_pre = missing, μ_ar = missing, σ_ar = missing
) where {T}
    # AR(1) prior
    # set mean of process to be 0.1, 1 std = 0.024-0.33
    μ_ar ~ Normal(T(-2.19), T(0.25))
    σ_ar ~ 𝒩₊(T(0.0), T(0.25))

    # 28 likely refers to the number of days in a month, and so we're scaling the autocorrelation
    # wrt. number of days used in each time-step (specified by `days_per_step`).
    σ_α = T(1 - exp(- days_per_step / 28))
    α_pre ~ transformed(Normal(T(0), σ_α), inv(Bijectors.Logit(T(0), T(1))))
    α = 1 - α_pre

    # Use bijector to transform to have support (0, 1) rather than ℝ.
    b = Bijectors.Logit{1}(T(0), T(1))
    ρₜ ~ transformed(AR1(num_steps, α, μ_ar, σ_ar), inv(b))

    return ρₜ
end

@model function NegBinomialWeeklyAdjustedTestingNaive(
    C, X, D, num_cond;
    weekly_case_variation = missing,
    ϕ = missing
)
    T = eltype(X)
    num_infer = size(C, 2)
    num_regions = size(C, 1)
    test_delay_cutoff = length(D)

    # Noise for cases
    ϕ ~ filldist(𝒩₊(0, 5), num_regions)

    # Weekly variation
    weekly_case_variation ~ Turing.DistributionsAD.TuringDirichlet(7 * ones(7))

    # `B` is the observations _without_ weekly adjustment.
    B = similar(C)
    for t = 1:num_infer
        # `X` consists of _both_ `X_cond` and the sampled `X`.
        ts_prev_delay = reverse(max(1, num_cond + t - test_delay_cutoff):num_cond + t - 1)
        # Clamp the values to avoid numerical issues during sampling from the prior.
        expected_positive_tests = clamp.(
            X[:, ts_prev_delay] * D[1:min(test_delay_cutoff, num_cond + t - 1)],
            T(1e-3),
            T(1e7)
        )

        expected_positive_tests_weekly_adj = (
            7 * weekly_case_variation[(t % 7) + 1] * expected_positive_tests
        )

        for i = 1:num_regions
            B[i, t] = rand(NegativeBinomial3(expected_positive_tests[i], ϕ[i]))
            C[i, t] ~ NegativeBinomial3(expected_positive_tests_weekly_adj[i], ϕ[i])
        end
    end

    return C, B
end

@doc raw"""
    NegBinomialWeeklyAdjustedTesting(C, X, D, num_cond[, ::Type{T}]; kwargs...)

Model for number of cases using a negative binomial with adjustment for within-week variation.

Return cases `C`, either sampled or observed.

# Arguments
- `C::AbstractMatrix`: cases in a matrix of size `(num_regions, num_times)`.
- `X::AbstractMatrix`: latent infections of size `(num_regions, num_times)`, e.g. as
    returned by [`RegionalFlux`](@ref).
- `D::AbstractVector`: testing delay distribution of length `< num_times`, i.e. `D[t]` is
    the probability that an infected person tests positive after `t` steps.

# Keyword arguments
- `weekly_case_variation = missing`: weekly case variation used.
- `ϕ = missing`: region-specific variance parameter for the likelihood.

# Mathematical model
```math
\begin{align*}
  \phi_i & \sim \mathcal{N}_{ + }(0, 5) & \quad \forall i = 1, \dots, n \\
  w & \sim \mathrm{Dirichlet}(7, 7) \\
  C_{i, t} & \sim \mathrm{NegativeBinomial3}\bigg( w_{t \mod 7}\sum_{\tau = 1}^{t} I(\tau < T_{\mathrm{test}}) X_{i, t - \tau} D_{\tau}, \ \phi_i \bigg) & \quad \forall  i = 1, \dots, n, \quad t = 1, \dots, T
\end{align*}
```
"""
@model function NegBinomialWeeklyAdjustedTesting(
    C, X, D, num_cond, ::Type{T} = Float64;
    weekly_case_variation = missing, ϕ = missing
) where {T}
    # Noise for cases
    num_regions = size(X, 1)
    ϕ ~ filldist(𝒩₊(T(0), T(5)), num_regions)

    # Weekly variation
    weekly_case_variation ~ Turing.DistributionsAD.TuringDirichlet(7 * ones(T, 7))

    # TODO: Should we remove this? We only do this to ensure that the results are
    # identical to `rmap_naive`.
    # Ensures that we'll be using the same ordering as the original model.
    weekly_case_variation_reindex = map(1:7) do i
        (i % 7) + 1
    end
    weekly_case_variation = weekly_case_variation[weekly_case_variation_reindex]

    # Convolution.
    # Clamp the values to avoid numerical issues during sampling from the prior.
    expected_positive_tests = clamp.(Epimap.conv(X, D)[:, num_cond + 1:end], T(1e-3), T(1e7))

    # Repeat one too many times and then extract the desired section `1:num_regions`
    num_days = size(expected_positive_tests, 2)
    weekly_case_variation = transpose(
        repeat(weekly_case_variation, outer=(num_days ÷ 7) + 1)[1:num_days]
    )

    expected_positive_tests_weekly_adj = 7 * expected_positive_tests .* weekly_case_variation

    # Observe
    # TODO: This should be done in a better way.
    if ismissing(C)
        C ~ arraydist(NegativeBinomial3.(expected_positive_tests_weekly_adj, ϕ))
    else
        # We extract only the time-steps after the imputation-step
        Turing.@addlogprob! sum(nbinomlogpdf3.(
            expected_positive_tests_weekly_adj,
            ϕ,
            T.(C) # conversion ensures precision is preserved
        ))
    end

    return C
end

@model function RegionalFluxPrior(
    num_steps,
    ::Type{T} = Float64;
    days_per_step = 1,
    σ_ξ = missing,
    ξ = missing,
    β = missing,
    ρₜ = missing,
    ψ = missing,
) where {T}
    # Noise for latent infections.
    ψ ~ 𝒩₊(T(0), T(5))

    # Global infection.
    σ_ξ ~ 𝒩₊(T(0), T(5))
    ξ ~ 𝒩₊(T(0), σ_ξ)

    # AR(1) prior
    @submodel ρₜ = LogisticAR1(num_steps, T; days_per_step)

    β ~ Uniform(T(0), T(1))

    return (; ψ, σ_ξ, ξ, ρₜ, β)
end

@model function RegionalFluxNaive(
    F_id, F_in, F_out,
    W, R, X_cond,
    ::Type{TV} = Matrix{Float64};
    days_per_step = 1,
    σ_ξ = missing,
    ξ = missing,
    β = missing,
    ρₜ = missing,
    ψ = missing
) where {TV}
    T = eltype(TV)

    num_steps = size(R, 2)
    num_cond = size(X_cond, 2)
    num_regions = size(F_in, 1)
    num_times = num_steps * days_per_step + num_cond

    prev_infect_cutoff = length(W)

    @submodel (ψ, σ_ξ, ξ, ρₜ, β) = RegionalFluxPrior(num_steps, T; days_per_step, σ_ξ, ξ, β, ρₜ, ψ)

    X = TV(undef, (num_regions, num_times - num_cond))
    X_full = TV(undef, (num_regions, num_times))

    Z = TV(undef, (num_regions, num_times - num_cond))

    if X_cond !== nothing
        X_full[:, 1:num_cond] = X_cond
    end

    for t = (num_cond + 1):num_times
        # compute the index of the step this day is in
        t_step = (t - num_cond - 1) ÷ days_per_step + 1

        # Flux matrix
        Fₜ = @. (1 - ρₜ[t_step]) * F_id + ρₜ[t_step] * (β * F_out + (1 - β) * F_in) # Eq. (16)

        # Eq. (4)
        # offset t's to account for the extra conditioning days of Xt
        ts_prev_infect = reverse(max(1, t - prev_infect_cutoff):t - 1)
        Zₜ = X_full[:, ts_prev_infect] * W[1:min(prev_infect_cutoff, t - 1)]
        Z̃ₜ = Fₜ * Zₜ # Eq. (5)

        # Use continuous approximation if the element type of `X` is non-integer.
        μ = R[:, t_step] .* Z̃ₜ .+ ξ
        if eltype(X) <: Integer
           for i = 1:num_regions
               X[i, t - num_cond] ~ NegativeBinomial3(μ[i], ψ; check_args=false)
            end
        else
            # Eq. (15), though there they use `Zₜ` rather than `Z̃ₜ`; I suspect they meant `Z̃ₜ`.
            for i = 1:num_regions
                X[i, t - num_cond] ~ 𝒩₊(μ[i], sqrt((1 + ψ) * μ[i]))
            end
        end

        # Update `X`.
        X_full[:, t] = X[:, t - num_cond]

        # Save the computed `R` value.
        Z[:, t - num_cond] = Zₜ
    end

    return (; X_full, Z)
end

@doc raw"""
    RegionalFlux(F_id, F_in, F_out, W, R, X_cond[, ::Type{T}]; kwargs...)

Model latent infections `X` using a regional flux model.

# Arguments
- `F_id::AbstractMatrix`: diagonal flux matrix representing local infection.
- `F_out::AbstractMatrix`: flux matrix representing outgoing infections.
- `F_in::AbstractMatrix`: flux matrix representing incoming infections.
- `W::AbstractVector`: generation distribution/infection profile of length `< num_times`, i.e.
    `W[t]` is the probability of a secondary infection after `t` steps.
- `R::AbstractMatrix`: Rt estimate for region `i` at time `t`, e.g. as returned by [`SpatioTemporalGP`](@ref).
- `X_cond::AbstractMatrix=nothing`: Precomputed Xt before the start of the modelling period to condition on.

# Keyword arguments
- `σ_ξ = missing`: square root of variance of the "global" infection pressure `ξ`.
- `ξ = missing`: "global" infection pressure.
- `β = missing`: weight used in convex combination of of `F_in` and `F_out`.
- `ρₜ = missing`: autoregressive process modelling time-variation of flux-matrices.
- `ψ = missing`: scale for latent infections.

# Mathematical description
```math
\begin{align*}
  \underline{\text{AR-process:}} \\
  \mu_{\mathrm{AR}} & \sim \mathcal{N}(-2.19, 0.25) \\
  \sigma_{\mathrm{AR}} & \sim \mathcal{N}_{ + }(0, 0.25) \\
  \tilde{\alpha} & \sim \mathcal{N}\big(0, 1 - e^{- \Delta t / 28} \big) \\
  \alpha & := 1 - \mathrm{constrain}(\tilde{\alpha}, 0, 1) \\
  \tilde{\rho} & \sim \mathrm{AR}_1(\alpha, \mu_{\mathrm{AR}}, \sigma_{\mathrm{AR}}) \\
  \rho_t &:= \mathrm{constrain}(\tilde{\rho}_t, 0, 1) & \quad \forall t = 1, \dots, T \\
  \underline{\text{Flux matrix:}} \\
  \beta & \sim \mathrm{Uniform}(0, 1) \\
  F_{t} & := (1 - \rho_t) F_{\mathrm{id}} + \rho_t \big(\beta F_{\mathrm{fwd}} + (1 - \beta) F_{\mathrm{rev}} \big) & \quad \forall t = 1, \dots, T
\end{align*}
```
"""
@model function RegionalFlux(
    F_id, F_in, F_out,
    W, R, X_cond,
    ::Type{T} = Float64;
    days_per_step = 1,
    σ_ξ = missing,
    ξ = missing,
    β = missing,
    ρₜ = missing,
    ψ = missing,
) where {T}
    num_steps = size(R, 2)
    num_cond = size(X_cond, 2)
    num_regions = size(F_in, 1)

    @submodel (ψ, σ_ξ, ξ, ρₜ, β) = RegionalFluxPrior(num_steps, T; days_per_step, σ_ξ, ξ, β, ρₜ, ψ)

    # Compute the flux matrix
    F = compute_flux(F_id, F_in, F_out, β, ρₜ, days_per_step)

    # Daily latent infections.
    num_infer = size(F, 3)
    @assert num_infer % days_per_step == 0

    # NOTE: Apparently AD-ing through `filldist` for large dimensions is bad.
    # So we're just going to ignore the log-computation (it's 0 for `Flat`) in the
    # case where we are evaluating the logjoint and extract from `__varinfo__`.
    # This brought us ~400ms/grad → ~200ms/grad for the "standard" setup.
    if Epimap.issampling(__context__) || !(__varinfo__ isa DynamicPPL.SimpleVarInfo)
        X ~ filldist(FlatPos(zero(T)), num_regions, num_infer)
    else
        X = __varinfo__.θ.X
    end
    X_full = hcat(X_cond, X)

    # Compute the logdensity
    Turing.@addlogprob! logjoint_X(F, X_full, W, R, ξ, ψ, num_cond, days_per_step)

    return X_full
end

@model function RegionalFluxWithoutCond(
    F_id, F_in, F_out,
    W, R, X_cond_means,
    populations,
    ::Type{T} = Float64;
    days_per_step = 1,
    σ_ξ = missing,
    ξ = missing,
    β = missing,
    ρₜ = missing,
    ψ = missing,
) where {T}
    num_steps = size(R, 2)
    num_cond = size(X_cond_means, 2)
    num_regions = size(F_in, 1)

    @submodel (ψ, σ_ξ, ξ, ρₜ, β) = RegionalFluxPrior(
        num_steps, T;
        days_per_step, σ_ξ, ξ, β, ρₜ, ψ
    )

    # Compute the flux matrix
    F = compute_flux(F_id, F_in, F_out, β, ρₜ, days_per_step)

    # Daily latent infections.
    num_infer = size(F, 3)
    @assert num_infer % days_per_step == 0

    # Initial latent infections.
    # NOTE: We add a small constant to the mean to ensure that mean 0 won't cause any issues.
    # NOTE: AD-ing through `arraydist` and `truncated` for large dimensions is bad.
    # So we'll do it "manually" in the case where we are:
    # 1. Evaluating, not sampling, and
    # 2. we're working with a `SimpleVarInfo` which supports extraction of the value.
    # This brought us ~1200ms/grad → ~400ms/grad for the "standard" setup.
    if Epimap.issampling(__context__) || !(__varinfo__ isa DynamicPPL.SimpleVarInfo)
        X_cond ~ filldist(Beta(T(2), T(5)), num_regions)
    else
        # With Zygote.jl:
        # - `truncated` is slow
        # - `arraydist` is slow
        # Hence we use a the following instead.
        X_cond = __varinfo__.θ.X_cond
        Turing.@addlogprob! sum(StatsFuns.betalogpdf.(T(2), T(5), X_cond))
    end
    X_cond = populations .* repeat(X_cond ./ num_cond, inner=(1, num_cond))

    # Latent infections for which we have observations.
    # NOTE: Apparently AD-ing through `filldist` for large dimensions is bad.
    # So we're just going to ignore the log-computation (it's 0 for `Flat`) in the
    # case where we are evaluating the logjoint and extract from `__varinfo__`.
    # This brought us ~400ms/grad → ~200ms/grad for the "standard" setup.
    if Epimap.issampling(__context__) || !(__varinfo__ isa DynamicPPL.SimpleVarInfo)
        X ~ filldist(FlatPos(zero(T)), num_regions, num_infer)
    else
        X = __varinfo__.θ.X
    end
    X_full = hcat(X_cond, X)

    # Compute the logdensity
    Turing.@addlogprob! logjoint_X(F, X_full, W, R, ξ, ψ, num_cond, days_per_step)

    return X_full
end


"""
    rmap(args...; kwargs...)

Vectorized implementation of full Rmap model.

See [`rmap_naive`](@ref) for description of arguments, the model and example code,
where you simply have to replace `rmap_naive` with `rmap`.
"""
@model function rmap(
    C, D, W,
    F_id, F_out, F_in,
    K_time, K_spatial, K_local,
    days_per_step = 1,
    X_cond = nothing,
    ::Type{T} = Float64;
    ρ_spatial = missing, ρ_time = missing,
    σ_spatial = missing, σ_local = missing,
    σ_ξ = missing
) where {T}
    num_cond = size(X_cond, 2)

    # GP-model for R-value.
    @submodel R = SpatioTemporalGP(K_spatial, K_local, K_time, T; σ_spatial, σ_local, ρ_spatial, ρ_time)

    # Latent infections.
    @submodel X = RegionalFlux(F_id, F_in, F_out, W, R, X_cond, T; days_per_step, σ_ξ)

    # Likelihood.
    @submodel C = NegBinomialWeeklyAdjustedTesting(C, X, D, num_cond, T)

    return (
        R = repeat(R, inner=(1, days_per_step)),
        X = X[:, num_cond + 1:end],
        C = C,
    )
end

@model function rmap_debiased(
    logitπ, σ_debias, populations,
    D, W,
    F_id, F_out, F_in,
    K_time, K_spatial, K_local,
    days_per_step,
    X_cond_means,
    observation_projection,
    ::Type{T} = Float64;
    skip_weeks_observe=0,
    ρ_spatial = missing, ρ_time = missing,
    σ_spatial = missing, σ_local = missing,
    σ_ξ = missing,
    β = missing,
    ρₜ = missing,
) where {T}
    num_cond = size(X_cond_means, 2)

    # GP-model for R-value.
    @submodel R = SpatioTemporalGP(K_spatial, K_local, K_time, T; σ_spatial, σ_local, ρ_spatial, ρ_time)

    # Latent infections.
    @submodel X = RegionalFluxWithoutCond(
        F_id, F_in, F_out,
        W, R,
        X_cond_means,
        populations,
        T;
        days_per_step, σ_ξ
    )

    # Likelihood.
    @submodel logitπ = DebiasedLikelihood(logitπ, σ_debias, populations, X, D, num_cond, observation_projection, T; skip_weeks_observe)

    return (
        R = repeat(R, inner=(1, days_per_step)),
        X = X[:, num_cond + 1:end],
        logitπ = logitπ,
    )
end

@model function DebiasedLikelihood(
    logitπ, σ_debias, populations, X, D, num_cond, P, ::Type{T}=Float64;
    skip_weeks_observe=0
) where {T}
    # Convolution.
    # Clamp the values to avoid numerical issues during sampling from the prior.
    expected_positive_tests = clamp.(
        Epimap.conv(X, D)[:, 7 * skip_weeks_observe + num_cond + 1:end],
        T(1e-3),
        T(1e7)
    )

    # Accumulate the weekly cases.
    # TODO: Implement something faster? Guessing the adjoint isn't the most performant.
    expected_positive_tests_weekly = mapreduce(
        x -> mean(x, dims=2),
        hcat,
        (@views(expected_positive_tests[:, i:i + 6]) for i = 1:7:size(expected_positive_tests, 2))
    )

    # Compute proportions.
    # NOTE: This is also where we project the latent regions onto the observed regions
    # using the projection matrix `P`. Furthermore, note that we also aggregate
    # the `populations`.
    expected_weekly_proportions = clamp.(
        (P * expected_positive_tests_weekly) ./ (P * populations),
        zero(T),
        one(T)
    )
    # Observe.
    if logitπ === missing
        logitπ ~ arraydist(Normal.(StatsFuns.logit.(expected_weekly_proportions), σ_debias))
    else
        Turing.@addlogprob! sum(@. StatsFuns.normlogpdf((logitπ - StatsFuns.logit(expected_weekly_proportions)) / σ_debias) - log(σ_debias))
    end

    return logitπ
end

function compute_flux(F_id, F_in, F_out, β, ρₜ, days_per_step = 1)
    ρₜ = repeat(ρₜ, inner=days_per_step)

    # Compute the full flux
    F_cross = @. β * F_out + (1 - β) * F_in
    oneminusρₜ = @. 1 - ρₜ

    # Tullio.jl doesn't seem to work nicely with `Diagonal`.
    F_id_ = F_id isa Diagonal ? Matrix(F_id) : F_id

    # NOTE: This is a significant bottleneck in the gradient computation.
    @tullio F[i, j, t] := oneminusρₜ[t] * F_id_[i, j] + ρₜ[t] * F_cross[i, j]

    # # NOTE: Doesn't seem faster than the above code.
    # # Everything within one operation to minimize memory-allocation.
    # # Also allows us to not compute gradients through the flux-matrices!
    # β_arr = FillArrays.Fill(β, size(F_id))
    # oneminusβ_arr = FillArrays.Fill(1 - β, size(F_id))
    # @tullio F[i, j, t] := ρₜ[t] * F_id[i, j] + oneminusρₜ[t] * (β_arr[i, j] * F_out[i, j] + oneminusβ_arr[i, j] * F_in[i, j]) nograd=(F_id, F_in, F_out)

    # F = @tensor begin
    #     F[i, j, t] := ρₜ[t] * F_id[i, j] + oneminusρₜ[t] * F_cross[i, j]
    # end

    # # Equivalent to the above `@tensor`
    # res1 = kron(1 .- ρₜ', F_cross)
    # res2 = kron(ρₜ', F_id)
    # F = reshape(res2 + res1, size(F_cross)..., length(ρₜ))

    return F
end

function logjoint_X(F, X_full, W, R, ξ, ψ, num_cond, days_per_step = 1)
    R_expanded = repeat(R, inner=(1, days_per_step))
    return logjoint_X_halfnorm(F, X_full, W, R_expanded, ξ, ψ, num_cond)
end

@inline function logjoint_X_halfnorm(F, X_full, W, R, ξ, ψ, num_cond)
    # Convolve `X` with `W`.
    # Slice off the conditioning days.
    Z = Epimap.conv(X_full, W)[:, num_cond:end - 1]

    # Compute `Z̃` for every time-step.
    # This is equivalent to
    #
    #   NNlib.batched_mul(F, reshape(Z, size(Z, 1), 1, size(Z, 2)))
    #
    # where we get
    #
    #   Z̃[:, k] := F[:, :, k] * Z[:, k]
    #
    # which is exactly what we want.
    Z̃ = NNlib.batched_vec(F, Z)

    # Compute the mean for the different regions at every time-step
    # HACK: seems like it can sometimes be negative due to numerical issues,
    # so we just `abs` to be certain. This is a bit hacky though.
    μ = R .* Z̃ .+ ξ

    # At this point `μ` will be of size `(num_regions, num_timesteps)`
    T = eltype(μ)
    X = X_full[:, (num_cond + 1):end]
    return sum(halfnormlogpdf.(μ, sqrt.((1 + ψ) .* μ), X))
end


@inline function logjoint_X_absnorm(F, X_full, W, R, ξ, ψ, num_cond)
    # Convolve `X` with `W`.
    # Slice off the conditioning days.
    Z = Epimap.conv(X_full, W)[:, num_cond:end - 1]

    # Compute `Z̃` for every time-step.
    # This is equivalent to
    #
    #   NNlib.batched_mul(F, reshape(Z, size(Z, 1), 1, size(Z, 2)))
    #
    # where we get
    #
    #   Z̃[:, k] := F[:, :, k] * Z[:, k]
    #
    # which is exactly what we want.
    Z̃ = NNlib.batched_vec(F, Z)

    # Compute the mean for the different regions at every time-step
    # HACK: seems like it can sometimes be negative due to numerical issues,
    # so we just `abs` to be certain. This is a bit hacky though.
    μ = map(abs, R .* Z̃ .+ ξ)

    # At this point `μ` will be of size `(num_regions, num_timesteps)`
    T = eltype(μ)
    X = X_full[:, (num_cond + 1):end]
    σ = sqrt.((1 + ψ) .* μ)
    return sum(StatsFuns.normlogpdf.((X - μ) / σ) .- log.(σ))
end


@inline function _loglikelihood(C, X, D, ϕ, weekly_case_variation, num_cond = 0)
    num_regions = size(C, 1)
    # Slice off the conditioning days
    # TODO: The convolution we're doing is for the PAST days, not current `t`, while
    # `conv` implements a convolution which involves the current day.
    # Instead maybe we should make `conv` use the "shifted" convolution, i.e. for all
    # PREVIOUS `t`.
    expected_positive_tests = Epimap.conv(X, D)[:, num_cond + 1:end]

    # Repeat one too many times and then extract the desired section `1:num_regions`
    num_days = size(expected_positive_tests, 2)
    weekly_case_variation = transpose(
        repeat(weekly_case_variation, (num_days ÷ 7) + 1)[1:num_days]
    )
    expected_positive_tests_weekly_adj = 7 * expected_positive_tests .* weekly_case_variation

    # We extract only the time-steps after the imputation-step
    T = eltype(expected_positive_tests_weekly_adj)
    return sum(nbinomlogpdf3.(
        expected_positive_tests_weekly_adj,
        ϕ,
        T.(C) # conversion ensures precision is preserved
    ))
end

@inline function rmap_loglikelihood(C, X, D, ϕ, weekly_case_variation, num_cond = 0)
    return _loglikelihood(C, X, D, ϕ, weekly_case_variation, num_cond)
end

function Epimap.make_logjoint(model::DynamicPPL.Model{DynamicPPLUtils.evaluatortype(rmap_naive)})
    # Construct an example `VarInfo`.
    vi = Turing.VarInfo(model)
    # Adapt parameters to use desired `eltype`.
    adaptor = Epimap.FloatMaybeAdaptor{eltype(model.args.D)}()
    θ = adapt(adaptor, ComponentArray(vi))
    # Construct the corresponding bijector.
    b_orig = TuringUtils.optimize_bijector_structure(
        Bijectors.bijector(vi; tuplify = true)
    )
    # Adapt bijector parameters to use desired `eltype`.
    b = fmap(b_orig) do x
        adapt(adaptor, x)
    end
    binv = inv(b)

    # Converter used for standard arrays.
    axis = first(ComponentArrays.getaxes(θ))
    nt(x) = Epimap.tonamedtuple(x, axis)

    function logjoint_unconstrained(args_unconstrained::AbstractVector)
        return logjoint_unconstrained(nt(args_unconstrained))
    end
    function logjoint_unconstrained(args_unconstrained::Union{NamedTuple, ComponentArray})
        args, logjac = forward(binv, args_unconstrained)
        return logjoint(args) + logjac
    end

    precomputed = Epimap.precompute(model)
    logjoint(args::AbstractVector) = logjoint(nt(args))
    function logjoint(args::Union{NamedTuple, ComponentArray})
        return DynamicPPL.logjoint(model, precomputed, args)
    end

    return (logjoint, logjoint_unconstrained, b, θ)
end

function Epimap.precompute(model::DynamicPPL.Model{DynamicPPLUtils.evaluatortype(rmap_naive)})
    C = model.args.C
    X_cond = model.args.X_cond

    num_regions = size(C, 1)
    num_infer = size(C, 2)
    num_cond = X_cond === nothing ? 0 : size(X_cond, 2)
    num_times = num_infer + num_cond

    # Ensures that we'll be using the same ordering as the original model.
    weekly_case_variation_reindex = map(1:7) do i
        (i + num_cond) % 7 + 1
    end

    precomputed = (; num_regions, num_times, num_cond, num_infer, weekly_case_variation_reindex)

    return precomputed
end

# To avoid ambiguity errors.
function DynamicPPL.logjoint(
    model::DynamicPPL.Model{DynamicPPLUtils.evaluatortype(rmap_naive)},
    vi::DynamicPPL.AbstractVarInfo
)
    model(vi, DynamicPPL.DefaultContext())
    return DynamicPPL.getlogp(vi)
end

function DynamicPPL.logjoint(
    model::DynamicPPL.Model{DynamicPPLUtils.evaluatortype(rmap_naive)},
    args
)
    return DynamicPPL.logjoint(model, Epimap.precompute(model), args)
end

function DynamicPPL.logjoint(
    model::DynamicPPL.Model{DynamicPPLUtils.evaluatortype(rmap_naive)},
    precomputed,
    args
)
    return DynamicPPL.logjoint(model, precomputed, args, Val{keys(args)}())
end

@generated function DynamicPPL.logjoint(
    model::DynamicPPL.Model{DynamicPPLUtils.evaluatortype(rmap_naive), outerkeys},
    precomputed,
    args,
    ::Val{innerkeys}
) where {innerkeys, outerkeys}
    outerexpr = Expr(:block)
    for k in outerkeys
        if !(k in innerkeys)
            push!(outerexpr.args, :($k = getproperty(model.args, $(QuoteNode(k)))))
        end
    end

    innerexpr = Expr(:block)
    for k in innerkeys
        push!(innerexpr.args, :($k = getproperty(args, $(QuoteNode(k)))))
    end

    return quote
        $outerexpr
        $innerexpr

        @unpack num_regions, num_times, num_cond, num_infer, weekly_case_variation_reindex = precomputed

        # Ensure that the univariates are treated as 0-dims
        Epimap.@map! first ψ μ_ar σ_ar α_pre ξ β σ_spatial σ_local ρ_spatial ρ_time σ_ξ

        X = if X isa AbstractVector
            # Need to reshape
            reshape(X, num_regions, :)
        else
            X
        end

        T = eltype(ψ) # TODO: Should probably find a better way to deal with this

        μ₀ = zero(T)
        σ₀ = T(5)

        # tack the conditioning X's back on to the samples
        X = X_cond === nothing ? X : hcat(X_cond, X)

        @assert num_infer % days_per_step == 0
        num_steps = num_infer ÷ days_per_step

        prev_infect_cutoff = length(W)
        test_delay_cutoff = length(D)

        # Noise for cases
        # ψ ~ 𝒩₊(0, 5)
        lp = halfnormlogpdf(μ₀, σ₀, ψ)
        # ϕ ~ filldist(𝒩₊(0, 5), num_regions)
        lp += sum(halfnormlogpdf.(μ₀, σ₀, ϕ))

        # Weekly case variation
        lp += logpdf(Turing.DistributionsAD.TuringDirichlet(7 * ones(T, 7)), weekly_case_variation)

        ### GP prior ###
        # Length scales
        # ρ_spatial ~ 𝒩₊(0, 5)
        lp += sum(halfnormlogpdf.(μ₀, σ₀, ρ_spatial))
        # ρ_time ~ 𝒩₊(0, 5)
        lp += sum(halfnormlogpdf.(μ₀, σ₀, ρ_time))

        # Scales
        # σ_spatial ~ 𝒩₊(0, 0.5)
        lp += sum(halfnormlogpdf.(μ₀, T(0.5), σ_spatial))
        # σ_local ~ 𝒩₊(0, 0.5)
        lp += sum(halfnormlogpdf.(μ₀, T(0.5), σ_local))

        # GP prior
        # E_vec ~ MvNormal(num_regions * num_times, 1.0)
        lp += sum(normlogpdf.(E_vec))
        E = reshape(E_vec, (num_regions, num_steps))

        # Get cholesky decomps using precomputed kernel matrices
        L_space = spatial_L(K_spatial, K_local, σ_spatial, σ_local, ρ_spatial)
        U_time = time_U(K_time, ρ_time)

        # Obtain the sample
        f = L_space * E * U_time
        # Repeat Rt to get Rt for every day in constant region
        R = exp.(f)

        # If we get an unreasonable value for `R`, we short-circuit.
        maximum(R) > 5 && return T(-Inf)

        ### Flux ###
        # Flux parameters
        # β ~ Uniform(0, 1)
        # HACK: don't add it since it's constant

        # AR(1) prior
        # set mean of process to be 0.1, 1 std = 0.024-0.33
        # μ_ar ~ Normal(-2.19, 0.25)
        lp += normlogpdf(T(-2.19), T(0.25), μ_ar)
        # σ_ar ~ 𝒩₊(0.0, 0.25)
        lp += halfnormlogpdf(T(0.0), T(0.25), σ_ar)

        # 28 likely refers to the number of days in a month, and so we're scaling the autocorrelation
        # wrt. number of days used in each time-step (specified by `days_per_step`).
        σ_α = 1 - exp(- days_per_step / T(28))
        # α_pre ~ transformed(Normal(0, σ_α), inv(Bijectors.Logit(0.0, 1.0)))
        b_α_pre = Bijectors.Logit(zero(T), one(T))
        α_pre_unconstrained, α_pre_logjac = forward(b_α_pre, α_pre)
        lp += normlogpdf(μ₀, σ_α, α_pre_unconstrained) + α_pre_logjac
        α = 1 - α_pre

        # Use bijector to transform to have support (0, 1) rather than ℝ.
        b_ρₜ = Bijectors.Logit{1}(zero(T), one(T))
        # ρₜ ~ transformed(AR1(num_times, α, μ_ar, σ_ar), inv(b_ρₜ))
        lp += logpdf(transformed(AR1(num_steps, α, μ_ar, σ_ar), inv(b_ρₜ)), ρₜ)

        # Global infection
        # σ_ξ ~ 𝒩₊(0, 5)
        lp += halfnormlogpdf.(μ₀, σ₀, σ_ξ)
        # ξ ~ 𝒩₊(0, σ_ξ)
        lp += halfnormlogpdf.(μ₀, σ_ξ, ξ)

        # for t = 2:num_times
        #     # Flux matrix
        #     Fₜ = @. ρₜ[t] * F_id + (1 - ρₜ[t]) * (β * F_out + (1 - β) * F_in) # Eq. (16)

        #     # Eq. (4) but we also add in the observed cases `C` at each time
        #     ts_prev_infect = reverse(max(1, t - prev_infect_cutoff):t - 1)
        #     Zₜ = X[:, ts_prev_infect] * W[1:min(prev_infect_cutoff, t - 1)]
        #     Z̃ₜ = Fₜ * Zₜ # Eq. (5)

        #     # Use continuous approximation
        #     μ = R[:, t] .* Z̃ₜ .+ ξ
        #     # # Eq. (15), though there they use `Zₜ` rather than `Z̃ₜ`; I suspect they meant `Z̃ₜ`.
        #     # for i = 1:num_regions
        #     #     X[i, t] ~ 𝒩₊(μ[i], sqrt((1 + ψ) * μ[i]))
        #     # end
        #     lp += sum(halfnormlogpdf.(μ, sqrt.((1 + ψ) .* μ), X[:, t], 0, Inf))
        # end
        # NOTE: This is the part which is the slowest.
        # Adds almost a second to the gradient computation for certain "standard" setups.
        F = compute_flux(F_id, F_in, F_out, β, ρₜ, days_per_step)
        F_expanded = F
        # Repeat F along time-dimension to get F for every day in constant region.
        # F_expanded = repeat(F, inner=(1, 1, days_per_step))
        lp += logjoint_X(F_expanded, X, W, R, ξ, ψ, num_cond, days_per_step)

        # for t = num_impute:num_times
        #     # Observe
        #     ts_prev_delay = reverse(max(1, t - test_delay_cutoff):t - 1)
        #     expected_positive_tests = X[:, ts_prev_delay] * D[1:min(test_delay_cutoff, t - 1)]

        #     # for i = 1:num_regions
        #     #     C[i, t] ~ NegativeBinomial3(expected_positive_tests[i], ϕ[i])
        #     # end
        # end
        lp += rmap_loglikelihood(
            C, X, D, ϕ,
            weekly_case_variation[weekly_case_variation_reindex],
            num_cond
        )

        return lp
    end
end

function MCMCChainsUtils.setconverters(
    chain::MCMCChains.Chains,
    model::DynamicPPL.Model{F}
) where {F<:Union{
    DynamicPPLUtils.evaluatortype(Rmap.rmap_naive),
    DynamicPPLUtils.evaluatortype(Rmap.rmap),
}}
    # In `Rmap.rmap_naive` `X` is a combination of the inferred latent infenctions and
    # `X_cond`, hence we need to replicate this structure. Here we add back the `X_cond`
    # though for usage in `fast_generated_quantities` and `fast_predict` we could just set
    # these to 0 as only the inferred variables are used.

    X_converter = let num_regions = size(model.args.X_cond, 1)
        X_chain -> begin
            num_iterations = length(X_chain)

            # Convert chain into an array.
            Xs = reshape(Array(X_chain), num_iterations, num_regions, :)
            return Xs
        end
    end

    return MCMCChainsUtils.setconverters(
        chain,
        X=X_converter
    );
end

function MCMCChainsUtils.setconverters(
    chain::MCMCChains.Chains,
    model::DynamicPPL.Model{DynamicPPLUtils.evaluatortype(Rmap.rmap_debiased)}
)
    # In `Rmap.rmap_naive` `X` is a combination of the inferred latent infenctions and
    # `X_cond`, hence we need to replicate this structure. Here we add back the `X_cond`
    # though for usage in `fast_generated_quantities` and `fast_predict` we could just set
    # these to 0 as only the inferred variables are used.

    X_converter = let num_regions = size(model.args.X_cond_means, 1)
        X_chain -> begin
            num_iterations = length(X_chain)

            # Convert chain into an array.
            Xs = reshape(Array(X_chain), num_iterations, num_regions, :)
            return Xs
        end
    end

    X_cond_converter = let num_regions = size(model.args.X_cond_means, 1)
        X_cond_chain -> begin
            num_iterations = length(X_cond_chain)

            # Convert chain into an array.
            Xs_cond = reshape(Array(X_cond_chain), num_iterations, num_regions, :)
            return Xs_cond
        end
    end

    return MCMCChainsUtils.setconverters(
        chain,
        X=X_converter,
        X_cond=X_cond_converter
    );
end
