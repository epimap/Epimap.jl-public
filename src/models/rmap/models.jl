import StatsFuns: normlogpdf

function truncatednormlogpdf(μ, σ, x, lb, ub)
    logtp = StatsFuns.normlogcdf(μ, σ, ub) - StatsFuns.normlogcdf(μ, σ, lb)
    # TODO: deal with outside of boundary
    StatsFuns.normlogpdf(μ, σ, x) - logtp
    # TODO: seems like there's something messed up with the way we return `Inf`
    # if lb <= x <= ub
    #     StatsFuns.normlogpdf(μ, σ, x) - logtp
    # else
    #     TF = float(eltype(x))
    #     -TF(Inf)
    # end
end



### Convenience methods ###
𝒩₊(μ, σ) = truncated(Normal(μ, σ), 1e-6, Inf)

PDMats.PDMat(P::PDMat) = P

function spatial_L(K_spatial_nonscaled, K_local, σ_spatial, σ_local)
    # Use `PDMats.ScalMat` to ensure that positive-definiteness is preserved
    # K_spatial = ScalMat(size(K_spatial_nonscaled, 1), σ_spatial^2) * K_spatial_nonscaled
    # K_local = ScalMat(size(K_local, 1), σ_local^2) * K_local
    # HACK: use this until we have an adjoint for `ScalMat` constructor in ChainRulesCore.jl
    K_spatial = PDMat(σ_spatial^2 .* K_spatial_nonscaled)
    K_local = PDMat(σ_local^2 .* K_local)


    K_space = PDMat(K_local + K_spatial) # `PDMat` is a no-op if the input is already a `PDMat`
    L_space = cholesky(K_space).L

    return L_space
end

function spatial_L(K_spatial_nonscaled, K_local, σ_spatial, σ_local, ρ_spatial)
    return spatial_L(K_spatial_nonscaled .^ inv.(ρ_spatial), K_local, σ_spatial, σ_local)
end

time_U(K_time) = cholesky(PDMat(K_time)).U
time_U(K_time, ρ_time) = time_U(K_time .^ inv.(ρ_time))

@doc raw"""
    rmap_naive(args...)

Naive implementation of full Rmap model.

## Arguments
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
- `ρ_spatial = missing`: length scale / "stretch" applied to `K_spatial`.
- `ρ_time = missing`: length scale / "stretch" applied to `K_time`.
- `σ_spatial = missing`: scale applied to `K_spatial`.
- `σ_local = missing`: scale applied to `K_local`.
- `σ_ξ = missing`: square root of variance of the "global" infection pressure `ξ`.
- `days_per_step = 1`: specifies how many days of data each time step corresponds to.

Note that those with default value `missing` will be sampled if not specified.

## Mathematical description
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
  F_{t} & := \rho_t F_{\mathrm{id}} + (1 - \rho_t) \big(\beta F_{\mathrm{fwd}} + (1 - \beta) F_{\mathrm{rev}} \big) & \quad \forall t = 1, \dots, T \\
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
    ρ_spatial = missing, ρ_time = missing,
    σ_spatial = missing, σ_local = missing,
    σ_ξ = missing,
    num_impute = 10,
    days_per_step = 1,
    ::Type{TV} = Matrix{Float64}
) where {TV}
    num_regions = size(C, 1)
    num_times = size(C, 2)

    prev_infect_cutoff = length(W)
    test_delay_cutoff = length(D)

    # Noise for cases
    ψ ~ 𝒩₊(0, 5)
    ϕ ~ filldist(𝒩₊(0, 5), num_regions)

    ### GP prior ###
    # Length scales
    ρ_spatial ~ 𝒩₊(0, 5)
    ρ_time ~ 𝒩₊(0, 5)

    # Scales
    σ_spatial ~ 𝒩₊(0, 5)
    σ_local ~ 𝒩₊(0, 5)

    # GP prior
    E_vec ~ MvNormal(num_regions * num_times, 1.0)
    E = reshape(E_vec, (num_regions, num_times))

    # Get cholesky decomps using precomputed kernel matrices
    L_space = spatial_L(K_spatial, K_local, σ_spatial, σ_local, ρ_spatial)
    U_time = time_U(K_time, ρ_time)

    # Obtain the sample
    f = L_space * E * U_time
    R = exp.(f)

    ### Flux ###
    # Flux parameters
    β ~ Uniform(0, 1)

    # AR(1) prior
    # set mean of process to be 0.1, 1 std = 0.024-0.33
    μ_ar ~ Normal(-2.19, 0.25)
    σ_ar ~ 𝒩₊(0.0, 0.25)

    # 28 likely refers to the number of days in a month, and so we're scaling the autocorrelation
    # wrt. number of days used in each time-step (specified by `days_per_step`).
    σ_α = 1 - exp(- days_per_step / 28)
    α_pre ~ transformed(Normal(0, σ_α), inv(Bijectors.Logit(0.0, 1.0)))
    α = 1 - α_pre

    # Use bijector to transform to have support (0, 1) rather than ℝ.
    b = Bijectors.Logit{1, Float64}(0.0, 1.0)
    ρₜ ~ transformed(AR1(num_times, α, μ_ar, σ_ar), inv(b))

    # Global infection
    σ_ξ ~ 𝒩₊(0, 5)
    ξ ~ 𝒩₊(0, σ_ξ)

    # TODO: move the computation of `Z̃ₜ` into a function, so we can define a custom adjoint for it,
    # to allow Zygote.jl/reverse-mode AD compatibility.
    X = TV(undef, (num_regions, num_times))

    X[:, 1] .= 0

    for t = 2:num_times
        # Flux matrix
        Fₜ = @. ρₜ[t] * F_id + (1 - ρₜ[t]) * (β * F_out + (1 - β) * F_in) # Eq. (16)

        # Eq. (4) but we also add in the observed cases `C` at each time
        ts_prev_infect = reverse(max(1, t - prev_infect_cutoff):t - 1)
        Zₜ = X[:, ts_prev_infect] * W[1:min(prev_infect_cutoff, t - 1)]
        Z̃ₜ = Fₜ * Zₜ # Eq. (5)

        # Use continuous approximation if the element type of `X` is non-integer.
        μ = R[:, t] .* Z̃ₜ .+ ξ
        if eltype(X) <: Integer
            for i = 1:num_regions
                X[i, t] ~ NegativeBinomial3(μ[i], ψ)
            end
        else
            # Eq. (15), though there they use `Zₜ` rather than `Z̃ₜ`; I suspect they meant `Z̃ₜ`.
            for i = 1:num_regions
                X[i, t] ~ 𝒩₊(μ[i], sqrt((1 + ψ) * μ[i]))
            end
        end
    end

    # Observe (if we're done imputing)
    for t = num_impute:num_times
        ts_prev_delay = reverse(max(1, t - test_delay_cutoff):t - 1)
        expected_positive_tests = X[:, ts_prev_delay] * D[1:min(test_delay_cutoff, t - 1)]


        for i = 1:num_regions
            C[i, t] ~ NegativeBinomial3(expected_positive_tests[i], ϕ[i])
        end
    end

    return (R = R, X = X)
end

@inline function logjoint_X(F_id, F_in, F_out, β, ρₜ, X, W, R, ξ, ψ)
    # Compute the full flux
    F_cross = @. β * F_out + (1 - β) * F_in
    # oneminusρₜ = @. 1 - ρₜ
    # kron(1 .- ρₜ', F_cross)
    # F = @tensor begin
    #     F[i, j, t] := ρₜ[t] * F_id[i, j] + oneminusρₜ[t] * F_cross[i, j]
    # end

    # Equivalent to the above `@tensor`
    res1 = kron(1 .- ρₜ', F_cross)
    res2 = kron(ρₜ', F_id)
    F = reshape(res2 + res1, size(F_cross)..., length(ρₜ))


    # Convolve `X` with `W`
    Z = Epimap.conv(X, W)

    # Compute `Z̃` for every time-step
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
    μ = R .* Z̃ .+ ξ

    # At this point `μ` will be of size `(num_regions, num_timesteps)`
    return sum(truncatednormlogpdf.(μ, sqrt.((1 + ψ) .* μ), X, 0, Inf))
end


@inline function _loglikelihood(C, X, D, ϕ, num_impute = 1)
    # Deal with potential numerical issues
    expected_positive_tests = clamp.(Epimap.conv(X, D), 0, Inf)
    # TODO: implement vectorized version of `NegativeBinomial`
    # We extract only the time-steps after the imputation-step
    return loglikelihood(
        arraydist(NegativeBinomial3.(expected_positive_tests[:, num_impute:end], ϕ)),
        C[:, num_impute:end]
    )
end


function Epimap.make_logjoint(
    ::typeof(rmap_naive),
    C, D, W,
    F_id, F_out, F_in,
    K_time, K_spatial, K_local,
    ρ_spatial = missing, ρ_time = missing,
    σ_spatial = missing, σ_local = missing,
    σ_ξ = missing,
    num_impute = 10,
    days_per_step = 1,
    ::Type{TV} = Matrix{Float64},
    ::Type{T} = Float64
) where {T, TV}
    function logjoint(args)
        @unpack ψ, ϕ, E_vec, β, μ_ar, σ_ar, α_pre, ρₜ, ξ, X = args

        lp = zero(T)

        num_regions = size(C, 1)
        num_times = size(C, 2)

        prev_infect_cutoff = length(W)
        test_delay_cutoff = length(D)

        # Noise for cases
        # ψ ~ 𝒩₊(0, 5)
        lp = truncatednormlogpdf(0, 5, ψ, 0, Inf)
        # ϕ ~ filldist(𝒩₊(0, 5), num_regions)
        lp += sum(truncatednormlogpdf.(0, 5, ϕ, 0, Inf))

        ### GP prior ###
        # Length scales
        # ρ_spatial ~ 𝒩₊(0, 5)
        lp += sum(truncatednormlogpdf.(0, 5, ρ_spatial, 0, Inf))
        # ρ_time ~ 𝒩₊(0, 5)
        lp += sum(truncatednormlogpdf.(0, 5, ρ_time, 0, Inf))

        # Scales
        # σ_spatial ~ 𝒩₊(0, 5)
        lp += sum(truncatednormlogpdf.(0, 5, σ_spatial, 0, Inf))
        # σ_local ~ 𝒩₊(0, 5)
        lp += sum(truncatednormlogpdf.(0, 5, σ_local, 0, Inf))

        # GP prior
        # E_vec ~ MvNormal(num_regions * num_times, 1.0)
        lp += sum(normlogpdf.(E_vec))
        E = reshape(E_vec, (num_regions, num_times))

        # Get cholesky decomps using precomputed kernel matrices
        L_space = spatial_L(K_spatial, K_local, σ_spatial, σ_local, ρ_spatial)
        U_time = time_U(K_time, ρ_time)

        # Obtain the sample
        f = L_space * E * U_time
        R = exp.(f)

        ### Flux ###
        # Flux parameters
        # β ~ Uniform(0, 1)
        # HACK: don't add it since it's constant

        # AR(1) prior
        # set mean of process to be 0.1, 1 std = 0.024-0.33
        # μ_ar ~ Normal(-2.19, 0.25)
        lp += normlogpdf(-2.19, 0.25, μ_ar)
        # σ_ar ~ 𝒩₊(0.0, 0.25)
        lp += normlogpdf(0.0, 0.25, σ_ar)

        # 28 likely refers to the number of days in a month, and so we're scaling the autocorrelation
        # wrt. number of days used in each time-step (specified by `days_per_step`).
        σ_α = 1 - exp(- days_per_step / 28)
        # α_pre ~ transformed(Normal(0, σ_α), inv(Bijectors.Logit(0.0, 1.0)))
        b_α_pre = inv(Bijectors.Logit(0.0, 1.0))
        lp += normlogpdf(b_α_pre(α_pre)) + logabsdetjac(b_α_pre, α_pre)
        α = 1 - α_pre

        # Use bijector to transform to have support (0, 1) rather than ℝ.
        b_ρₜ = Bijectors.Logit{1, Float64}(0.0, 1.0)
        # ρₜ ~ transformed(AR1(num_times, α, μ_ar, σ_ar), inv(b_ρₜ))
        lp += logpdf(transformed(AR1(num_times, α, μ_ar, σ_ar), inv(b_ρₜ)), ρₜ)

        # Global infection
        # σ_ξ ~ 𝒩₊(0, 5)
        lp += truncatednormlogpdf.(0, 5, σ_ξ, 0, Inf)
        # ξ ~ 𝒩₊(0, σ_ξ)
        lp += truncatednormlogpdf.(0, σ_ξ, ξ, 0, Inf)

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
        #     lp += sum(truncatednormlogpdf.(μ, sqrt.((1 + ψ) .* μ), X[:, t], 0, Inf))
        # end

        lp += logjoint_X(F_id, F_in, F_out, β, ρₜ, X, W, R, ξ, ψ)

        # for t = num_impute:num_times
        #     # Observe
        #     ts_prev_delay = reverse(max(1, t - test_delay_cutoff):t - 1)
        #     expected_positive_tests = X[:, ts_prev_delay] * D[1:min(test_delay_cutoff, t - 1)]

        #     # for i = 1:num_regions
        #     #     C[i, t] ~ NegativeBinomial3(expected_positive_tests[i], ϕ[i])
        #     # end
        # end
        lp += _loglikelihood(C, X, D, ϕ, num_impute)

        return lp
    end
end
