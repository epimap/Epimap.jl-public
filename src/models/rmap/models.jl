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

function spatial_L(K_spatial_nonscaled, K_local, σ_spatial, σ_local)
    # Use `PDMats.ScalMat` to ensure that positive-definiteness is preserved
    K_spatial = ScalMat(size(K_spatial_nonscaled, 1), σ_spatial^2) * K_spatial_nonscaled
    K_local = ScalMat(size(K_local, 1), σ_local^2) * K_local

    K_space = PDMat(K_local + K_spatial) # `PDMat` is a no-op if the input is already a `PDMat`
    L_space = cholesky(K_space).L

    return L_space
end

function spatial_L(K_spatial_nonscaled, K_local, σ_spatial, σ_local, ρ_spatial)
    return spatial_L(K_spatial_nonscaled .^ inv.(ρ_spatial), K_local, σ_spatial, σ_local)
end

time_U(K_time) = cholesky(PDMat(K_time)).U
time_U(K_time, ρ_time) = time_U(K_time .^ inv.(ρ_time))

"""
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

## Notes
- Currently [2021-03-24 Wed] specifying `Matrix{Int}` won't work since we are now using
  a continuous variable to fill in the first values ("global" infection pressure `ξ`).
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


function Epimap.make_logjoint(
    ::typeof(rmap_naive),
    C, D, W,
    F_id, F_out, F_in,
    K_time, K_spatial, K_local,
    ρ_spatial = missing, ρ_time = missing,
    σ_spatial = missing, σ_local = missing,
    σ_ξ = missing,
    days_per_step = 1,
    ::Type{TV} = Matrix{T}
) where {T<:Real, TV}
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

        # TODO: move the computation of `Z̃ₜ` into a function, so we can define a custom adjoint for it,
        # to allow Zygote.jl/reverse-mode AD compatibility.
        X = TV(undef, (num_regions, num_times))

        X[:, 1] .= 0

        for t = 2:num_times
            # Flux matrix
            Fₜ = @. ρₜ[t] * F_id + (1 - ρₜ[t]) * (β * F_out + (1 - β) * F_in) # Eq. (16)

            # Eq. (4) but we also add in the observed cases `C` at each time
            ts_prev_infect = reverse(max(1, t - prev_infect_cutoff):t - 1)
            Zₜ = (X[:, ts_prev_infect] + C[:, ts_prev_infect]) * W[1:min(prev_infect_cutoff, t - 1)]
            Z̃ₜ = Fₜ * Zₜ # Eq. (5)

            # Use continuous approximation
            μ = R[:, t] .* Z̃ₜ .+ ξ
            # # Eq. (15), though there they use `Zₜ` rather than `Z̃ₜ`; I suspect they meant `Z̃ₜ`.
            # for i = 1:num_regions
            #     X[i, t] ~ 𝒩₊(μ[i], sqrt((1 + ψ) * μ[i]))
            # end
            lp += truncatednormlogpdf.(μ, sqrt.((1 + ψ) .* μ), X[:, t])

            # Observe
            ts_prev_delay = reverse(max(1, t - test_delay_cutoff):t - 1)
            expected_positive_tests = X[:, ts_prev_delay] * D[1:min(test_delay_cutoff, t - 1)]

            # for i = 1:num_regions
            #     C[i, t] ~ NegativeBinomial3(expected_positive_tests[i], ϕ[i])
            # end
            lp += loglikelihood(arraydist(NegativeBinomial3.(expected_positive_tests, ϕ)), C[:, t])
        end

        return lp
    end
end
