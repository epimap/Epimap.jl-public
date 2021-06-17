using Epimap
using AdvancedHMC, Zygote
using DataFrames
using Dates
using Adapt

using Serialization, DrWatson, Dates

macro cleanbreak(ex)
    quote
        try
            $(esc(ex))
        catch e
            if e isa InterruptException
                @warn "Computation interrupted"
            else
                rethrow()
            end
        end
    end
end

const _intermediatedir = projectdir("intermediate", "$(Dates.now())")
intermediatedir() = _intermediatedir
intermediatedir(args...) = joinpath(intermediatedir(), args...)
mkpath(intermediatedir())

# Load data
const DATADIR = "file://" * joinpath(ENV["HOME"], "Projects", "private", "epimap-data", "processed_data")
data = Rmap.load_data(get(ENV, "EPIMAP_DATA", DATADIR));
T = Float32

# Construct the model arguments from data
setup_args = Rmap.setup_args(
    Rmap.rmap_naive, data, T;
    num_steps = 15,
    timestep = Week(1)
)

# Arguments not related to the data which are to be set up
default_args = (
    ρ_spatial = 10.0,
    ρ_time = 0.1,
    σ_spatial = 0.1,
    σ_local = 0.1,
    σ_ξ = 1.0
)

args = adapt(Epimap.FloatMaybeAdaptor{T}(), merge(setup_args, default_args))
serialize(intermediatedir("args.jls"), args)

# Instantiate model
m = Rmap.rmap_naive(args...);
logπ, logπ_unconstrained, b, θ_init = Epimap.make_logjoint(Rmap.rmap_naive, args..., Matrix{T});
const b⁻¹ = inv(b)

# Give it a try
logπ(θ_init)

# Get the unconstrained initial parameters
ϕ_init = b(θ_init)

# Give the gradient a try
Zygote.gradient(logπ_unconstrained, ϕ_init)

################
### SAMPLING ###
################
# Setup the sampler.
D = length(ϕ_init)
metric = DiagEuclideanMetric(T, D)
hamiltonian = Hamiltonian(metric, logπ_unconstrained, Zygote)

# Get initial parameters.
# ϕ_init = rand(T, D)

# Find a good step-size.
@info "Finding a good stepsize..."
initial_ϵ = find_good_stepsize(hamiltonian, ϕ_init)
@info "Found initial stepsize" initial_ϵ

# Construct integrator and trajectory.
integrator = Leapfrog(initial_ϵ)

τ = Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn(8, 1000.0))
κ = HMCKernel(τ)
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(T(0.8), integrator))

# Set-up for AbstractMCMC.
import AdvancedHMC: AbstractMCMC
rng = Turing.Random.MersenneTwister(42)

sampler = AdvancedHMC.HMCSampler(κ, metric, adaptor)
model = AdvancedHMC.DifferentiableDensityModel(hamiltonian.ℓπ, hamiltonian.∂ℓπ∂θ);

# Parameters
nadapts = 1_000;
nsamples = 1_000;

# Callback to use for progress-tracking.
cb = AdvancedHMC.HMCProgressCallback(nadapts + nsamples, progress = true, verbose = false)

# Create the iterator.
it = AbstractMCMC.steps(
    rng, model, sampler; 
    init_params = ϕ_init, 
    nadapts = nadapts
);

# Initial sample.
@info "Obtaining initial sample..."
transition, state = iterate(it);
@info "Initial sample" transition.stat

# Create the sample container.
samples = AbstractMCMC.samples(SimpleTransition(b⁻¹(transition.z.θ), transition.stat), model, sampler);

# [OPTIONAL] Keep track of some states for debugging purposes.
states = [state];

@info "Adapting!"
@cleanbreak for i = 1:nadapts
    global transition, state

    # Step
    transition, state = iterate(it, state)
    
    # Run callback
    cb(rng, model, sampler, transition, state, state.i; it.kwargs...)

    # Save some of the states just for fun
    if i % 200 == 0
        push!(states, state)
    end
    
    # Save sample
    # Transform `transition` back to original space.
    t = SimpleTransition(b⁻¹(transition.z.θ), transition.z.ℓπ.value, transition.stat)
    AbstractMCMC.save!!(samples, t, state.i, model, sampler, nsamples)
end

# Serialize
serialize(intermediatedir("state_adaptation.jls"), state)
serialize(intermediatedir("sampler_adaptation.jls"), sampler)
serialize(intermediatedir("kwargs_adaptation.jls"), it.kwargs)

# Sample!
@info "Sampling!"
@cleanbreak for i = 1:nsamples
    global transition, state

    # Step
    transition, state = iterate(it, state)
    
    # Run callback
    cb(rng, model, sampler, transition, state, state.i; it.kwargs...)

    # Save some of the states just for fun
    if i % 200 == 0
        push!(states, state)
    end
    
    # Save sample
    t = SimpleTransition(b⁻¹(transition.z.θ), transition.z.ℓπ.value, transition.stat)
    AbstractMCMC.save!!(samples, t, state.i, model, sampler, nsamples)
end

serialize(intermediatedir("state_last.jls"), state)
serialize(intermediatedir("sampler_last.jls"), sampler)
serialize(intermediatedir("chain.jls"), samples)
