using Epimap
using AdvancedHMC, Zygote
using DataFrames
using Dates
using Adapt

# Load data
const DATADIR = "file://" * joinpath(ENV["HOME"], "Projects", "private", "epimap-data", "processed_data")
data = Rmap.load_data(DATADIR);
T = Float64

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

# Instantiate model
m = Rmap.rmap_naive(args...);
logπ, logπ_unconstrained, b, θ_init = Epimap.make_logjoint(Rmap.rmap_naive, args..., Matrix{T});

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

τ = Trajectory{MultinomialTS}(integrator, AdvancedHMC.FixedNSteps(10))
κ = HMCKernel(τ)
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(T(0.8), integrator))

# Set-up for AbstractMCMC.
import AdvancedHMC: AbstractMCMC
rng = Turing.Random.MersenneTwister(42)

sampler = AdvancedHMC.HMCSampler(κ, metric, adaptor)
model = AdvancedHMC.DifferentiableDensityModel(hamiltonian.ℓπ, hamiltonian.∂ℓπ∂θ);

# Parameters
nadapts = 1_000;
nsamples = 3_000;

# Callback to use for progress-tracking.
cb = AdvancedHMC.HMCProgressCallback(nsamples, progress = true, verbose = false)

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
samples = AbstractMCMC.samples(transition, model, sampler);

# [OPTIONAL] Keep track of some states for debugging purposes.
states = [state];

# Sample!
@info "Sampling!"
for i = 1:nsamples
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
    AbstractMCMC.save!!(samples, transition, state.i, model, sampler, nsamples)
end
