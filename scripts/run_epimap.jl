using Epimap
using AdvancedHMC, Zygote
using DataFrames
using Dates
using Adapt
using TuringCallbacks

using DynamicPPL, Random, ComponentArrays
Random.seed!(1)

using Serialization, DrWatson, Dates

include(scriptsdir("utils.jl"))

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

const _gitname = default_name(Epimap; include_commit_id=true)
const _intermediatedir = projectdir("intermediate", "$(Dates.now())-$(_gitname)")
intermediatedir() = _intermediatedir
intermediatedir(args...) = joinpath(intermediatedir(), args...)
mkpath(intermediatedir())
@info "Output of run is found in $(intermediatedir())"

# Load data
# const DATADIR = "file://" * joinpath(ENV["HOME"], "Projects", "private", "epimap-data", "processed_data")
const DATADIR = "file://" * joinpath(ENV["HOME"], "Projects", "private", "Rmap", "data")
data = Rmap.load_data(get(ENV, "EPIMAP_DATA", DATADIR));

# Filter out areas for which we don'do not have unbiased estimates for.
area_names_original = data.areas.area;
area_names_debiased = data.debiased.ltla;
area_names = intersect(area_names_original, area_names_debiased);

data = Rmap.filter_areas_by_distance(data, area_names; radius=1e-6);

area_names = data.areas[:, :area]
@info "Doing inference for $(length(area_names)) regions."

const T = Float64
const model_def = Rmap.rmap_debiased

# Construct the model arguments from data
args, dates = Rmap.setup_args(
    model_def, data, T;
    num_steps = 15,
    timestep = Week(1),
    include_dates = true,
    last_date = Date(2021, 02, 07)
)

# With `area_names` and `dates` we can recover the data being used.
serialize(intermediatedir("area_names.jls"), area_names)
serialize(intermediatedir("dates.jls"), dates)

# Instantiate model
m = model_def(
    args...;
    ρ_spatial = T(0.1), ρ_time = T(100.0), σ_ξ = T(0.1),
    # ρₜ = ones(T, 15), β = 0.0,
)
serialize(intermediatedir("args.jls"), m.args)
serialize(intermediatedir("model.jls"), m)

logπ, logπ_unconstrained, b, θ_init = Epimap.make_logjoint(m);
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
D = length(ϕ_init);
metric = DiagEuclideanMetric(T, D);
hamiltonian = Hamiltonian(metric, logπ_unconstrained, Zygote);

# Get initial parameters.
# ϕ_init = rand(T, D)

# Find a good step-size.
@info "Finding a good stepsize..."
initial_ϵ = find_good_stepsize(hamiltonian, ϕ_init);
@info "Found initial stepsize" initial_ϵ

# Construct integrator and trajectory.
integrator = Leapfrog(initial_ϵ);

τ = Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn(8, 1000.0));
κ = HMCKernel(τ);
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(T(0.8), integrator));

# Set-up for AbstractMCMC.
import AdvancedHMC: AbstractMCMC
rng = Turing.Random.MersenneTwister(43);

sampler = AdvancedHMC.HMCSampler(κ, metric, adaptor);
model = AdvancedHMC.DifferentiableDensityModel(hamiltonian.ℓπ, hamiltonian.∂ℓπ∂θ);

# Parameters
nadapts = 5_00;
nsamples = 5_00;

# Callback to use for progress-tracking.
cb1 = AdvancedHMC.HMCProgressCallback(nadapts + nsamples, progress = true, verbose = false)

using TuringCallbacks.OnlineStats
stat = Series(
    # Estimators using the entire chain
    Series(Mean(), Variance(), AutoCov(10), KHist(100)),
    # Estimators using only the last 1000 samples
    WindowStat(100, Series(Mean(), Variance(), AutoCov(10), KHist(100)))
)
cb2 = TensorBoardCallback(
    "tensorboard_logs/$(_gitname)",
    stat,
    include = ["ψ", "ξ", "σ_spatial", "σ_local"]
)

# HACK: Super-hacky impl. Should improve `TuringCallbacks` to be more flexible instead.
function Turing.Inference._params_to_array(ts::Vector{<:SimpleTransition})
    @assert length(ts) == 1
    # Convert to `Float64` because otherwise TuringCallbacks won't be happy.
    nt = NamedTuple(Float64.(ts[1].θ))
    return keys(nt), map(values(nt)) do x
        return if length(x) == 1
            first(x)
        else
            x
        end
    end
end

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
samples = let
    θ, logjac = forward(b⁻¹, transition.z.θ)
    AbstractMCMC.samples(SimpleTransition(θ, transition.z.ℓπ.value + logjac, transition.stat), model, sampler);
end

# [OPTIONAL] Keep track of some states for debugging purposes.
states = [state];

@info "Adapting!"
@cleanbreak for i = 1:nadapts
    global transition, state

    # Step
    transition, state = iterate(it, state)
    
    # Run callback
    cb1(rng, model, sampler, transition, state, state.i; it.kwargs...)

    # Save some of the states just for fun
    if i % 200 == 0
        push!(states, state)
    end
    
    # Save sample
    # Transform `transition` back to original space.
    t = let
        θ, logjac = forward(b⁻¹, transition.z.θ)
        SimpleTransition(θ, transition.z.ℓπ.value + logjac, transition.stat)
    end
    cb2(rng, model, sampler, t, state, state.i; it.kwargs...)
    AbstractMCMC.save!!(samples, t, state.i, model, sampler)
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
    cb1(rng, model, sampler, transition, state, state.i; it.kwargs...)

    # Save some of the states just for fun
    if i % 200 == 0
        push!(states, state)
    end
    
    # Save sample
    t = let
        θ, logjac = forward(b⁻¹, transition.z.θ)
        SimpleTransition(θ, transition.z.ℓπ.value + logjac, transition.stat)
    end
    cb2(rng, model, sampler, t, state, state.i; it.kwargs...)
    AbstractMCMC.save!!(samples, t, state.i, model, sampler)
end

serialize(intermediatedir("state_last.jls"), state)
serialize(intermediatedir("sampler_last.jls"), sampler)
serialize(intermediatedir("chain.jls"), samples)
