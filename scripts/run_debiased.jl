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
# NOTE: This is also currently done in `Rmap.setup_args` but we want to
# save the `area_names` and so we just perform the filtering here too.
area_names_rmap = unique(data.areas.area);
area_names_debiased = unique(data.debiased.ltla);
dest2sources = Dict(
    "North Northamptonshire" => ["Kettering", "Corby", "Wellingborough", "East Northamptonshire"],
    "West Northamptonshire" => ["Daventry", "Northampton", "South Northamptonshire"],
    "Cornwall" => ["Cornwall and Isles of Scilly"],
    "Hackney" => ["Hackney and City of London"]
)
area_names_latent, area_names_observed, P = Rmap.make_projection(area_names_rmap, area_names_debiased, dest2sources)
area_names = area_names_latent

data = Rmap.filter_areas_by_distance(data, area_names; radius=1e-6, filter_debiased=false);
@info "Doing inference for $(length(area_names)) regions."

T = Float64
model_def = Rmap.rmap_debiased

# Construct the model arguments from data.
# We only skip weeks if we're working with `rmap_debiased`. Otherwise
# we use the deterministic `X_cond`.
# TODO: Make this part of `setup_args`?
skip_weeks_observe = model_def === Rmap.rmap_debiased ? 3 : 0
args, dates = Rmap.setup_args(
    model_def, data, T;
    num_steps = 15 + skip_weeks_observe,
    timestep = Week(1),
    include_dates = true,
    last_date = Date(2021, 02, 03)
)

kwargs = (??_spatial = T(0.1), ??_time = T(100.0), ??_?? = T(0.1))
kwargs = if model_def === Rmap.rmap_debiased
    merge(kwargs, (skip_weeks_observe=skip_weeks_observe, ))
else
    kwargs
end

# With `area_names` and `dates` we can recover the data being used.
serialize(intermediatedir("area_names.jls"), area_names)
serialize(intermediatedir("area_names_latent.jls"), area_names_latent)
serialize(intermediatedir("area_names_observed.jls"), area_names_observed)
serialize(intermediatedir("dates.jls"), dates)

# Instantiate model
m = if model_def === Rmap.rmap_debiased
    model_def(
        args[1][:, skip_weeks_observe + 1:end], args[2][:, skip_weeks_observe + 1:end],
        Iterators.drop(args, 2)...;
        kwargs...,
        # ????? = ones(T, 15), ?? = 0.0,
    );
else
    model_def(args...; kwargs...)
end
serialize(intermediatedir("args.jls"), m.args)
serialize(intermediatedir("model.jls"), m)

log??, log??_unconstrained, b, ??_init = Epimap.make_logjoint(m);
const b????? = inv(b)

# Give it a try
log??(??_init)

# Get the unconstrained initial parameters
??_init = b(??_init)

# Give the gradient a try
Zygote.gradient(log??_unconstrained, ??_init)

################
### SAMPLING ###
################
# Setup the sampler.
D = length(??_init);
metric = DiagEuclideanMetric(T, D);
hamiltonian = Hamiltonian(metric, log??_unconstrained, Zygote);

# Get initial parameters.
# ??_init = rand(T, D)

# Find a good step-size.
@info "Finding a good stepsize..."
initial_?? = find_good_stepsize(hamiltonian, ??_init);
@info "Found initial stepsize" initial_??

# Construct integrator and trajectory.
integrator = Leapfrog(initial_??);

?? = Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn(8, 1000.0));
?? = HMCKernel(??);
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(T(0.8), integrator));

# Set-up for AbstractMCMC.
import AdvancedHMC: AbstractMCMC
rng = Turing.Random.GLOBAL_RNG;

sampler = AdvancedHMC.HMCSampler(??, metric, adaptor);
model = AdvancedHMC.DifferentiableDensityModel(hamiltonian.?????, hamiltonian.?????????????);

# Parameters
nadapts = 1_000;
nsamples = 1_000;

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
    include = ["??", "??", "??_spatial", "??_local"]
)

# HACK: Super-hacky impl. Should improve `TuringCallbacks` to be more flexible instead.
function Turing.Inference._params_to_array(ts::Vector{<:SimpleTransition})
    @assert length(ts) == 1
    # Convert to `Float64` because otherwise TuringCallbacks won't be happy.
    nt = NamedTuple(Float64.(ts[1].??))
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
    init_params = ??_init, 
    nadapts = nadapts
);

# Initial sample.
@info "Obtaining initial sample..."
transition, state = iterate(it);
@info "Initial sample" transition.stat

# Create the sample container.
samples = let
    ??, logjac = forward(b?????, transition.z.??)
    AbstractMCMC.samples(SimpleTransition(??, transition.z.?????.value + logjac, transition.stat), model, sampler);
end

# [OPTIONAL] Keep track of some states for debugging purposes.
states = [state];

@info "Adapting!"
@cleanbreak while state.i < nadapts
    global transition, state

    # Step
    transition, state = iterate(it, state)
    
    # Run callback
    cb1(rng, model, sampler, transition, state, state.i; it.kwargs...)

    # Save some of the states just for fun
    if state.i % 200 == 0
        push!(states, state)
    end
    
    # Save sample
    # Transform `transition` back to original space.
    t = let
        ??, logjac = forward(b?????, transition.z.??)
        SimpleTransition(??, transition.z.?????.value + logjac, transition.stat)
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
@cleanbreak while state.i < nadapts + nsamples
    global transition, state

    # Step
    transition, state = iterate(it, state)
    
    # Run callback
    cb1(rng, model, sampler, transition, state, state.i; it.kwargs...)

    # Save some of the states just for fun
    if state.i % 200 == 0
        push!(states, state)
    end
    
    # Save sample
    t = let
        ??, logjac = forward(b?????, transition.z.??)
        SimpleTransition(??, transition.z.?????.value + logjac, transition.stat)
    end
    cb2(rng, model, sampler, t, state, state.i; it.kwargs...)
    AbstractMCMC.save!!(samples, t, state.i, model, sampler)
end

serialize(intermediatedir("state_last.jls"), state)
serialize(intermediatedir("sampler_last.jls"), sampler)
serialize(intermediatedir("chain.jls"), samples)
