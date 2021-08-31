# NOTE: This only sets up the benchmark suite; it does NOT run it!
# To run it, open a Julia repl and include this file (using `include`), and then
# follow up with `run(suite, verbose=True, seconds=60)` or whatever options you want.

using ArgParse

s = ArgParseSettings()
@add_arg_table! s begin
    "model"
    help = "the model to run benchmarks for"
    required = true
    "--type"
    help = "the element type to use"
    default = Float64
    eval_arg = true
end

if !(@isdefined(_args))
    _args = ARGS
end

parsed_args = parse_args(_args, s)

# Done with argument parsing.
using BenchmarkTools
using Epimap, Adapt, Zygote, Dates, Random, ComponentArrays
using Epimap.Turing

# HACK: Using `@threads` with `SimpleVarInfo` isn't going to work.
function DynamicPPL.evaluate(model::DynamicPPL.Model, varinfo::DynamicPPL.AbstractVarInfo, context::DynamicPPL.AbstractContext)
    return DynamicPPL.evaluate_threadunsafe(model, varinfo, context)
end

modeldef = eval(Meta.parse(parsed_args["model"]))
T = parsed_args["type"]

# Useful for benchmarks
function pb(f, ::Type{T}, args...) where {T}
    y, ȳ = Zygote.pullback(f, args...)
    return y, ȳ(one(T))
end

# Load data
data = Rmap.load_data();

area_names_original = data.areas.area;
area_names_debiased = data.debiased.ltla;
area_names = intersect(area_names_original, area_names_debiased);

data = Rmap.filter_areas_by_distance(data, area_names; radius=1e-6);

# Construct the model arguments from data
args, dates = Rmap.setup_args(
    modeldef, data, T;
    num_steps = 15,
    timestep = Week(1),
    include_dates = true,
    last_date = Date(2021, 02, 07)
);


# Instantiate model
m = modeldef(
    args...;
    ρ_spatial = T(0.1), ρ_time = T(100.0), σ_ξ = T(0.1)
);

# Get the necessary functions, etc.
logπ, logπ_unconstrained, b, θ = Epimap.make_logjoint(m);
ϕ = inv(b)(θ);

logπ(θ)
Zygote.gradient(logπ_unconstrained, ϕ)

@benchmark $(Zygote.gradient)($logπ_unconstrained, $ϕ)

# # Construct benchmark-suite
# suite = BenchmarkGroup()

# suite["logjoint"] = BenchmarkGroup()
# suite["logjoint_unconstrained"] = BenchmarkGroup()

# suite["logjoint"]["evaluation"] = BenchmarkGroup()
# suite["logjoint"]["gradient"] = BenchmarkGroup()

# suite["logjoint_unconstrained"]["evaluation"] = BenchmarkGroup()
# suite["logjoint_unconstrained"]["gradient"] = BenchmarkGroup()

# for T ∈ [Float32, Float64]
#     logπ, logπ_unconstrained, b, θ = Epimap.make_logjoint(Rmap.rmap_naive, setup_args...)
#     ϕ = inv(b)(θ)

#     # HACK: Execute once to compile (does this even matter? I thought BenchmarkTools would take care of this.)
#     pb(logπ, T, θ)
#     pb(logπ_unconstrained, T, ϕ)

#     suite["logjoint"]["evaluation"]["$T"] = @benchmarkable $logπ($θ)
#     suite["logjoint"]["gradient"]["$T"] = @benchmarkable $(pb)($logπ, $T, $θ)

#     suite["logjoint_unconstrained"]["evaluation"]["$T"] = @benchmarkable $logπ_unconstrained($ϕ)
#     suite["logjoint_unconstrained"]["gradient"]["$T"] = @benchmarkable $(pb)($logπ_unconstrained, $T, $ϕ)
# end

# suite
