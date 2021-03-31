# NOTE: This only sets up the benchmark suite; it does NOT run it!
# To run it, open a Julia repl and include this file (using `include`), and then
# follow up with `run(suite, verbose=True, seconds=60)` or whatever options you want.

using BenchmarkTools

using Epimap, Adapt, Zygote
using Epimap.Turing

# Useful for benchmarks
function pb(f, ::Type{T}, args...) where {T}
    y, ȳ = Zygote.pullback(f, args...)
    return y, ȳ(one(T))
end

# Load data
data = Rmap.load_data();

# Arguments not related to the data which are to be set up
default_args = (
    ρ_spatial = 10.0,
    ρ_time = 0.1,
    σ_spatial = 0.1,
    σ_local = 0.1,
    σ_ξ = 1.0,
)

# Construct the model arguments from data
setup_args = merge(Rmap.setup_args(Rmap.rmap_naive, data, days_per_step=4, num_cond=60), default_args);

# Instantiate model
m = Rmap.rmap_naive(setup_args...);

# HACK: using Turing to get a sample from the prior
var_info = DynamicPPL.VarInfo(m);

num_regions = size(data.cases, 1);
nt = map(DynamicPPL.tonamedtuple(var_info)) do (v, ks)
    if startswith(string(first(ks)), "X")
        # Add back in the first column since it's not inferred
        reshape(v, (num_regions, :))
    elseif length(v) == 1
        first(v)
    else
        v
    end
end;

# Construct benchmark-suite
suite = BenchmarkGroup()

suite["evaluation"] = BenchmarkGroup()
suite["gradient"] = BenchmarkGroup()
for T ∈ [Float32, Float64]
    let adaptor = Epimap.FloatMaybeAdaptor{T}(), nt = adapt(adaptor, nt), setup_args = adapt(adaptor, setup_args)
        logπ = Epimap.make_logjoint(Rmap.rmap_naive, setup_args...)

        # HACK: Execute once to compile (does this even matter? I thought BenchmarkTools would take care of this.)
        logπ(nt)
        pb(logπ, T, nt)

        suite["evaluation"]["$T"] = @benchmarkable $logπ($nt)
        suite["gradient"]["$T"] = @benchmarkable $(pb)($logπ, $T, $nt)

    end
end

suite
