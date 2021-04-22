# NOTE: This only sets up the benchmark suite; it does NOT run it!
# To run it, open a Julia repl and include this file (using `include`), and then
# follow up with `run(suite, verbose=True, seconds=60)` or whatever options you want.

using BenchmarkTools

using Epimap, Adapt, Zygote, Dates
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
setup_args = merge(Rmap.setup_args(
    Rmap.rmap_naive, 
    data, 
    first_date = Date("2020-10-01", "y-m-d"),
    num_steps = 15,
    timestep = Week(1),
    num_condition_days = 30
), default_args);

# Construct benchmark-suite
suite = BenchmarkGroup()

suite["logjoint"] = BenchmarkGroup()
suite["logjoint_unconstrained"] = BenchmarkGroup()

suite["logjoint"]["evaluation"] = BenchmarkGroup()
suite["logjoint"]["gradient"] = BenchmarkGroup()

suite["logjoint_unconstrained"]["evaluation"] = BenchmarkGroup()
suite["logjoint_unconstrained"]["gradient"] = BenchmarkGroup()

for T ∈ [Float32, Float64]
    logπ, logπ_unconstrained, b, θ = Epimap.make_logjoint(Rmap.rmap_naive, setup_args...)
    ϕ = inv(b)(θ)

    # HACK: Execute once to compile (does this even matter? I thought BenchmarkTools would take care of this.)
    pb(logπ, T, θ)
    pb(logπ_unconstrained, T, ϕ)

    suite["logjoint"]["evaluation"]["$T"] = @benchmarkable $logπ($θ)
    suite["logjoint"]["gradient"]["$T"] = @benchmarkable $(pb)($logπ, $T, $θ)

    suite["logjoint_unconstrained"]["evaluation"]["$T"] = @benchmarkable $logπ_unconstrained($ϕ)
    suite["logjoint_unconstrained"]["gradient"]["$T"] = @benchmarkable $(pb)($logπ_unconstrained, $T, $ϕ)
end

suite
