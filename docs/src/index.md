```@meta
CurrentModule = Epimap
```

# Epimap

Documentation for [Epimap](https://github.com/epimap/Epimap.jl).

```@index
```

```@autodocs
Modules = [Epimap]
```

## Rmap
```@autodocs
Modules = [Epimap.Rmap]
```

## Examples

### Running [`Rmap.rmap_naive`](@ref)

```julia
using Epimap

# Load data
data = Rmap.load_data();

# Construct the model arguments from data
setup_args = Rmap.setup_args(Rmap.rmap_naive, data)

# Arguments not related to the data which are to be set up
default_args = (
    ρ_spatial = 10.0,
    ρ_time = 0.1,
    σ_spatial = 0.1,
    σ_local = 0.1,
    σ_ξ = 1.0
)

# Instantiate model
m = Rmap.rmap_naive(merge(setup_args, default_args)...);
```

We can then check that the model runs by doing:

```julia
m()
```

Normally this would be followed up by a simple `sample(m, sampler, num_samples)` statement, e.g. `sample(m, NUTS(), 1000)`, but since this model is going to be huge (~300K parameters) that isn't going to go so well (at the time of writing [2021-03-26 Fri]).

**Soon™**

