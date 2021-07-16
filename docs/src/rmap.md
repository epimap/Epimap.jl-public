# Rmap

## Models
```@meta
CurrentModule = Epimap.Rmap
```

```@docs
SpatioTemporalGP
RegionalFlux
NegBinomialWeeklyAdjustedTesting
rmap
rmap_naive
```

## Others
```@autodocs
Modules = [Epimap.Rmap]
Filter = t -> nameof(t) ∉ [:SpatioTemporalGP, :RegionalFlux, :NegBinomialWeeklyAdjustedTesting, :rmap, :rmap_naive]
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
