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
args = Rmap.setup_args(Rmap.rmap_naive, data)

# Instantiate model
m = Rmap.rmap_naive(args...; ρ_spatial = 0.1, ρ_time = 100.0, σ_ξ = 0.1);
```

We can then check that the model runs by doing:

```julia
m()
```
