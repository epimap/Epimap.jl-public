using DrWatson, CSV, DataFrames, RemoteFiles, UnPack, KernelFunctions, LinearAlgebra, PDMats, Adapt


function Base.map(f, d::Dict)
    pairs = map(f, zip(keys(d), values(d)))
    return Dict(pairs)
end

# TODO: make this nicer
"""
    load_data(rmap_path)

Loads the Rmap data from the data-processing project and returns a named tuple of dataframes.

`rmap_path` can be a local (absolute) path (which should then be prefixed by `"file://"`), or it can be
a direct link to the data-folder in the repository.
"""
function load_data(rmap_path = "file://" * joinpath(ENV["HOME"], "Projects", "private", "Rmap", "data"))
    # Download files if not present
    @RemoteFileSet datasets "Rmap data" begin
        cases = @RemoteFile "$(rmap_path)/cases.csv" dir=datadir("rmap") updates=:daily
        areas = @RemoteFile "$(rmap_path)/areas.csv" dir=datadir("rmap") updates=:daily
        serial_intervals = @RemoteFile "$(rmap_path)/serial_interval.csv" dir=datadir("rmap") updates=:never
        traffic_flux_in = @RemoteFile "$(rmap_path)/uk_reverse_commute_flow.csv" dir=datadir("rmap") updates=:never
        traffic_flux_out = @RemoteFile "$(rmap_path)/uk_forward_commute_flow.csv" dir=datadir("rmap") updates=:never
    end

    # Download files if out of date
    download(datasets)

    # Load files
    dataframes = map(datasets.files) do (name, remotefile)
        (name, DataFrame(CSV.File(path(remotefile))))
    end

    # Find the areas that we have all kinds of data for
    areas_sorted = sort(dataframes[:areas][:, :area])
    valid_areas = Set(areas_sorted)
    df2areacol = Dict{Symbol, String}()

    # Try to determine which column correspond to the area name
    for (name, df) in pairs(dataframes)
        idx = findfirst(names(df)) do colname
            startswith(lowercase(colname), "area") && return true

            if string(colname) == "Column1" && eltype(df[:, colname]) <: String
                return true
            end

            return false
        end

        # If we found a column that matches our "area" search criterion, we update the intersection.
        if !isnothing(idx)
            colname = names(df)[idx]
            df2areacol[name] = colname
            intersect!(valid_areas, Set(df[:, colname]))
        end
    end

    # Filter data based on the available areas
    for (name, colname) in pairs(df2areacol)
        # First sort the dataframe according to names to ensure that we get consistent ordering
        df = sort(dataframes[name], colname)
        mask = ∈(valid_areas).(df[:, colname])
        if startswith(string(name), "traffic_flux")
            # For the traffic flux we need to ensure that we only extract those columns too.
            # Note that because also need to re-order the columns in the same manner, these
            # also needs to be sorted.
            dataframes[name] = df[mask, ["Column1"; sort(df[:, colname])[mask];]]
        else
            dataframes[name] = df[mask, :]
        end
    end

    # Convert into `NamedTuple` because nicer to work with
    data = DrWatson.dict2ntuple(dataframes)

    # Perform certain transformations
    data.cases[!, 3:end] = convert.(Int, data.cases[:, 3:end])

    # TODO: Make this a part of the test-suite instead.
    @assert (
        data.areas[:, "area"] ==
        data.cases[:, "Area name"] ==
        data.traffic_flux_in[:, "Column1"] ==
        names(data.traffic_flux_in)[2:end] ==
        data.traffic_flux_out[:, "Column1"] ==
        names(data.traffic_flux_out)[2:end]
    ) "something went wrong with the sorting"

    return data
end

"""
    setup_args(::typeof(rmap_naive), data[, T = Float64]; kwargs...)

Converts `data` into a named tuple with order corresponding to `rmap_naive` constructor.

`T` specifies which element type to use for the data, e.g. `T = Float32` will convert
all floats to `Float32` rather than the default `Float64`.

This allows one to do the following

```julia
data = Rmap.load_data()
setup_args = Rmap.setup_args(Rmap.rmap_naive, data)
model = Rmap.rmap_naive(setup_args...)
```

Note that this method does *not* return arguments for which `rmap_naive` has default values.
If one wants to override the default arguments, then one needs to as follows:
```julia
data = Rmap.load_data()

# IMPORTANT: Order needs to be the same as model-arguments!!!
default_args = (
    ρ_spatial = 10.0,
    ρ_time = 0.1,
    σ_spatial = 0.1,
    σ_local = 0.1,
    σ_ξ = 1.0,
)

setup_args = merge(Rmap.setup_args(Rmap.rmap_naive, data), default_args)
model = Rmap.rmap_naive(setup_args...)
```


## Arguments
- [`rmap_naive`](@ref)
- `data`: as returned by [`load_data`](@ref)

## Keyword arguments
- `days_per_step = 1`: specifies how many days to use per step (WARNING: does nothing at the moment)
- `num_cond = 1`: specifies how many days at the start of the cases to compute approximate Xt for to condition on in the model.
- `infection_cutoff = 30`: number of previous timesteps which can cause infection on current timestep
- `test_delay_days = 21`: maximum number of days from infection to test
- `presymptomdays = 2`: number of days in which the infection is discoverable

"""
function setup_args(
    ::typeof(rmap_naive),
    data,
    ::Type{T} = Float64;
    days_per_step = 1,
    num_cond = 1,
    infection_cutoff = 30,
    test_delay_days = 21,
    presymptomdays = 2
) where {T}

    @unpack cases, areas, serial_intervals, traffic_flux_in, traffic_flux_out = data

    # Convert `cases` into a matrix, removing the area-columns
    cases = Array(cases[:, Not(["Country", "Area name"])])
    cases = cases[:,241:420] # TODO: proper parsing of what region of time we want to infer for. Temp: cut off the first 240 days to prevent issues with zero cases
    (num_regions, num_days) =  size(cases)
    num_infer = num_days - num_cond
    @assert num_infer % days_per_step == 0

    # Serial intervals / infection profile
    serial_intervals = serial_intervals[1:min(infection_cutoff, size(serial_intervals, 1)), :fit]
    normalize!(serial_intervals, 1) # re-normalize wrt. ℓ1-norm to ensure still sums to 1
    @assert sum(serial_intervals) ≈ 1.0 "truncated serial_intervals does not sum to 1"
    mean_serial_intervals = sum((1:size(serial_intervals,1)) .* serial_intervals)
    mean_serial_intervals_int = Int(floor(mean_serial_intervals))
    mean_serial_intervals_rem = mean_serial_intervals - mean_serial_intervals_int

    # Precompute conditioning X approximation
    X_cond = (1.0 - mean_serial_intervals_rem) * cases[:, mean_serial_intervals_int .+ (1:num_cond)] + mean_serial_intervals_rem * cases[:, 1 + mean_serial_intervals_int .+ (1:num_cond)]

    # Test delay (numbers taken from original code `Adp` and `Bdp`)
    test_delay_profile = let a = 5.8, b = 0.948
        tmp = cdf(Gamma(a, b), 1:(test_delay_days - presymptomdays))
        tmp ./= tmp[end]
        tmp = tmp - vcat(zeros(1), tmp[1:end - 1])
        vcat(zeros(presymptomdays), tmp)
    end
    @assert sum(test_delay_profile) ≈ 1.0 "test_delay_profile does not sum to 1"

    ### Spatial kernel ###
    # TODO: make it this an argument?
    centers = Array(areas[1:end, ["longitude", "latitude"]])'
    k_spatial = Matern12Kernel()
    K_spatial = PDMat(kernelmatrix(k_spatial, centers))
    K_local = PDiagMat(ones(num_regions))

    ### Temporal kernel ###
    # TODO: make it this an argument?
    k_time = Matern12Kernel()
    K_time = PDMat(kernelmatrix(k_time, 1:days_per_step:num_infer))

    # Flux matrices
    F_id = Diagonal(ones(num_regions))
    F_out = Array(traffic_flux_out[1:end, 2:end])
    F_in = Array(traffic_flux_in[1:end, 2:end])

    # Resulting arguments
    result = (
        C = cases,
        D = test_delay_profile,
        W = serial_intervals,
        X_cond = X_cond,
        F_id = F_id,
        F_out = F_out,
        F_in = F_in,
        K_time = K_time,
        K_spatial = K_spatial,
        K_local = K_local,
        days_per_step = days_per_step,
    )

    # 
    return adapt(Epimap.FloatMaybeAdaptor{T}(), result)
end
