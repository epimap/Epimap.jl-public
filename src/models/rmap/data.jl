using DrWatson,
    CSV,
    DataFrames,
    RemoteFiles,
    UnPack,
    KernelFunctions,
    LinearAlgebra,
    PDMats,
    Adapt,
    Distances,
    DocStringExtensions,
    Dates

function Base.map(f, d::Dict)
    pairs = map(f, zip(keys(d), values(d)))
    return Dict(pairs)
end

function distance_from_areas(metric, areas; units = 100_000)
    area_names = areas[:, "area"]
    coords = Array(areas[:, [:longitude, :latitude]])'
    D = pairwise(metric, coords) ./ units # units are 100km
    return DataFrame(
        hcat(area_names, D),
        vcat(["Column1"], area_names)
    )
end

# TODO: make this nicer
"""
    load_data(rmap_path)

Loads the Rmap data from the data-processing project and returns a named tuple of dataframes.

`rmap_path` can be a local (absolute) path (which should then be prefixed by `"file://"`), or it can be
a direct link to the data-folder in the repository.
"""
function load_data(rmap_path = "file://" * joinpath(ENV["HOME"], "Projects", "private", "Rmap", "data"); metric = nothing)
    # Download files if not present
    @RemoteFileSet datasets "Rmap data" begin
        cases = @RemoteFile "$(rmap_path)/cases.csv" dir=datadir("rmap") updates=:daily
        areas = @RemoteFile "$(rmap_path)/areas.csv" dir=datadir("rmap") updates=:daily
        distances = @RemoteFile "$(rmap_path)/distances.csv" dir=datadir("rmap") updates=:daily
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
        if startswith(string(name), "traffic_flux") || startswith(string(name), "distance")
            # For the traffic flux we need to ensure that we only extract those columns too.
            # Note that because also need to re-order the columns in the same manner, these
            # also needs to be sorted.
            dataframes[name] = df[mask, ["Column1"; sort(df[:, colname])[mask];]]
        else
            dataframes[name] = df[mask, :]
        end
    end

    # If `metric` is specified, e.g. `Haversine`, use that to compute the distances rather than use pre-computed ones.
    if !isnothing(metric)
        # Compute distance dataframe using the provided metric
        @info "Computing distances using $(metric) instead of using pre-computed"
        @info keys(dataframes)
        dataframes[:distances] = distance_from_areas(metric, dataframes[:areas]; units = 100_000)
    end

    # Convert into `NamedTuple` because nicer to work with
    data = DrWatson.dict2ntuple(dataframes)

    # Perform certain transformations
    data.cases[!, 3:end] = convert.(Int, data.cases[:, 3:end])

    # TODO: Make this a part of the test-suite instead.
    # Verify that they're all aligned wrt. area names.
    @assert (
        data.areas[:, "area"] ==
        data.cases[:, "Area name"] ==
        data.traffic_flux_in[:, "Column1"] ==
        names(data.traffic_flux_in)[2:end] ==
        data.traffic_flux_out[:, "Column1"] ==
        names(data.traffic_flux_out)[2:end]==
        data.distances[:, "Column1"] ==
        names(data.distances)[2:end]
    ) "something went wrong with the sorting"

    return data
end

"""
    setup_args(::typeof(rmap_naive), data[, T = Float64]; kwargs...)

Converts `data` into a named tuple with order corresponding to `rmap_naive` constructor.

`T` specifies which element type to use for the data, e.g. `T = Float32` will convert
all floats to `Float32` rather than the default `Float64`.

## Arguments
- [`rmap_naive`](@ref)
- `data`: as returned by [`load_data`](@ref)

## Keyword arguments
- `days_per_step = 1`: specifies how many days to use per step
- `infection_cutoff = 30`: number of previous timesteps which can cause infection on current timestep
- `test_delay_days = 21`: maximum number of days from infection to test
- `presymptomdays = 2`: number of days in which the infection is discoverable
- `first_day_modelled = nothing`: Date of first day to model
- `last_day_modelled = nothing`: Date of last day to model
- `steps_modelled = nothing`: Number of steps to model
- `end_days_ignored = 0`: Number fo days at the end of the data to ignore
- `days_per_step = 7`: Number of days in a single timestep
- `conditioning_days = 30`: Number of conditioning days to use before the start of the modelling

N.b. you do not specify all of the arguments after `first_day_modelled`,
only a combination that allows the periods to use to be computed.
Valid combinations are:
- `first_day_modelled` + `last_day_modelled`, with `(first_day_modelled + last_day_modelled) % days_per_step == 0`
- `first_day_modelled` + `steps_modelled`
- `last_day_modelled` + `steps_modelled`
- `steps_modelled` - will assume you want to model the most recent data minus `end_days_ignored`

## Examples
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

"""
function setup_args(
    ::typeof(rmap_naive),
    data,
    ::Type{T} = Float64;
    infection_cutoff = 30,
    test_delay_days = 21,
    presymptomdays = 2, 
    first_day_modelled = nothing, 
    last_day_modelled = nothing, 
    steps_modelled = nothing, 
    end_days_ignored = 0, 
    days_per_step = 7,
    conditioning_days = 30
) where {T}

    @unpack cases, areas, serial_intervals, traffic_flux_in, traffic_flux_out = data

    # Convert `cases` into a matrix, removing the area-columns
    cases = cases[:, Not(["Country", "Area name"])]
    conditioning_days, modelled_days = process_dates_modelled(
        Date.(names(cases), "y-m-d"),
        first_day_modelled, 
        last_day_modelled, 
        steps_modelled, 
        end_days_ignored, 
        days_per_step,
        conditioning_days
    )
    conditioning_days = Dates.format.(conditioning_days, "yyyy-mm-dd")
    modelled_days = Dates.format.(modelled_days, "yyyy-mm-dd")
    cases = cases[:, vcat(conditioning_days, modelled_days)]
    cases = Array(cases)

    (num_regions, num_days) =  size(cases)
    num_cond = size(conditioning_days)[1]
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
    X_cond = (
        (1.0 - mean_serial_intervals_rem) * cases[:, mean_serial_intervals_int .+ (1:num_cond)]
        + mean_serial_intervals_rem * cases[:, 1 + mean_serial_intervals_int .+ (1:num_cond)]
    )

    # Test delay (numbers taken from original code `Adp` and `Bdp`)
    test_delay_profile = let a = 5.8, b = 0.948
        tmp = cdf.(Gamma(a, b), 1:(test_delay_days - presymptomdays))
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


function process_dates_modelled(
    dates, 
    first_day_modelled = nothing, 
    last_day_modelled = nothing, 
    steps_modelled = nothing, 
    end_days_ignored = 0, 
    days_per_step = 7,
    conditioning_days = 30
)
    dates = dates[1:end-end_days_ignored]

    if (isnothing(first_day_modelled) & ~isnothing(last_day_modelled) & ~isnothing(steps_modelled))
        last_day_index = findall(d->d==last_day_modelled, dates)[1]
        first_day_index = last_day_index - (steps_modelled * days_per_step) + 1
    elseif (~isnothing(first_day_modelled) & isnothing(last_day_modelled) & ~isnothing(steps_modelled))
        first_day_index = findall(d->d==first_day_modelled, dates)[1]
        last_day_index = first_day_index + (steps_modelled * days_per_step) - 1
    elseif (~isnothing(first_day_modelled) & ~isnothing(last_day_modelled) & isnothing(steps_modelled))
        last_day_index = findall(d->d==last_day_modelled, dates)[1]
        first_day_index = findall(d->d==first_day_modelled, dates)[1]
        @assert (last_day_index - first_day_index) % days_per_step == 0
        steps_modelled = (last_day_index - first_day_index) ÷ days_per_step 
    elseif (isnothing(first_day_modelled) & isnothing(last_day_modelled) & ~isnothing(steps_modelled))
        last_day_index = size(dates)[1]
        first_day_index = last_day_index - (steps_modelled * days_per_step) + 1
    else
        # Throw error
    end

    @assert last_day_index <= size(dates)[1]
    @assert first_day_index - conditioning_days >= 1

    return (conditioning_days=dates[(first_day_index - conditioning_days):(first_day_index - 1)], modelled_days=dates[first_day_index:last_day_index])
end

"""
    $(SIGNATURES)

Filters `data` by regions within `radius` of the `main_regions`.
"""
function filter_areas_by_distance(data; num_main_regions = 1, epidemic_start = 241, kwargs...)
    @unpack cases, areas, serial_intervals, traffic_flux_in, traffic_flux_out = data

    # Find the top-k areas in terms of number of cases when the pandemic "began".
    area_names = areas[:, :area]
    top_k_areas = partialsortperm(cases[:, names(cases)[epidemic_start]], 1:num_main_regions, rev=true)
    main_areas_names = area_names[top_k_areas]

    return filter_areas_by_distance(data, main_areas_names; kwargs...)
end

function filter_areas_by_distance(data, main_area::String; kwargs...)
    filter_areas_by_distance(data, [main_area]; kwargs...)
end

function filter_areas_by_distance(
    data, main_areas::AbstractVector{String};
    radius = Inf
)
    @info "Filtering based regions which are within $radius of $(main_areas)"
    @unpack cases, distances, areas, serial_intervals, traffic_flux_in, traffic_flux_out = data
    main_indices = findall(∈(main_areas), areas[:, :area])

    # Find the nearest neighbors
    # Get the coords for all areas
    coords = Array(areas[:, [:longitude, :latitude]])'
    # Only for the top ones
    top_k_coords = coords[:, main_indices]
    # Compute pairwise distances
    # If `metric` is specified, e.g. `Haversine`, use that to compute the distances rather than use pre-computed ones.
    distances_mat = Array(data.distances[main_indices, Not("Column1")])
    # For each `row` (i.e. each of the `top_k_ares` that we're targeting), filter based others by radius.
    close_indices = map(eachrow(distances_mat)) do row
        findall(<(radius), row)
    end

    # Concatenate
    indices_to_include = vcat(close_indices...)
    # Ensure that we don't include a region twice
    unique!(indices_to_include)
    # Sorting indices by value corresponds to sorting the areas according to the original ordering.
    sort!(indices_to_include)
    # Extract the names
    names_to_include = areas[indices_to_include, :area]

    new_data = (
        areas = areas[indices_to_include, :],
        cases = cases[indices_to_include, :],
        traffic_flux_in = traffic_flux_in[indices_to_include, vcat("Column1", names_to_include)],
        traffic_flux_out = traffic_flux_out[indices_to_include, vcat("Column1", names_to_include)],
        area_names = names_to_include
    )

    return merge(data, new_data)
end
