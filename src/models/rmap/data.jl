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
    Dates,
    StatsFuns,
    SparseArrays

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

function make_test_delay_profile(test_delay_days; presymptomdays=2, a=5.8, b=0.948)
    tmp = cdf.(Gamma(a, b), 1:(test_delay_days - presymptomdays))
    tmp ./= tmp[end]
    tmp = tmp - vcat(zeros(1), tmp[1:end - 1])

    return vcat(zeros(presymptomdays), tmp)
end

# TODO: make this nicer
"""
    load_data(rmap_path)

Loads the Rmap data from the data-processing project and returns a named tuple of dataframes.

`rmap_path` can be a local (absolute) path (which should then be prefixed by `"file://"`), or it can be
a direct link to the data-folder in the repository.
"""
function load_data(
    rmap_path = "file://" * get(ENV, "EPIMAP_DATA", joinpath(ENV["HOME"], "Projects", "private", "Rmap", "data"));
    metric=nothing,
    debiased_path=datadir("logit_moments.csv")
)
    # Download files if not present
    @RemoteFileSet datasets "Rmap data" begin
        cases = @RemoteFile "$(rmap_path)/cases.csv" dir=datadir("rmap") updates=:daily
        areas = @RemoteFile "$(rmap_path)/areas.csv" dir=datadir("rmap") updates=:daily
        distances = @RemoteFile "$(rmap_path)/distances.csv" dir=datadir("rmap") updates=:daily
        serial_intervals = @RemoteFile "$(rmap_path)/serial_interval.csv" dir=datadir("rmap") updates=:never
        traffic_flux_in = @RemoteFile "$(rmap_path)/uk_reverse_commute_flow.csv" dir=datadir("rmap") updates=:never
        traffic_flux_out = @RemoteFile "$(rmap_path)/uk_forward_commute_flow.csv" dir=datadir("rmap") updates=:never
        debiased = @RemoteFile "file://" * debiased_path dir=datadir("debiased") updates=:never
        pcr_positivity = @RemoteFile "file://" * datadir("pcr_daily.csv") dir=datadir("debiased") updates=:never
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
        mask = ???(valid_areas).(df[:, colname])
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
    # TODO: Fix the issue that `distances` aren't being filtered appropriately.
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

function compute_X_cond(cases, test_delay_profile, num_cond)
    # For time `t` we backshift the cases from time `t + mean_test_delay_profile_int`.
    mean_test_delay_profile = sum((1:size(test_delay_profile,1)) .* test_delay_profile)
    mean_test_delay_profile_int = Int(floor(mean_test_delay_profile))
    mean_test_delay_profile_rem = mean_test_delay_profile - mean_test_delay_profile_int

    # Linear interoplation between the value of the floored index and the ceiled index.
    return (
        (1.0 - mean_test_delay_profile_rem) * cases[:, mean_test_delay_profile_int .+ (1:num_cond)]
        + mean_test_delay_profile_rem * cases[:, 1 + mean_test_delay_profile_int .+ (1:num_cond)]
    )
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
- `days_per_step = 1`: specifies how many days to use per step.
- `infection_cutoff = 30`: number of previous timesteps which can cause
   infection on current timestep.
- `test_delay_days = 21`: maximum number of days from infection to test.
- `presymptomdays = 2`: number of days in which the infection is discoverable.
- `first_date = nothing`: date of first day to model.
- `last_date = nothing`: date of last day to model.
- `num_steps = nothing`: number of steps to model.
- `end_days_ignored = 0`: number of days at the end of the data to ignore.
- `timestep = Week(1)`: period of time between steps to model.
- `num_condition_days = 30`: number of conditioning days to use before the start of the
  modelling. Specifices how many to use in `X_cond` in the return-value.

## Notes
- The dates used for conditioning and modelling will be computed by [`split_dates`](@ref).
  See its documentation for more information how exactly this is done.

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
    ??_spatial = 10.0,
    ??_time = 0.1,
    ??_spatial = 0.1,
    ??_local = 0.1,
    ??_?? = 1.0,
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
    first_date = nothing, 
    last_date = nothing, 
    num_steps = nothing, 
    num_end_days_ignore = 0, 
    timestep = Week(1),
    num_condition_days = 30,
    condition_observations = false,
    include_dates = false
) where {T}
    days_per_step = Dates.days(timestep)
    
    @unpack cases, areas, serial_intervals, traffic_flux_in, traffic_flux_out = data

    # Convert `cases` into a matrix, removing the area-columns
    cases = cases[:, Not(["Country", "Area name"])]
    dates_str = names(cases)[1:end - num_end_days_ignore]
    dates_condition, dates_model = split_dates(
        Date.(dates_str, "y-m-d");
        first_date = first_date,
        last_date = last_date,
        num_steps = num_steps,
        timestep = timestep,
        num_condition_days = num_condition_days
    )
    @info "Using the following dates of data" dates_condition dates_model
    dates_condition_str = Dates.format.(dates_condition, "yyyy-mm-dd")
    dates_model_str = Dates.format.(dates_model, "yyyy-mm-dd")
    cases = Array(cases[:, vcat(dates_condition_str, dates_model_str)])

    (num_regions, num_days) =  size(cases)
    num_cond = size(dates_condition, 1)
    num_infer = num_days - num_cond
    @assert num_infer % days_per_step == 0 "$(num_infer) not divisible by $(days_per_step)"

    # Serial intervals / infection profile
    serial_intervals = serial_intervals[1:min(infection_cutoff, size(serial_intervals, 1)), :fit]
    # re-normalize wrt. ???1-norm to ensure still sums to 1
    normalize!(serial_intervals, 1)
    @assert sum(serial_intervals) ??? 1.0 "truncated serial_intervals does not sum to 1"

    # Test delay (numbers taken from original code `Adp` and `Bdp`)
    test_delay_profile = make_test_delay_profile(test_delay_days)
    @assert sum(test_delay_profile) ??? 1.0 "test_delay_profile does not sum to 1"

    # Precompute conditioning X approximation
    X_cond = if condition_observations
        cases[:, 1:num_cond]
    elseif num_condition_days > 0
        compute_X_cond(cases, test_delay_profile, num_condition_days)
    else
        nothing
    end

    ### Spatial kernel ###
    # TODO: make it this an argument?
    centers = Array(areas[1:end, ["longitude", "latitude"]])
    k_spatial = Matern12Kernel()
    spatial_distances = KernelFunctions.pairwise(
        KernelFunctions.Haversine(),
        KernelFunctions.RowVecs(centers)
    ) ./ 100_000 # want in units of 100km
    K_spatial = PDMat(map(Base.Fix1(KernelFunctions.kappa, k_spatial), spatial_distances))
    K_local = PDiagMat(ones(num_regions))

    ### Temporal kernel ###
    # TODO: make it this an argument?
    k_time = Matern12Kernel()
    times = 1:days_per_step:num_infer
    time_distances = KernelFunctions.pairwise(KernelFunctions.Euclidean(), times)
    K_time = PDMat(map(Base.Fix1(KernelFunctions.kappa, k_time), time_distances))

    # Flux matrices
    F_id = Diagonal(ones(num_regions))
    F_out = Array(traffic_flux_out[1:end, 2:end])
    F_in = Array(traffic_flux_in[1:end, 2:end])

    # Normalize rows in the flux-matrices, ensuring that they sum to 1.
    for row in eachslice(F_in, dims=1)
        normalize!(row, 1)
    end
    for row in eachslice(F_out, dims=1)
        normalize!(row, 1)
    end

    # Resulting arguments
    result = (
        C = cases[:, num_cond + 1:end],
        D = test_delay_profile,
        W = serial_intervals,
        F_id = F_id,
        F_out = F_out,
        F_in = F_in,
        K_time = K_time,
        K_spatial = K_spatial,
        K_local = K_local,
        days_per_step = days_per_step,
        X_cond = clamp.(X_cond, eps(T), Inf)
    )

    result_adapted = adapt(Epimap.FloatMaybeAdaptor{T}(), result)
    return if include_dates
        result_adapted, (condition = dates_condition, model = dates_model)
    else
        result_adapted
    end
end

function setup_args(::typeof(rmap), data, ::Type{T} = Float64; kwargs...) where {T}
    # `rmap` and `rmap_naive` take the same arguments.
    return setup_args(rmap_naive, data, T; kwargs...)
end

function make_projection(_names_latent, names_observed, dest2sources=Dict{String,Vector{String}}(); drop_invalid=true, zeros=spzeros)
    # Filter out those which are not present in `names_observed` nor in `dest2sources`, since
    # for those we have no observations.
    names_latent = filter(_names_latent) do name
        name ??? names_observed || any(name .??? values(dest2sources))
    end

    # Extract the indices in `names_latent` which are to be projected to the corresponding
    # index in `names_observed`.
    dest2sources_indices = map(names_observed) do dest
        if haskey(dest2sources, dest)
            map(dest2sources[dest]) do source
                findfirst(==(source), names_latent)
            end
        else
            [findfirst(==(dest), names_latent)]
        end
    end

    if !drop_invalid && length(names_latent) !== length(_names_latent)
        messed_up = findall(dest2sources_indices) do inds
            any(isnothing, inds)
        end
        @warn "The following regions could not be resolved" names_observed[messed_up]
        error("some names in `names_latent` are not present in `names_observed` or `dest2sources`")
    end

    # Construct Spares
    nobserved = length(dest2sources_indices)
    nlatent = sum(length, dest2sources_indices)
    P = zeros(nobserved, nlatent)
    for (dest, sources) in enumerate(dest2sources_indices)
        for source in sources
            P[dest, source] = 1
        end
    end

    return names_latent, names_observed, P
end


function setup_args(
    ::typeof(rmap_debiased),
    data,
    ::Type{T} = Float64;
    infection_cutoff = 30,
    test_delay_days = 21,
    presymptomdays = 2,
    first_date = nothing,
    last_date = nothing,
    num_steps = nothing,
    num_end_days_ignore = 0,
    timestep = Week(1),
    num_condition_days = 30,
    condition_observations = false,
    include_dates = false,
    observation_timestep = Week(1),
    dest2sources = nothing
) where {T}
    days_per_step = Dates.days(timestep)
    days_per_observation = Dates.days(observation_timestep)

    @unpack cases, areas, serial_intervals, traffic_flux_in, traffic_flux_out = data

    # Convert `cases` into a matrix, removing the area-columns
    cases = cases[:, Not(["Country", "Area name"])]
    dates_str = names(cases)[1:end - num_end_days_ignore]
    dates_condition, dates_model = split_dates(
        Date.(dates_str, "y-m-d");
        first_date = first_date,
        last_date = last_date,
        num_steps = num_steps,
        timestep = timestep,
        num_condition_days = num_condition_days
    )
    @info "Using the following dates of data" dates_condition dates_model
    dates_condition_str = Dates.format.(dates_condition, "yyyy-mm-dd")
    dates_model_str = Dates.format.(dates_model, "yyyy-mm-dd")

    # Some counting of DAYS, not timesteps.
    num_cond = length(dates_condition)
    num_infer = length(dates_model)
    num_times = num_cond + num_infer

    # Sanity checking
    @assert num_infer % days_per_step == 0 "$(num_infer) not divisible by $(days_per_step)"

    # Debiased data.
    debiased = data.debiased
    @assert dayofweek(dates_model[1]) == 4 "First date modeled $(dates_model[end]) is not a Thursday"
    @assert dayofweek(dates_model[end]) == 3 "Last date modeled $(dates_model[end]) is not a Wednesday"
    @assert (dates_model[1] + Day(3)) in debiased[:, :mid_week] "First date modeled $(dates_model[end]) is not in the debiased dataset"
    @assert (dates_model[end] - Day(3)) in debiased[:, :mid_week] "Last date modeled $(dates_model[end]) is not in the debiased dataset"
    # Filter the dates.
    debiased = debiased[debiased[:, :mid_week] .??? Ref(vcat(dates_condition, dates_model)), :]

    # TODO: Resolve the regions with what we have unbiased estimates for.
    dest2sources = !isnothing(dest2sources) ? dest2sources : Dict(
        "North Northamptonshire" => ["Kettering", "Corby", "Wellingborough", "East Northamptonshire"],
        "West Northamptonshire" => ["Daventry", "Northampton", "South Northamptonshire"],
        "Cornwall" => ["Cornwall and Isles of Scilly"],
        "Hackney" => ["Hackney and City of London"]
    )
    area_names_rmap = unique(areas[:, :area])
    area_names_debiased = unique(debiased[:, :ltla])
    area_names_latent, area_names_observed, P = make_projection(area_names_rmap, area_names_debiased, dest2sources)
    area_mask = findall(???(area_names_latent), area_names_rmap)
    area_mask_debiased = findall(???(area_names_observed), area_names_debiased)

    # Populations for latent areas.
    populations = areas[areas[:, :area] .??? Ref(area_names_latent), :population]

    # Extract cases used for `X_cond`.
    cases = Array(
        cases[in(area_names_latent).(data.cases[:, "Area name"]), vcat(dates_condition_str, dates_model_str)]
    )
    @assert size(cases, 1) == length(area_names_latent)

    # Extract wanted information from filtered `debiased`.
    debiased = debiased[debiased[:, :ltla] .??? Ref(area_names_observed), :]
    by_ltla = groupby(debiased, :ltla)
    logit?? = permutedims(mapreduce(g -> g[:, :mean], hcat, by_ltla), (2, 1))
    ??_debias = permutedims(mapreduce(g -> g[:, :sd], hcat, by_ltla), (2, 1))

    # Data to be modeled. It's just easier to extract it like this than mess around
    # with the number of steps
    debiased_model = debiased[debiased.mid_week .??? Ref(dates_model), :]
    by_ltla_model = groupby(debiased_model, :ltla)
    logit??_model = permutedims(mapreduce(g -> g[:, :mean], hcat, by_ltla_model), (2, 1))
    ??_debias_model = permutedims(mapreduce(g -> g[:, :sd], hcat, by_ltla_model), (2, 1))

    # Test delay (numbers taken from original code `Adp` and `Bdp`)
    test_delay_profile = make_test_delay_profile(test_delay_days)
    @assert sum(test_delay_profile) ??? 1.0 "test_delay_profile does not sum to 1"

    # Serial intervals / infection profile
    serial_intervals = serial_intervals[1:min(infection_cutoff, size(serial_intervals, 1)), :fit]
    # re-normalize wrt. ???1-norm to ensure still sums to 1
    normalize!(serial_intervals, 1)
    @assert sum(serial_intervals) ??? 1.0 "truncated serial_intervals does not sum to 1"

    # PCR positivity profile
    pcr_positivity_profile = vec(Array(data.pcr_positivity))

    # Test delay (numbers taken from original code `Adp` and `Bdp`)
    test_delay_profile = let a = 5.8, b = 0.948
        tmp = cdf.(Gamma(a, b), 1:(test_delay_days - presymptomdays))
        tmp ./= tmp[end]
        tmp = tmp - vcat(zeros(1), tmp[1:end - 1])
        vcat(zeros(presymptomdays), tmp)
    end
    @assert sum(test_delay_profile) ??? 1.0 "test_delay_profile does not sum to 1"

    # Precompute conditioning X approximation
    X_cond = if condition_observations
        cases[:, 1:num_cond]
    elseif num_condition_days > 0
        compute_X_cond(cases, test_delay_profile, num_condition_days)
    else
        nothing
    end

    num_regions = length(area_names_latent)
    @assert size(cases, 1) == num_regions

    ### Spatial kernel ###
    # TODO: make it this an argument?
    centers = Array(areas[area_mask, ["longitude", "latitude"]])
    @assert areas[area_mask, :area] == area_names_latent "the filter used for `data.areas` results in incorrect area names"
    k_spatial = Matern12Kernel()
    spatial_distances = KernelFunctions.pairwise(
        KernelFunctions.Haversine(),
        KernelFunctions.RowVecs(centers)
    ) ./ 100_000 # want in units of 100km
    K_spatial = PDMat(map(Base.Fix1(KernelFunctions.kappa, k_spatial), spatial_distances))
    K_local = PDiagMat(ones(num_regions))

    ### Temporal kernel ###
    # TODO: make it this an argument?
    k_time = Matern12Kernel()
    times = 1:days_per_step:num_infer
    time_distances = KernelFunctions.pairwise(KernelFunctions.Euclidean(), times)
    K_time = PDMat(map(Base.Fix1(KernelFunctions.kappa, k_time), time_distances))

    # Flux matrices
    F_id = Diagonal(ones(num_regions))    
    @assert traffic_flux_out[area_mask, 1] == area_names_latent
    @assert names(traffic_flux_out)[1 .+ area_mask] == area_names_latent

    @assert traffic_flux_in[area_mask, 1] == area_names_latent
    @assert names(traffic_flux_in)[1 .+ area_mask] == area_names_latent

    F_out = Array(traffic_flux_out[area_mask, 1 .+ area_mask])
    F_in = Array(traffic_flux_in[area_mask, 1 .+ area_mask])

    # Normalize rows in the flux-matrices, ensuring that they sum to 1.
    for row in eachslice(F_in, dims=1)
        normalize!(row, 1)
    end
    for row in eachslice(F_out, dims=1)
        normalize!(row, 1)
    end

    # Resulting arguments
    result = (
        logit?? = logit??_model,
        ??_debias = ??_debias_model,
        populations = populations,
        D = pcr_positivity_profile,
        W = serial_intervals,
        F_id = F_id,
        F_out = F_out,
        F_in = F_in,
        K_time = K_time,
        K_spatial = K_spatial,
        K_local = K_local,
        days_per_step = days_per_step,
        X_cond = clamp.(X_cond, eps(T), T(Inf)),
        observed_projection=P
    )

    result_adapted = adapt(Epimap.FloatMaybeAdaptor{T}(), result)
    return if include_dates
        result_adapted, (condition = dates_condition, model = dates_model)
    else
        result_adapted
    end
end

# TODO: Move out of `Rmap` module since it can be useful for other parts.
"""
    split_dates(dates; kwargs...)

Split `dates` into days to condition on and days to be inferred/modelled.

By default chooses days to be inferred/modelled as longest period
divisible by `days_per_step`.

## Keyword arguments
- `first_date = nothing`: the first date which will be inferred.
- `last_date = nothing`: the last date which will be inferred.
- `num_steps = nothing`: number of time-steps to include in period to infer.
  Only makes sense if either `first_date` or `last_date` is specified.
- `timestep = Week(1)`: length between each . Checks will be made to
  ensure that the period to be inferred is divisible by `days_per_step`.
- `num_condition_days = 30`: number of days to condition on, i.e. not to be
  learned by the model.
"""
function split_dates(
    dates;
    first_date = nothing,
    last_date = nothing,
    num_steps = nothing,
    timestep = Week(1),
    num_condition_days = 30
)
    days_per_step = Dates.days(timestep)

    if isnothing(first_date) & isnothing(last_date) & isnothing(num_steps)
        first_date = dates[num_condition_days + 1]
        last_available_day = dates[end]
        num_steps = Dates.days(last_available_day - first_date) ?? days_per_step
        # Subtract 1 day since we include data for `last_date`
        last_date = first_date + Day((num_steps * days_per_step) - 1)
    elseif ~isnothing(first_date) & isnothing(last_date) & isnothing(num_steps)
        last_available_day = dates[end]
        num_steps = Dates.days(last_available_day - first_date) ?? days_per_step
        # Subtract 1 day since we include data for `last_date`
        last_date = first_date + Day((num_steps * days_per_step) - 1)
    elseif isnothing(first_date) & ~isnothing(last_date) & isnothing(num_steps)
        # Only `last_date` is known => choose longest possible period that ends
        # on `last_date`.
        first_date_available = dates[num_condition_days + 1]
        num_steps = Dates.days(last_date - first_date_available) ?? days_per_step
        first_date = last_date - Day(days_per_step * num_steps - 1)
    elseif isnothing(first_date) & ~isnothing(last_date) & ~isnothing(num_steps)
        first_date = last_date - Day((num_steps * days_per_step) - 1)
    elseif ~isnothing(first_date) & isnothing(last_date) & ~isnothing(num_steps)
        last_date = first_date + Day((num_steps * days_per_step) - 1)
    elseif isnothing(first_date) & isnothing(last_date) & ~isnothing(num_steps)
        last_date = dates[end]
        first_date = last_date - Day((num_steps * days_per_step) - 1)
    elseif ~isnothing(first_date) & ~isnothing(last_date)
        # If we both `first_date` and `last_date` are provided, we ignore `num_steps`.
        @warn "ignoring `num_steps` since both `first_date` and `last_date` are given"
        period = first_date:Day(1):last_date
        num_steps = length(period) ?? days_per_step
        # If the provided `first_date` and `last_date` are provided BUT they don't
        # satisfy the divisibility constraint, we act like `first_date` wasn't provided
        # and choose the largest possible period ending on `last_date`.
        if length(period) % days_per_step != 0
            first_date_available = dates[num_condition_days + 1]
            num_steps = Dates.days(last_date - first_date_available) ?? days_per_step
            first_date = last_date - Day(days_per_step * num_steps - 1)
            new_period = first_date:Day(1):last_date
            @warn "period $(period) not divisible by $(days_per_step); choosing $(new_period) instead"
        end
    else
        throw(ArgumentError("don't know what to do with the arguments"))
    end

    # Date to condition on is either `num_condition_days` before `first_date`
    # or the very first day available; we choose whichever is most recent.
    condition_start_date = max(first_date - Day(num_condition_days), dates[1])
    if first_date - Day(num_condition_days) < condition_start_date
        # Warn user if the above `max` chose `dates[1]` instead of following the desired
        # `num_condition_days`.
        @warn "Attempted to condition on days prior to available dates; starting conditioning from $(dates[1]) instead"
    end

    @assert last_date ??? dates[end] "`last_date` is out-of-bounds"
    @assert first_date ??? dates[1] "`first_date` is out-of-bounds"
    # If `num_condition_days` is 0, then `condition_start_date` will be the same as `first_date`
    # but the resulting range will be empty, i.e. no need to throw an error.
    @assert num_condition_days == 0 || first_date > condition_start_date "`first_date` is within the conditioning-period"

    return (
        condition=condition_start_date:Day(1):first_date - Day(1),
        model=first_date:Day(1):last_date
    )
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
    radius = Inf,
    filter_debiased = true
)
    @info "Filtering based regions which are within $radius of $(main_areas)"
    @unpack cases, distances, areas, serial_intervals, traffic_flux_in, traffic_flux_out, debiased = data
    main_indices = findall(???(main_areas), areas[:, :area])

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

    # Filter debiased data based on the area names.
    debiased_filtered = if filter_debiased
        debiased[debiased.ltla .??? Ref(names_to_include), :]
    else
        debiased
    end

    new_data = (
        areas = areas[indices_to_include, :],
        cases = cases[indices_to_include, :],
        distances = distances[indices_to_include, vcat("Column1", names_to_include)],
        traffic_flux_in = traffic_flux_in[indices_to_include, vcat("Column1", names_to_include)],
        traffic_flux_out = traffic_flux_out[indices_to_include, vcat("Column1", names_to_include)],
        debiased = debiased_filtered,
        area_names = names_to_include
    )

    return merge(data, new_data)
end
