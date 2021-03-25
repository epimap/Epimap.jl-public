using DrWatson, CSV, DataFrames, RemoteFiles, UnPack, KernelFunctions, LinearAlgebra, PDMats


function Base.map(f, d::Dict)
    pairs = map(f, zip(keys(d), values(d)))
    return Dict(pairs)
end

# TODO: make this nicer
function load_data(rmap_path = "file://" * joinpath(ENV["HOME"], "Projects", "private", "Rmap", "data"))
    # Download files if not present
    @RemoteFileSet datasets "Rmap data" begin
        cases = @RemoteFile "$(rmap_path)/cases.csv" dir=datadir("rmap")
        areas = @RemoteFile "$(rmap_path)/areas.csv" dir=datadir("rmap")
        serial_intervals = @RemoteFile "$(rmap_path)/serial_interval.csv" dir=datadir("rmap")
        traffic_flux_in = @RemoteFile "$(rmap_path)/uk_reverse_commute_flow.csv" dir=datadir("rmap")
        traffic_flux_out = @RemoteFile "$(rmap_path)/uk_forward_commute_flow.csv" dir=datadir("rmap")
    end

    # Download files if out of date
    download(datasets)

    # Load files
    dataframes = map(datasets.files) do (name, remotefile)
        (name, DataFrame(CSV.File(path(remotefile))))
    end

    # Find the areas that we have all kinds of data for
    valid_areas = Set(dataframes[:areas][:, :area])
    df2areacol = Dict{Symbol, String}()

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
        df = dataframes[name]
        mask = ∈(valid_areas).(df[:, colname])
        dataframes[name] = df[mask, :]
    end

    # Convert into `NamedTuple` because nicer to work with
    data = DrWatson.dict2ntuple(dataframes)

    # Perform certain transformations
    data.cases[!, 3:end] = convert.(Int, data.cases[:, 3:end])

    return data
end


function setup_args(
    ::typeof(rmap_naive),
    data,
    args...;
    days_per_step = 1,
    infection_cutoff = 30,
    test_delay_days = 21,
    presymptomdays = 2,
    kwargs...
)
    (days_per_step != 1) && @warn "setting `days_per_step` to ≠ 1 has no effect at the moment"

    @unpack cases, areas, serial_intervals, traffic_flux_in, traffic_flux_out = data

    # Convert `cases` into a matrix, removing the area-columns
    cases = Array(cases[:, Not(["Country", "Area name"])])
    (num_regions, num_days) =  size(cases)

    # TODO: should we "sub-sample" the infection and test delay profiles to account for `days_per_step`?

    # Serial intervals / infection profile
    serial_intervals = serial_intervals[1:min(infection_cutoff, size(serial_intervals, 1)), :fit]
    normalize!(serial_intervals, 1) # re-normalize wrt. ℓ1-norm to ensure still sums to 1
    @assert sum(serial_intervals) ≈ 1.0 "truncated serial_intervals does not sum to 1"

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
    K_time = PDMat(kernelmatrix(k_time, 1:days_per_step:num_days))

    # Flux matrices
    F_id = Diagonal(ones(num_regions))
    F_out = Array(traffic_flux_out[1:end, 2:end])
    F_in = Array(traffic_flux_in[1:end, 2:end])

    # Resulting arguments
    return (
        C = cases,
        D = test_delay_profile,
        W = serial_intervals,
        F_id = F_id,
        F_out = F_out,
        F_in = F_in,
        K_time = K_time,
        K_spatial = K_spatial,
        K_local = K_local
    )
end
