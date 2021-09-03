using DrWatson, ArgParse

include(scriptsdir("utils.jl"))

s = ArgParseSettings()
add_default_args!(s)
@add_arg_table! s begin
    "input"
    help = "path to input file"
    action = :store_arg
    nargs = '*'
    "--geojson"
    help = "path to GeoJSON file describing the regions"
    default = datadir("uk_lad_boundaries_geo.json")
    "--column"
    help = "column name used for the colouring of the regions"
    default = "Rt_50"
    "--bounds"
    help = "boundaries for the value from the dataframe"
    default = (0.5, 2.0)
    eval_arg = true
end
args = @parse_args(s)
verbose = args["verbose"]
num_inputs = length(args["input"])
verbose && @info args
@assert num_inputs ≥ 1 "need at least one input file"

bounds = args["bounds"]

# For working with the data.
using Dates
using CSV, DataFrames, DataFramesMeta

# For plotting.
using GLMakie # Alternatives: `CairoMakie`, `WGLMakie`, etc.
using GeoMakie
using GeoMakie: Makie, GeoJSON, GeoInterface
using Proj4
using ColorSchemes: ColorSchemes

using Bijectors

# Function that we use to extract the desired value from the `row`.
# NOTE: `row` will in general be a `DataFrame` so we add a `first` here.
getval(row) = first(getproperty(row, args["column"]))

# Load in the GeoJSON which describes the UK.
geo = GeoJSON.read(read(args["geojson"]));
getname(feature) = GeoInterface.properties(feature)["lad20nm"]

# Select a coordinate projection, using a string that PROJ accepts.
# See e.g. https://proj.org/operations/projections/index.html
source = "+proj=longlat +datum=WGS84"
dest = "+proj=natearth2"
trans = Proj4.Transformation(source, dest, always_xy=true)
ptrans = Makie.PointTrans{2}(trans)

# We first need to figure out the common dates to use.
results = map(args["input"]) do inpath
    if !ispath(inpath)
        title, path = split(inpath, "=")
    else
        title = basename(inpath)
        path = inpath
    end
    return title, path
end
titles = map(Base.Fix2(getindex, 1), results)
inpaths = map(Base.Fix2(getindex, 2), results)

dfs = map(inpaths) do inpath
    DataFrame(CSV.File(inpath))
end
areas_all = map(dfs) do df
    unique(df.area)
end
areanames = sort(intersect(areas_all...))
verbose && @info "Using $(length(areanames)) areas"

dates_all = map(dfs) do df
    unique(df.Date)
end
dates = sort(intersect(dates_all...))
verbose && @info "Using dates from $(dates[1]) to $(dates[end])"

# Set up the figure.
fig = Figure(resolution=(600, 800))
display(fig)

# Set up a slider which we can use to specify the target date.
labelslider = labelslider!(
    fig[1, 1:num_inputs], "Date:", 1:length(dates);
    format=i -> string(dates[i])
)
slider = labelslider.slider
fig[1, 1:num_inputs] = labelslider.layout

# TODO: What does `DataAspect` do?
for (i, df) in enumerate(dfs)
    # Load the data.
    df = @linq df |> where(:Date .∈ Ref(dates), :area .∈ Ref(areanames))
    groups = groupby(df, :area)

    ax = Axis(fig[2, i], title=titles[i], aspect=DataAspect(), width=Auto(1.0))

    # All input data coordinates are projected using `ptrans`.
    # NOTE: The automatic limits for the plot are derived from the
    # untransformed values (probably a bug). So we need to deal with this properly.
    ax.scene.transformation.transform_func[] = ptrans

    # The target date is determined by the `slider`.
    target_date = lift(slider.value) do dateindex
        dates[dateindex]
    end

    # Filter out those not present in `df`.
    features_all = GeoInterface.features(geo);
    length(features_all)

    # Extract the dataframe corresponding to this area.
    present_mask = getname.(features_all) .∈ Ref(areanames)

    features = features_all[present_mask]
    features_inactive = features_all[.~(present_mask)]

    length(features)
    length(features_inactive)

    # For each `feature` we create an `Observable` which selects
    # the row corresponding to `target_date`.
    colors = map(features) do feature
        # Extract the dataframe corresponding to this area.
        name = getname(feature)
        area_df = get(groups, (name, ), nothing)

        lift(target_date) do date
            # Extract the wanted row.
            target_row = @linq area_df |> where(:Date .== date)

            # Median of `Rₜ`
            R = getval(target_row)
            # Normalize wrt. range [0.5, 2.0]
            val = clamp(R / (bounds[2] - bounds[1]), 0.0, 1.0)
            # Multiply by constant in `(0, 1)` to avoid
            # extremal colors.
            # b = inv(Bijectors.Logit(0.05, 0.95)) ∘ Bijectors.Logit(0.0, 1.0)
            get(ColorSchemes.balance, val)
        end
    end

    # Convert into something we can iterate over and plot independently.
    geobasics = map(GeoMakie.geo2basic, features);
    for (name, color, g) in zip(areanames, colors, geobasics)
        poly!(ax, g, color=color, strokecolor=:white, strokewidth=0.05)
    end

    # Plot the "inactive" regions too.
    for feature in features_inactive
        poly!(ax, GeoMakie.geo2basic(feature), color=:gray, strokecolor=:white, strokewidth=0.05)
    end

    # HACK: Fix the limits of the plot.
    limits = let rect = ax.finallimits.val
        Rect2D(ptrans.f(rect.origin), ptrans.f(rect.widths))
    end
    limits = 
    limits!(ax, limits)

    # Let's just remove the axes and the ticklabels.
    hidedecorations!(ax)
    hidespines!(ax)
end
# Add `ColorBar`
# TODO: Make this look nicer. ATM it's way too large.
Colorbar(fig[2, num_inputs + 1], limits=bounds, colormap=ColorSchemes.balance, flipaxis=true)

# Finally display the figure.
display(fig)

# # TODO: Show region names, etc. upon hover.
# inspector = DataInspector(fig)
