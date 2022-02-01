using DrWatson, ArgParse

include(scriptsdir("utils.jl"))

_usage = """
usage: julia --project=path-to-Epimap mapviz.jl [OPTION]... [TITLE=FILE]...

examples:
  # Single interactive maps using GLMakie.jl.
  julia --project mapviz.jl Rt.csv
  # Multiple interactive maps using GLMakie.jl with different titles.
  julia --project mapviz.jl run1=run1/Rt.csv run2=run2/Rt.csv
  # Save static map to PDF using CairoMakie.jl for a particular date.
  julia --project mapviz.jl --date=2020-12-31 --backend=CairoMakie --out=out.pdf run1=run1/Rt.csv run2=run2/Rt.csv
"""

s = ArgParseSettings(
    description="Visualize generated outputs from Epimap models as maps.",
    usage=_usage
)
add_default_args!(s)
@add_arg_table! s begin
    "file"
    help = "Path(s) to input file(s). If multiple files are provided, maps will plotted horizontally. To specify titles for the different files use the `title=path` syntax for the plots, otherwise the basename of the path will be used as the title. Note also that an empty title is allowed, e.g .`=path`."
    action = :store_arg
    nargs = '*'
    "--geojson"
    help = "Path to GeoJSON file describing the regions."
    default = datadir("uk_lad_boundaries_geo.json")
    "--column"
    help = "Column name used for the colouring of the regions."
    default = "Rt_50"
    "--bounds"
    help = "Boundaries for the value from the dataframe."
    default = (0.0, 2.0)
    eval_arg = true
    "--drop-missing"
    help = "If specified, areas for which we have no data will not be plotted."
    action = :store_true
    "--date"
    help = "If specified, then the date will be fixed and slider dropped. Useful for generating static output. Example: \"2020-12-31\""
    arg_type = Date
    "--out"
    help = "If specified, the figure will be saved to this destination."
    # Visuals
    "--backend"
    help = "Backend to use for Makie. Choices: CairoMakie (best for static), GLMakie (best for interactive, using GPU for rendering), and WGLMakie (web-based)."
    default = :GLMakie
    arg_type = Symbol
    "--stroke-color"
    help = "Color of map boundaries."
    default = :white
    arg_type = Symbol
    "--stroke-width"
    help = "Stroke width used for map boundaries."
    default = 0.05
    arg_type = Float64
    "--title-font-size"
    help = "Font size for the titles."
    default = 10.0
    arg_type = Float64
    "--figure-size"
    help = "Size of the figure."
    default = (1000, 500)
    eval_arg = true
end
args = @parse_args(s)
verbose = args["verbose"]
num_inputs = length(args["file"])
verbose && @info args
@assert num_inputs ≥ 1 "need at least one input file"

bounds = args["bounds"]

# For working with the data.
using CSV, DataFrames, DataFramesMeta

# For plotting.
let backend = args["backend"]
    eval(:(using $(backend); $(backend).activate!()))
end
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

# Renamings of regions.
feature_name_renames = Dict(
    "Cornwall" => "Cornwall and Isles of Scilly"
)

function getname(feature)
    name = GeoInterface.properties(feature)["lad20nm"]
    return haskey(feature_name_renames, name) ? feature_name_renames[name] : name
end

# Select a coordinate projection, using a string that PROJ accepts.
# See e.g. https://proj.org/operations/projections/index.html
source = "+proj=longlat +datum=WGS84"
dest = "+proj=natearth2"
trans = Proj4.Transformation(source, dest, always_xy=true)
ptrans = Makie.PointTrans{2}(trans)

# We first need to figure out the common dates to use.
results = map(args["file"]) do inpath
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
fig = Figure(resolution=args["figure-size"])
display(fig)

# Set up a slider which we can use to specify the target date.
slider = if !isnothing(args["date"])
    # "Fake" observable as a replacement for the observable given
    # by `labelslider!` below.
    let date = args["date"]
        idx = findfirst(==(date), dates)
        if isnothing(idx)
            error("date $(date) provided is not present in the inputs")
        end
        # `Slider` will have a `.value` field, so we replicate this using
        # a simple `NamedTuple`.
        (value = Makie.Observable(idx), )
    end
else
    labelslider = labelslider!(
        fig[1, 1:num_inputs], "Date:", 1:length(dates);
        format=i -> string(dates[i])
    )
    fig[1, 1:num_inputs] = labelslider.layout
    labelslider.slider
end

# If indeed `slider isa Slider`, then we want to put the `Slider` on the top row.
# Otherwise the maps go on top.
map_row_idx = slider isa Slider ? 2 : 1

# TODO: What does `DataAspect` do?
for (i, df) in enumerate(dfs)
    # Load the data.
    df = @linq df |> where(:Date .∈ Ref(dates), :area .∈ Ref(areanames))
    groups = groupby(df, :area)

    ax = Axis(
        fig[map_row_idx, i],
        title=titles[i],
        titlesize=args["title-font-size"],
        aspect=DataAspect(),
        width=Auto(1.0),
    )

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
            val = clamp((R - bounds[1]) / (bounds[2] - bounds[1]), 0.0, 1.0)
            # Multiply by constant in `(0, 1)` to avoid
            # extremal colors.
            b = inv(Bijectors.Logit(0.05, 0.95)) ∘ Bijectors.Logit(0.0, 1.0)
            get(ColorSchemes.balance, b(val))
        end
    end

    # Convert into something we can iterate over and plot independently.
    geobasics = map(GeoMakie.geo2basic, features);
    for (name, color, g) in zip(areanames, colors, geobasics)
        poly!(ax, g, color=color, strokecolor=args["stroke-color"], strokewidth=args["stroke-width"])
    end

    # Plot the "inactive" regions too.
    if !args["drop-missing"]
        for feature in features_inactive
            poly!(ax, GeoMakie.geo2basic(feature), color=:gray, strokecolor=args["stroke-color"], strokewidth=args["stroke-width"])
        end
    end

    # HACK: Fix the limits of the plot.
    limits = let rect = ax.finallimits.val
        Rect2D(ptrans.f(rect.origin), ptrans.f(rect.widths))
    end
    limits!(ax, limits)

    # Let's just remove the axes and the ticklabels.
    hidedecorations!(ax)
    hidespines!(ax)
end
# Add `ColorBar`
# TODO: Make this look nicer. ATM it's way too large.
Colorbar(fig[map_row_idx, num_inputs + 1], limits=bounds, colormap=ColorSchemes.balance, flipaxis=true)

# Finally display the figure.
display(fig)

# # TODO: Show region names, etc. upon hover.
# inspector = DataInspector(fig)

# Save, if specified.
if !isnothing(args["out"])
    save(args["out"], fig)
end
