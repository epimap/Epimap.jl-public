using DrWatson, ArgParse

s = ArgParseSettings()
@add_arg_table! s begin
    "input"
    help = "path to input file"
    required = true
    "--geojson"
    help = "path to GeoJSON file describing the regions"
    default = datadir("uk_lad_boundaries_geo.json")
end

if !(@isdefined(_args))
    _args = ARGS
end

args = parse_args(_args, s)


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

# Load the data.
df = DataFrame(CSV.File(args["input"]))
areanames = unique(df.area)
dates = unique(df.Date)
groups = groupby(df, :area)

# Function that we use to extract the desired value from the `row`.
# NOTE: `row` will in general be a `DataFrame` so we add a `first` here.
getval(row) = first(row.Rt_50)

# Load in the GeoJSON which describes the UK.
geo = GeoJSON.read(read(args["geojson"]));
getname(feature) = GeoInterface.properties(feature)["lad20nm"]

# Select a coordinate projection, using a string that PROJ accepts.
# See e.g. https://proj.org/operations/projections/index.html
source = "+proj=longlat +datum=WGS84"
dest = "+proj=natearth2"
trans = Proj4.Transformation(source, dest, always_xy=true)
ptrans = Makie.PointTrans{2}(trans)

# Set up the figure.
fig = Figure(resolution=(600, 800))
display(fig)

# Set up a slider which we can use to specify the target date.
labelslider = labelslider!(
    fig[1, 1], "Date:", 1:length(dates);
    format=i -> string(dates[i])
)
slider = labelslider.slider
fig[1, 1] = labelslider.layout

# TODO: What does `DataAspect` do?
ax = Axis(fig[2, 1], aspect=DataAspect(), width=Auto(1.0))

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
        val = clamp(R / (2 - 0.5), 0.0, 1.0)
        # Multiply by constant in `(0, 1)` to avoid
        # extremal colors.
        b = inv(Bijectors.Logit(0.05, 0.95)) ∘ Bijectors.Logit(0.0, 1.0)
        get(ColorSchemes.balance, b(val))
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
limits!(ax, limits)

# Let's just remove the axes and the ticklabels.
hidedecorations!(ax)
hidespines!(ax)

# Add `ColorBar`
# TODO: Make this look nicer. ATM it's way too large.
# Colorbar(fig[2, 3], limits=(0.5, 2.0), ticks=[0.5, 1.0, 1.5, 2.0], colormap=ColorSchemes.balance, flipaxis=true)

# Finally display the figure.
display(fig)

# # TODO: Show region names, etc. upon hover.
# inspector = DataInspector(fig)
