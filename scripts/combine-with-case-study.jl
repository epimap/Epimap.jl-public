using ArgParse, DrWatson

include(scriptsdir("utils.jl"))

s = ArgParseSettings(
    description="Combine outputs from Epimap.jl and case study 1."
)

@add_arg_table! s begin
    "debiased-path"
    help="path of the Rt-values for the SIR model based on debiased data from case study 1"
    required=true
    "epimap-path"
    help="path of the Rt-values for the Epimap model based on raw daily counts"
    required=true
    "epimap-debiased-path"
    help="path of the Rt-values for the Epimap model based on debiased data from case study 1"
    required=true
    "--out"
    help="if specified, the resulting dataframe will be stored at this path"
end

add_default_args!(s)
args = @parse_args(s)
verbose = args["verbose"]

using DataFrames, CSV

debiased_columns = ["R_l", "R_m", "R_u"]
epimap_columns = ["Rt_2_5", "Rt_50", "Rt_97_5"]

debiased_path = args["debiased-path"]
epimap_path = args["epimap-path"]
epimap_debiased_path = args["epimap-debiased-path"]

df_debiased = DataFrame(CSV.File(debiased_path))
df_epimap = DataFrame(CSV.File(epimap_path))
df_epimap_debiased = DataFrame(CSV.File(epimap_debiased_path))

for df in [df_epimap, df_epimap_debiased]
    transform!(df, Dict(zip(epimap_columns, debiased_columns))...)
    rename!(df, Dict(:area => :ltla, :Date => :mid_week))
end

verbose && @info "Joining dataframes..."
df = innerjoin(
    select(df_debiased, "ltla", "mid_week", map(s -> s => "debiased_$s", debiased_columns)...),
    select(df_epimap, "ltla", "mid_week", map(s -> s => "epimap_$s", debiased_columns)...),
    select(df_epimap_debiased, "ltla", "mid_week", map(s -> s => "epimap_debiased_$s", debiased_columns)...),
    on=[:ltla, :mid_week]
)
verbose && @info "OK!"

if !isnothing(args["out"])
    out = args["out"]
    verbose && @info "Saving resulting dataframe to $(out)"
    CSV.write(out, df)
    verbose && @info "Done!"
else
    # We show the dataframe if no output is desired.
    @show df
end

