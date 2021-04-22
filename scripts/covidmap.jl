using ArgParse, Dates
using Epimap, Adapt, Zygote, AdvancedHMC
using Epimap.Turing

function ArgParse.parse_item(::Date, date::AbstractString)
    return Date(date, "y-m-d")
end

defaults = Dict(
    :spatialkernel                  => "matern12",
    :temporalkernel                 => "matern12",
    :localkernel                    => "local",
    :globalkernel                   => "global",
    :gp_space_scale                 => 0.5,
    :gp_space_decay_scale           => 0.25,
    :gp_time_scale                  => 50.0,
    :gp_time_decay_scale            => 0.25,
    :fixed_gp_space_length_scale    => 0.1,
    :fixed_gp_time_length_scale     => 100,
    :first_day_modelled             => nothing,
    :last_day_modelled              => nothing,
    :steps_modelled                 => nothing,
    :days_per_step                  => 7,
    :conditioning_days              => 30,
    :days_ignored                   => 0,
    :days_predicted                 => 2,
    :num_steps_forecasted           => 3,
    :thinning                       => 10,
    :chains                         => 1,
    :iterations                     => 3000, 
    :data_directory                 => "/data/ziz/not-backed-up/mhutchin/epimap/epimap-data/processed_data",
    :results_directory              => nothing,
)

s = ArgParseSettings()
@add_arg_table s begin
    
    "--spatialkernel"
        default = defaults[:spatialkernel]
        help = "Use spatial kernel (matern12/matern32/matern52/exp_quad/none); default =" * defaults[:spatialkernel]
    
    "--temporalkernel"
        default = defaults[:temporalkernel]
        help = "Use temporal kernel (matern12/matern32/matern52/exp_quad/none); default =" * defaults[:temporalkernel]
 
    "--localkernel"
        default = defaults[:localkernel]
        help = "Use local kernel (local/none); default =" * defaults[:localkernel]    

    "--globalkernel"
        default = defaults[:globalkernel]
        help = "Use global kernel (global/none); default =" * defaults[:globalkernel]
    
    "--gp_space_scale"
        arg_type = Number
        default = defaults[:gp_space_scale]
        help = "If given and positive, set minimum space length scale to the value; default =" * string(defaults[:gp_space_scale])

    "--gp_time_scale"
        arg_type = Number
        default = defaults[:gp_time_scale]
        help = "If given and positive, set minimum time length scale to the value; default =" * string(defaults[:gp_time_scale])

    "--fixed_gp_space_length_scale"
        arg_type = Number
        default = defaults[:fixed_gp_space_length_scale]
        help = "If given and positive, fix the space length scale to the value; default =" * string(defaults[:fixed_gp_space_length_scale])

    "--fixed_gp_time_length_scale"
        arg_type = Number
        default = defaults[:fixed_gp_time_length_scale]
        help = "If given and positive, fix the time length scale to the value; default =" * string(defaults[:fixed_gp_time_length_scale])
  
    # "-m", "--metapop"
    #     default = defaults[:metapop]
    #     help = "metapopulation model for inter-region cross infections" *
    #             "none, or comma separated list containing radiation{1,2,3},{alt_}traffic_{forward,reverse},uniform,in,in_out);" *
    #             "default = " * defaults[:metapop]
    
    # "-v", "--observation_data"
    #     default = defaults[:observation_data]
    #     help = "observation values to use in the model" *
    #             "(counts/clatent_mean/clatent_sample/clatent_recon/latent_reports);" *
    #             "default =" * defaults[:observation_data]
    
    # "-o", "--observation_model"
    #     default = defaults[:observation_model]
    #     hel = "observation model",
    #             "(poisson/neg_binomial_{2,3}/gaussian);",
    #             "default =" * defaults[:observation_model]
    
    "--chains"
        arg_type = Int
        default = defaults[:chains]
        help = "number of MCMC chains; default =" * string(defaults[:chains])
    
    "--iterations"
        arg_type = Int
        default = defaults[:iterations]
        help = "Length of MCMC chains; defualt =" * string(defaults[:iterations])
    
    "--first_day_modelled"
        arg_type=Date
        default = defaults[:first_day_modelled]
        help = "Date of first day to model; default =" * string(defaults[:first_day_modelled])
    
    "--steps_modelled"
        arg_type = Int
        default = defaults[:steps_modelled]
        help = "Number of steps to model; default =" * string(defaults[:steps_modelled])
    
    "--days_per_step"
        arg_type = Int
        default = defaults[:days_per_step]
        help = "Number of days per step modelled; default = " * string(defaults[:days_per_step])
    
    "--last_day_modelled"
        arg_type=Date
        default = defaults[:last_day_modelled]
        help = "Date of last day to model; default =" * string(defaults[:last_day_modelled])

    "--conditioning_days"
        arg_type = Int
        default= defaults[:conditioning_days]
        help = "Number of previous days to condition on using MAP deconvolution of the observed cases; default = " * defaults[:conditioning_days]
    
    "--days_ignored"
        arg_type = Int
        default = defaults[:days_ignored]
        help = "Days ignored; default =" * string(defaults[:days_ignored])
    
    "--days_predicted"
        arg_type = Int
        default = defaults[:days_predicted]
        help = "Days predicted; default =" * string(defaults[:days_predicted])
    
    "--num_steps_forecasted"
        arg_type = Int
        default = defaults[:num_steps_forecasted]
        help = "Days predicted; default =" * string(defaults[:num_steps_forecasted])
    
    "--results_directory"
        default = defaults[:results_directory]
        help = "If specified, store outputs in directory, otherwise use a unique directory"
    
    "--data_directory"
        default = defaults[:data_directory]
        help="Directory from which to load data about regions and raw cases"
    
end

# parsed_args = parse_args(s)
parsed_args = defaults
println(parsed_args)

# default for testing
parsed_args[:first_day_modelled] = Date("2020-10-01", "y-m-d")
parsed_args[:steps_modelled] = 15

data = Rmap.load_data("file://" * parsed_args[:data_directory])

default_args = (
    ρ_spatial = parsed_args["fixed_gp_space_length_scale"],
    ρ_time = parsed_args["fixed_gp_time_length_scale"],
    σ_spatial = 0.1,
    σ_local = 0.1,
    σ_ξ = 1.0,
)

setup_args = merge(Rmap.setup_args(
    Rmap.rmap_naive, 
    data, 
    first_day_modelled = parsed_args["first_day_modelled"],
    last_day_modelled = parse_args["last_day_modelled"], 
    steps_modelled = parsed_args["steps_modelled"], 
    days_per_step = parsed_args["days_per_step"],
    conditioning_days = parsed_args["conditioning_days"]
), default_args);

m = Rmap.rmap_naive(setup_args...);
logπ = Epimap.make_logjoint(Rmap.rmap_naive, setup_args...)

# sample
samples = ...

# generate outputs
outputs = ...

# postprocess and save results
