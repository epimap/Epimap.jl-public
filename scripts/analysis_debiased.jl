using DrWatson
using ArgParse

# Include some utilities.
include(scriptsdir("utils.jl"))

s = ArgParseSettings()
add_default_args!(s)
@add_arg_table! s begin
    "path"
    help = "path to the run"
    # required = true
    "--list"
    help = "If specified, the available runs will be listed."
    action = :store_true
    # NOTE: We could also make a custom `parse_arg` for this, but it means that
    # we have to do `using Epimap` before `parse_args` which can be annoying.
    # For example if we just want to `--list` the available runs, we don't want to
    # wait until `Epimap` has compiled to do so.
    "--thin"
    help = "Specifies the thinning to use for `predict`and `generated_quantities`."
    default = 1
    arg_type = Int
    "--area"
    help = "Additional areas to produce plots for."
    action = :append_arg
end

parsed_args = @parse_args(s)
rundir = parsed_args["path"]
thin = parsed_args["thin"]
verbose = parsed_args["verbose"]
verbose && @info parsed_args

if parsed_args["list"]
    let d = projectdir("intermediate")
        println("The following runs are available in $d:")
        for x in readdir(d)
            println("- ", x)
        end
    end
    exit(0)
end

# Ensure that we're using the correct version of the package.
if !parsed_args["ignore-commit"]
    interactivate_checkout_maybe(rundir)
end

# Now we're at a point where we know we're going to actual run the code
# so let's `using` everything.
using Epimap
using AdvancedHMC, Zygote
using DataFrames, CSV
using Dates
using ComponentArrays
using Adapt
using TuringUtils
using Serialization
using StatsPlots
using StatsFuns
using NNlib

pyplot()

# Some useful methods for resolving the paths.
intermediatedir(args...) = projectdir("intermediate", rundir, args...)
figdir(args...) = intermediatedir("figures", args...)
mkpath(figdir())
outdir(args...) = intermediatedir("out", args...)
mkpath(outdir())

println("All outputs of this script can be found in $(outdir())")

# Run-related information.
dates_full = deserialize(intermediatedir("dates.jls"))
args = deserialize(intermediatedir("args.jls"))
area_names = deserialize(intermediatedir("area_names.jls"))
T = eltype(args.D)

# Setup
data = Rmap.load_data();
data = Rmap.filter_areas_by_distance(data, area_names; radius=1e-6);
verbose && @info "Working with $(length(area_names)) regions."

# Some useful constants.
num_cond = haskey(args, :X_cond) ? size(args.X_cond, 2) : size(args.X_cond_means, 2)
num_regions = size(args.K_spatial, 1)
num_steps = size(args.K_time, 1)

# Instantiate model.
@info "Loading model from $(intermediatedir())"
m = deserialize(intermediatedir("model.jls"));

# TODO: Do this properly.
dates = (
    condition = dates_full.condition,
    # Cut of the initial period which we did not observe on.
    model = dates_full.model[m.args.skip_weeks_observe * 7 + 1:end]
)

# Useful to compare against recorded cases.
cases = let cases = data.cases
    col_mask = names(cases) .∈ Ref(Dates.format.(dates.model, "yyyy-mm-dd"))
    Array(cases[:, col_mask])
end

@assert m.name == :rmap_debiased "model is not `Rmap.rmap_debiased`"
logπ, logπ_unconstrained, b, θ_init = Epimap.make_logjoint(m);
binv = inv(b);
var_info = DynamicPPL.VarInfo(m);

# Load the samples.
samples = deserialize(intermediatedir("chain.jls"));
adapt_end = findlast(t -> t.stat.is_adapt, samples);
samples_adapt = samples[1:adapt_end];
samples = samples[adapt_end + 1:end];

# Convert into a more usual format.
chain = AbstractMCMC.bundle_samples(samples, var_info, MCMCChains.Chains);
chain = MCMCChainsUtils.setconverters(chain, m);

# Statistics
# ESS
print("Computing and visualizing the ESS...")

essdf = DataFrame(MCMCChains.ess(chain));
essdf = sort(essdf, :ess);
CSV.write(outdir("ess.csv"), essdf);

histogram(essdf[:, :ess], label="")
title!("Histogram of ESS")
savefig(figdir("ess.pdf"))

println("DONE!")

# Sampler internals
print("Visualizing sampler statistics...")

internals = filter(chain.name_map.internals) do name
    # This might sometimes be ∞ which causes issues in the plotting, so we just remove.
    name != :max_hamiltonian_error
end

plot(MCMCChains.get_sections(chain, :internals)[internals])
savefig(figdir("internals.pdf"))

println("DONE!")

# Selected variables
print("Visualizing some traces...")

plot(chain[[:σ_local, :σ_spatial]])
savefig(figdir("traceplot-gp-parameters.pdf"))

plot(MCMCChains.group(chain, :ρₜ))
savefig(figdir("traceplot-rho.pdf"))

plot(chain[[:β, :ψ, :μ_ar, :σ_ar, :α_pre]])
savefig(figdir("traceplot-univariate-parameters.pdf"))

println("DONE!")

# Predictive posterior
print("Predicting...")

parameters = MCMCChains.get_sections(chain, :parameters);
m_predict = DynamicPPLUtils.replace_args(m, logitπ=missing);
predictions = TuringUtils.fast_predict(m_predict, parameters[1:thin:end])

println("DONE!")

# TODO: Generalize.
logitπ_true = m.args.logitπ
π_true = StatsFuns.logistic.(logitπ_true)
logitπ_pred = reshape(Array(predictions), length(predictions), num_regions, :)
π_pred = StatsFuns.logistic.(logitπ_pred)
π_pred_daily = repeat(π_pred, inner=(1, 1, 7))

# Compute prediction for each weeky by repeating the prevalence across the week
# and then divinding
prevalence_true = π_true .* args.populations
prevalence_true_daily = repeat(prevalence_true, inner=(1, 7))
prevalence_pred = π_pred .* reshape(args.populations, 1, :, 1)
prevalence_pred_daily = repeat(
    prevalence_pred,
    inner=(1, 1, 7)
)

function plot_density_wrt_time(ts, xs; Δ=0.005, lb=0.025, ub=0.975, kwargs...)
    p = plot(; kwargs...)
    return plot_density_wrt_time!(p, ts, xs; Δ, lb, ub, kwargs...)
end

function plot_density_wrt_time!(p, ts, xs; Δ=0.005, lb=0.025, ub=0.975, kwargs...)
    ps = lb:Δ:ub
    qs = mapreduce(vcat, xs) do x
        adjoint(quantile(x, ps))
    end

    prob_accounted_for = 0.0
    midpoint = length(ps) ÷ 2 + 1
    for i = reverse(1:midpoint - 1)
        median = qs[:, midpoint]
        lb = median - qs[:, midpoint - i]
        ub = qs[:, midpoint + i] - median
        prob_outside = (1 - ps[midpoint + i]) + ps[midpoint - i]
        Δprob = prob_outside - prob_accounted_for
        plot!(
            ts,
            qs[:, midpoint],
            ribbon=(lb, ub),
            linewidth=0,
            fillalpha=Δprob,
            color=:blue,
            label=""
        )
        prob_accounted_for += Δprob
    end

    return p
end

function compute_R(Xs, Zs)
    strided = let X = Xs ./ Zs, step = args.days_per_step
        (@views(X[:, start:start + step - 1, :] for start = 1:step:size(X,2)))
    end;

    Rs = mapreduce((acc, x) -> cat(acc, x; dims=2), strided) do x
        mean(x; dims=2)
    end;
    Rs = repeat(Rs, inner=(1, args.days_per_step, 1))

    return Rs
end

function extract_results(results, sym)
    s = size(getfield(results[1], sym))
    Xs = zeros(s..., length(results))

    index_prefix = map(_ -> Colon(), 1:length(s))
    for i = 1:length(results)
        Xs[index_prefix..., i] = getfield(results[i], sym)
    end

    return Xs
end

# Generated quantities.
print("Computing generated quantities...")

results = fast_generated_quantities(m, parameters[1:thin:end]);
Rs = extract_results(results, :R)

Xs_cond = permutedims(reshape(Array(MCMCChains.group(chain, :X_cond)), length(chain), num_regions, :), (2, 3, 1))
Xs_cond = m.args.populations .* repeat(Xs_cond ./ num_cond, inner=(1, num_cond, 1))

Xs = extract_results(results, :X)
Zs, expected_prevalence = if haskey(results[1], :Z)
    extract_results(results, :Z), extract_results(results, :expected_prevalence)
else
    let args = m.args, ρₜs = Array(MCMCChains.group(chain, :ρₜ)), βs = vec(chain[:β]), D = m.args.D
        Z̃s = similar(Xs)
        expected_prevalence = similar(Xs)
        for (i, res) in enumerate(results)
            ρₜ = ρₜs[i, :]
            β = βs[i, :]
            X_cond = Xs_cond[:, :, i]

            X_full = hcat(X_cond, res.X)
            F = Rmap.compute_flux(args.F_id, args.F_in, args.F_out, β, ρₜ, args.days_per_step)
            Z = Epimap.conv(X_full, args.W)[:, num_cond:end - 1]
            Z̃s[:, :, i] = NNlib.batched_vec(F, Z)

            expected_prevalence[:, :, i] = Epimap.conv(X_full, D)[:, num_cond + 1:end]
        end

        Z̃s, expected_prevalence
    end
end

Rs_full = Rs
Xs_full = Xs
Zs_full = Zs

Rs = Rs[:, m.args.skip_weeks_observe * 7 + 1:end, :]
Xs = Xs[:, m.args.skip_weeks_observe * 7 + 1:end, :]
Zs = Zs[:, m.args.skip_weeks_observe * 7 + 1:end, :]
expected_prevalence = expected_prevalence[:, m.args.skip_weeks_observe * 7 + 1:end, :]


# NOTE: It's a bit unclear whether we should be computing the R-value
# from `Xs` and `Zs` or directly from the inferred `π_pred`.
Rs_computed_full = compute_R(Xs_full, Zs_full)
Rs_computed_daily_full = Xs_full ./ Zs_full

Rs_computed = compute_R(Xs, Zs)
Rs_computed_daily = Xs ./ Zs
Rs_computed_prevalence = repeat(
    permutedims(
        cat(
            ones(size(π_pred)[1:2]..., 1),
            π_pred[:, :, 2:end] ./ π_pred[:, :, 1:end-1];
            dims=3
        ),
        (2, 3, 1)
    ),
    inner=(1, 7, 1)
)

println("DONE!")

verbose && @info "" size(Rs) size(Xs) size(Zs) size(Rs_computed) size(Rs_computed_daily) size(prevalence_pred_daily) size(cases)

# Visualize the posterior predictive.
function plot_posterior_predictive(ts; area=nothing, area_name=nothing, start_idx=1)
    @assert !isnothing(area) || !isnothing(area_name)

    area = !isnothing(area) ? area : findfirst(==(area_name), area_names)
    area_name = !isnothing(area_name) ? area_name : area_names[area]

    # Drop the initial indices if we're not going to use them.
    ts = ts[start_idx:end]

    # Plot!
    ps = []

    num_samples = size(prevalence_pred_daily, 1)

    p1 = plot()
    plot!(p1, ts, transpose(prevalence_pred_daily[:, area, start_idx:end]), label="", alpha= 1 / num_samples^(3/5), color=:blue)
    plot!(p1, ts, prevalence_true_daily[area, start_idx:end], color=:black, label="true")
    ylabel!("Prevalence", labelfontsize=10)
    push!(ps, p1)
    title!(area_name)

    p2 = plot()
    plot_density_wrt_time!(p2, ts, eachrow(Rs[area, start_idx:end, :]))
    ylabel!("Rt (latent)", labelfontsize=10)
    push!(ps, p2)

    p3 = plot()
    plot_density_wrt_time!(p3, ts, eachrow(Rs_computed[area, start_idx:end, :]))
    ylabel!("Rt (i.p.)", labelfontsize=10)
    push!(ps, p3)

    p8 = plot()
    plot_density_wrt_time!(p8, ts, eachrow(Rs_computed_daily[area, start_idx:end, :]))
    ylabel!("Rt daily (i.p.)", labelfontsize=10)
    push!(ps, p8)

    p7 = plot()
    plot_density_wrt_time!(p7, ts, eachrow(Rs_computed_prevalence[area, start_idx:end, :]))
    ylabel!("Growth rate (prev)", labelfontsize=10)
    push!(ps, p7)

    p4 = plot()
    plot!(p4, ts, repeat(logitπ_true[area, :], inner=7)[start_idx:end], color=:black, label="")
    plot_density_wrt_time!(p4, ts, eachcol(repeat(logitπ_pred[:, area, :], inner=(1, 7))[:, start_idx:end]))
    ylabel!("logitπ", labelfontsize=10)
    push!(ps, p4)

    p5 = plot()
    plot!(ts, cases[area, start_idx:end], color=:black, label="Recorded cases")
    plot_density_wrt_time!(p5, ts, eachrow(Xs[area, start_idx:end, :]))
    ylabel!("Xₜ", labelfontsize=10)
    push!(ps, p5)

    p6 = plot()
    plot_density_wrt_time!(p6, ts, eachrow(Zs[area, start_idx:end, :]))
    ylabel!("Z̃ₜ", labelfontsize=10)
    push!(ps, p6)

    p6 = plot()
    plot_density_wrt_time!(p6, ts, eachrow(expected_prevalence[area, start_idx:end, :]))
    ylabel!("Expected prevalence", labelfontsize=10)
    push!(ps, p6)

    return plot(ps..., layout=(length(ps), 1), size=(1000, length(ps) * 200))
end

let area_with_most_cases = argmax(vec(sum(cases; dims=2)))
    area_name = area_names[area_with_most_cases]
    plot_posterior_predictive(dates.model, area_name=area_name)
    savefig(figdir("$(area_name)-posterior-predictive.pdf"))

    plot_posterior_predictive(dates.model, area_name=area_name, start_idx=8)
    savefig(figdir("$(area_name)-posterior-predictive-ignore-first-week.pdf"))
end

let area_name = "Cambridge"
    plot_posterior_predictive(dates.model, area_name=area_name)
    savefig(figdir("$(area_name)-posterior-predictive.pdf"))
    
    plot_posterior_predictive(dates.model, area_name=area_name, start_idx=8)
    savefig(figdir("$(area_name)-posterior-predictive-ignore-first-week.pdf"))
end

let area_name = "Craven"
    plot_posterior_predictive(dates.model, area_name=area_name)
    savefig(figdir("$(area_name)-posterior-predictive.pdf"))
    
    plot_posterior_predictive(dates.model, area_name=area_name, start_idx=8)
    savefig(figdir("$(area_name)-posterior-predictive-ignore-first-week.pdf"))
end

for area_name in area_names[1:10]
    plot_posterior_predictive(dates.model, area_name=area_name)
    savefig(figdir("$(area_name)-posterior-predictive.pdf"))

    plot_posterior_predictive(dates.model, area_name=area_name, start_idx=8)
    savefig(figdir("$(area_name)-posterior-predictive-ignore-first-week.pdf"))
end

for area_name in parsed_args["area"]
    plot_posterior_predictive(dates.model, area_name=area_name)
    savefig(figdir("$(area_name)-posterior-predictive.pdf"))

    plot_posterior_predictive(dates.model, area_name=area_name, start_idx=8)
    savefig(figdir("$(area_name)-posterior-predictive-ignore-first-week.pdf"))
end
