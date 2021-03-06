using Dates, ArgParse, DrWatson

# Include some utilities.
include(scriptsdir("utils.jl"))

# Argument parsing.
s = ArgParseSettings()
add_default_args!(s)
@add_arg_table! s begin
    "rundir"
    help = "Output directory for the run."
    required = true
    "--thin"
    arg_type = Int
    default = 1
    help = "Thinning interval used for the chain."
end

# Be explicit with `ARGS` so that we can override it in the REPL
# and `include` if we want.
# _args = [
#     "--ignore-commit",
#     "--thin=1",
#     "../intermediate/2021-08-30T05:32:21.054-tor_debiased-1ff44107ef2189b4d84fb77df453b63658ddc7d9"
# ]

parsed_args = @parse_args(s)
verbose = parsed_args["verbose"]
verbose && @info parsed_args

rundir = let tmp = parsed_args["rundir"]
    # Might have been provided with a trailing `/`, which we want to remove.
    isdirpath(tmp) ? dirname(tmp) : tmp
end
verbose && @info rundir

# Ensure that we're using the correct version of the package.
if !parsed_args["ignore-commit"]
    interactive_checkout_maybe(rundir)
    using Pkg; Pkg.instantiate()
end

using Printf
using Serialization
using ProgressMeter
using DataFrames, CSV
using Epimap
using TuringUtils
using StatsFuns
using NNlib
using LinearAlgebra

using Random
Random.seed!(parsed_args["seed"])

# Quantiles we're going to compute.
qs = [0.025, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.975];

intermediatedir(args...) = joinpath(rundir, args...)

figdir(args...) = intermediatedir("figures", args...)
mkpath(figdir())

outdir(args...) = intermediatedir("out", args...)
mkpath(outdir())

# Run-related information.
dates_full = deserialize(intermediatedir("dates.jls"))
args = deserialize(intermediatedir("args.jls"))
area_names_latent = if "area_names_latent.jls" ∉ readdir(intermediatedir())
    deserialize(intermediatedir("area_names.jls"))
else
    deserialize(intermediatedir("area_names_latent.jls"))
end
area_names_observed = if "area_names_observed.jls" ∉ readdir(intermediatedir())
    deserialize(intermediatedir("area_names.jls"))
else
    deserialize(intermediatedir("area_names_observed.jls"))
end
area_names = area_names_latent
T = eltype(args.D)

# Setup
data = Rmap.load_data();
data = Rmap.filter_areas_by_distance(data, area_names; radius=1e-6);
verbose && @info "Working with $(length(area_names)) regions."

# Some useful constants.
num_cond = haskey(args, :X_cond) ? size(args.X_cond, 2) : size(args.X_cond_means, 2)
num_steps = size(args.K_time, 1)

# Instantiate model.
@info "Loading model from $(intermediatedir())"
m = deserialize(intermediatedir("model.jls"));

# TODO: Do this properly.
dates = dates_full

num_regions_latent = length(area_names_latent)
num_regions_observed = length(area_names_observed)
num_regions = num_regions_latent

# Useful to compare against recorded cases.
cases = let cases = data.cases
    col_mask = names(cases) .∈ Ref(Dates.format.(dates.model, "yyyy-mm-dd"))
    Array(cases[:, col_mask])
end

@assert m.name == :rmap "model is not `Rmap.rmap`"
logπ, logπ_unconstrained, b, θ_init = Epimap.make_logjoint(m);
binv = inv(b);
var_info = DynamicPPL.VarInfo(m);

# Load the samples.
samples = deserialize(intermediatedir("chain.jls"));
adapt_end = findlast(t -> t.stat.is_adapt, samples);
samples_adapt = samples[1:adapt_end];
samples = samples[adapt_end + 1:end];

# Set the converters so we can use `TuringUtils.fast_predict` and
# `TuringUtils.fast_generated_quantities` instead of `predict` and `generated_quantities`.
chain = AbstractMCMC.bundle_samples(samples, var_info, MCMCChains.Chains);
chain = MCMCChainsUtils.setconverters(chain, m);

# Thining used for `predict` and `generated_quantities`.
thin = parsed_args["thin"]

# Predict.
parameters = MCMCChains.get_sections(chain, :parameters);
m_predict = DynamicPPLUtils.replace_args(m, C = missing);
predictions = @trynumerical TuringUtils.fast_predict(m_predict, parameters[1:thin:end]);

# Quantiles.
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

subtract_diag(A) = A - Diagonal(A)

# Generated quantities.
print("Computing generated quantities...")

results = @trynumerical TuringUtils.fast_generated_quantities(m, parameters[1:thin:end]);
Rs = extract_results(results, :R)
Xs = extract_results(results, :X)
Xs_cond = permutedims(reshape(Array(MCMCChains.group(chain, :X_cond)), length(chain), num_regions, :), (2, 3, 1))
Zs, Zs_inside, Zs_outside, expected_prevalence = let args = m.args, ρₜs = Array(MCMCChains.group(chain, :ρₜ)), βs = vec(chain[:β]), D = m.args.D
    Z̃s = similar(Xs)
    Z̃s_inside = similar(Xs)
    Z̃s_outside = similar(Xs)
    expected_prevalence = similar(Xs)
    @showprogress 1 "Computing Zs..." for (i, res) in enumerate(results)
        ρₜ = ρₜs[i, :]
        β = βs[i, :]
        X_cond = m.args.X_cond

        X_full = hcat(X_cond, res.X)
        F = Rmap.compute_flux(args.F_id, args.F_in, args.F_out, β, ρₜ, args.days_per_step)
        Z = Epimap.conv(X_full, args.W)[:, num_cond:end - 1]
        Z̃s[:, :, i] = NNlib.batched_vec(F, Z)

        expected_prevalence[:, :, i] = Epimap.conv(X_full, D)[:, num_cond + 1:end]

        # Disentangling external and internal infection pressure.

        # We don't want to include `F_id`, nor the diagonals of `F_in` and `F_out`,
        # so we replace `F_id` with `zeros` and `F_in` and `F_out` with the diagonals
        # dropped.
        F_outside = Rmap.compute_flux(
            zeros(size(args.F_id)),
            subtract_diag(args.F_in),
            subtract_diag(args.F_out),
            β,
            ρₜ,
            args.days_per_step
        )
        Z̃s_outside[:, :, i] = NNlib.batched_vec(F_outside, Z)

        # Treating `F_in` and `F_out` as `Diagonal` means that only
        # the "internal" parts of the fluxes are going to be accounted for.
        F_inside = Rmap.compute_flux(
            args.F_id,
            Diagonal(args.F_in),
            Diagonal(args.F_out),
            β,
            ρₜ, # Disables `F_id`
            args.days_per_step
        )
        Z̃s_inside[:, :, i] = NNlib.batched_vec(F_inside, Z)
    end

    Z̃s, Z̃s_inside, Z̃s_outside, expected_prevalence
end

# Verify that the computation is correct.
if !((Zs_inside + Zs_outside) ≈ Zs)
    @warn "sum internal and external infection pressure does not equal full infection pressure!"
end
Zs_inside_portion = Zs_inside ./ Zs
Zs_outside_portion = Zs_outside ./ Zs

# "Fake" quantities.
Cs = reshape(Array(predictions), length(predictions), num_regions, :)
Cs = permutedims(Cs, (2, 3, 1))
Bs = Cs

# NOTE: It's a bit unclear whether we should be computing the R-value
# from `Xs` and `Zs` or directly from the inferred `π_pred`.
Rs_computed = compute_R(Xs, Zs)
Rs = Rs_computed
# Rs_computed_daily = Xs ./ Zs

println("DONE!")

qs = [0.025, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.975];
@info "Creating dataframes using quantiles $(qs)"
# Construct names in valid format.
names_qs = map(qs) do q
    x = @sprintf("%.03f", q)[3:end]
    return if x[end] == '0'
        x[1:end - 1]
    else
        x
    end
end

Rt_names_qs = map(qs) do q
    x = @sprintf("%.1f", q * 100)
    x = replace(x, "." => "_")
    if x[end] == '0'
        x[1:end - 2]
    else
        x
    end
end

function flatten_by_area(arr::AbstractArray{<:Any, 3})
    return mapreduce(vcat, enumerate(area_names)) do (i, area)
        a = @view arr[i, :, :]
        num_times = size(a, 1)
        hcat(repeat([area], num_times), dates.model[1]:Day(1):dates.model[1] + Day(num_times - 1), a)
    end
end

function make_colnames(basesym, nms)
    return [:area, :Date, map(Base.Fix1(Symbol, basesym), nms)..., :provenance]
end

# (✓) `Xpred.csv`
@info "Computing quantiles for `Xs`, i.e. `Xpred.csv`"
Xs_qs = mapslices(Xs; dims=3) do X
    quantile(X, qs)
end;
Xs_vals = flatten_by_area(Xs_qs);
Xs_nms = make_colnames("X_", names_qs);
Xpred = DataFrame(hcat(Xs_vals, repeat(["inferred"], size(Xs_vals, 1))), Xs_nms);
CSV.write(outdir("Xpred.csv"), Xpred)
Xpred

# (✓) `Zpred.csv`
@info "Computing quantiles for `Zs`, i.e. `Zpred.csv`"
Zs_qs = mapslices(Zs; dims=3) do X
    quantile(X, qs)
end;
Zs_vals = flatten_by_area(Zs_qs);
Zs_nms = make_colnames("Z_", names_qs);
Zpred = DataFrame(hcat(Zs_vals, repeat(["inferred"], size(Zs_vals, 1))), Zs_nms);
CSV.write(outdir("Zpred.csv"), Zpred)
Zpred

# (✓) `Z_inside_pred.csv`
@info "Computing quantiles for `Zs_inside`, i.e. `Z_inside_pred.csv`"
Zs_inside_qs = mapslices(Zs_inside; dims=3) do X
    quantile(X, qs)
end;
Zs_inside_vals = flatten_by_area(Zs_inside_qs);
Zs_inside_nms = make_colnames("Z_", names_qs);
Z_inside_pred = DataFrame(hcat(Zs_inside_vals, repeat(["inferred"], size(Zs_inside_vals, 1))), Zs_inside_nms);
CSV.write(outdir("Z_inside_pred.csv"), Z_inside_pred)
Z_inside_pred

# (✓) `Z_outside_pred.csv`
@info "Computing quantiles for `Zs_outside`, i.e. `Z_outside_pred.csv`"
Zs_outside_qs = mapslices(Zs_outside; dims=3) do X
    quantile(X, qs)
end;
Zs_outside_vals = flatten_by_area(Zs_outside_qs);
Zs_outside_nms = make_colnames("Z_", names_qs);
Z_outside_pred = DataFrame(hcat(Zs_outside_vals, repeat(["inferred"], size(Zs_outside_vals, 1))), Zs_outside_nms);
CSV.write(outdir("Z_outside_pred.csv"), Z_outside_pred)
Z_outside_pred

# (✓) `Z_inside_portion_pred.csv`
@info "Computing quantiles for `Zs_inside_portion`, i.e. `Z_inside_portion_pred.csv`"
Zs_inside_portion_qs = mapslices(Zs_inside_portion; dims=3) do X
    quantile(X, qs)
end;
Zs_inside_portion_vals = flatten_by_area(Zs_inside_portion_qs);
Zs_inside_portion_nms = make_colnames("Z_", names_qs);
Z_inside_portion_pred = DataFrame(hcat(Zs_inside_portion_vals, repeat(["inferred"], size(Zs_inside_portion_vals, 1))), Zs_inside_portion_nms);
CSV.write(outdir("Z_inside_portion_pred.csv"), Z_inside_portion_pred)
Z_inside_portion_pred

# (✓) `Z_outside_portion_pred.csv`
@info "Computing quantiles for `Zs_outside_portion`, i.e. `Z_outside_portion_pred.csv`"
Zs_outside_portion_qs = mapslices(Zs_outside_portion; dims=3) do X
    quantile(X, qs)
end;
Zs_outside_portion_vals = flatten_by_area(Zs_outside_portion_qs);
Zs_outside_portion_nms = make_colnames("Z_", names_qs);
Z_outside_portion_pred = DataFrame(hcat(Zs_outside_portion_vals, repeat(["inferred"], size(Zs_outside_portion_vals, 1))), Zs_outside_portion_nms);
CSV.write(outdir("Z_outside_portion_pred.csv"), Z_outside_portion_pred)
Z_outside_portion_pred

# (✓) `Bpred.csv`
@info "Computing quantiles for `Bs`, i.e. `Bpred.csv`"
Bs_qs = mapslices(Bs; dims=3) do B
    quantile(B, qs)
end;
Bs_vals = flatten_by_area(Bs_qs);
Bs_nms = make_colnames("C_", names_qs)
Bpred = DataFrame(hcat(Bs_vals, repeat(["inferred"], size(Bs_vals, 1))), Bs_nms);
CSV.write(outdir("Bpred.csv"), Bpred)
Bpred
# (✓) `Pexceed.csv`
thresholds = [0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0];
threshold_nms = map(t -> replace(string(t), "." => ""), thresholds)
@info "Computing empirical probabilities of R exceeding thresholds $(thresholds), i.e. `Pexceed.csv`"
ps = mapslices(Rs, dims=3) do R
    mean(r -> r .≥ thresholds, R)
end
Pexceed_vals = flatten_by_area(ps);
Pexceed_nms = make_colnames("P_", threshold_nms)
Pexceed = DataFrame(hcat(Pexceed_vals, repeat(["inferred"], size(Pexceed_vals, 1))), Pexceed_nms);
CSV.write(outdir("Pexceed.csv"), Pexceed)
Pexceed

# (✓) `Cpred.csv`
@info "Computing quantiles for `Cs`, i.e. `Cpred.csv`"
Cs_qs = mapslices(Cs; dims=3) do C
    quantile(C, qs)
end;
Cs_vals = flatten_by_area(Cs_qs);
Cs_nms = make_colnames("C_", names_qs);
Cpred = DataFrame(hcat(Cs_vals, repeat(["inferred"], size(Cs_vals, 1))), Cs_nms);
CSV.write(outdir("Cpred.csv"), Cpred)
Cpred

# Fake projections. TODO: Make them real. Requires support in AR1 process for `missing`.
projection_start = dates.model[end] + Day(1)
num_project = 14
projection_dates = projection_start:Day(1):projection_start + Day(num_project - 1)

Xproj = repeat(Xpred[Xpred[:, :Date] .≥ dates.model[end], :], inner=num_project)
Bproj = repeat(Bpred[Bpred[:, :Date] .≥ dates.model[end], :], inner=num_project)
Cproj = repeat(Cpred[Cpred[:, :Date] .≥ dates.model[end], :], inner=num_project)

@info "" size(Xproj) size(repeat(projection_dates, outer=num_regions))

Xproj[:, :Date] .= repeat(projection_dates, outer=num_regions)
Xproj[:, :provenance] .= "projected"
Cproj[:, :Date] .= repeat(projection_dates, outer=num_regions)
Cproj[:, :provenance] .= "projected"
Bproj[:, :Date] .= repeat(projection_dates, outer=num_regions)
Bproj[:, :provenance] .= "projected"

CSV.write(outdir("Xproj.csv"), Xproj)
CSV.write(outdir("Cproj.csv"), Cproj)
CSV.write(outdir("Bproj.csv"), Bproj)

# (✓) `Rt.csv`
@info "Computing quantiles for `Rs`, i.e. `Rt.csv`"
# Expected format is:
#   Columns: area,Date,qs...,provenance
Rs_qs = mapslices(Rs; dims=3) do R
    quantile(R, qs)
end;

Rs_vals = flatten_by_area(Rs_qs);
Rs_nms = make_colnames("Rt_", Rt_names_qs)
Rt = DataFrame(hcat(Rs_vals, repeat(["inferred"], size(Rs_vals, 1))), Rs_nms)

# Since both predicted and projected Rt-values are combined into the same
# CSV, we need to "fake" some Rt-values for the future.
Rt_proj = repeat(Rt[Rt[:, :Date] .≥ dates.model[end], :], inner=num_project)
Rt_proj[:, :Date] .= repeat(projection_dates, outer=num_regions)
Rt_proj[:, :provenance] .= "projected"

# Add the projected values.
Rt = sort(vcat(Rt, Rt_proj), :area)

# Round off to replicate original results.
Rt[:, 3:end - 1] .= round.(Rt[:, 3:end - 1]; digits=2)

CSV.write(outdir("Rt.csv"), Rt)
Rt
# (✓) `Cweekly.csv`
Cs_with_project = cat(Cs, repeat(Cs[:, end:end, :], inner=(1, num_project, 1)); dims=2)
Csweekly = similar(Cs_with_project, (size(Cs_with_project, 1), size(Cs_with_project, 2) ÷ 7, 1))
for t = 1:7:size(Cs_with_project, 2)
    Csweekly[:, t ÷ 7 + 1, :] .= dropdims(median(sum(Cs_with_project[:, t:t + 6, :]; dims=2); dims=3); dims=2)
end
Csweekly = repeat(Csweekly, inner=(1, 7, 1));
Cweekly_vals = flatten_by_area(Csweekly);
Cweekly_nms = make_colnames("C_", ["weekly"]);
Cweekly = DataFrame(hcat(Cweekly_vals, repeat(["inferred"], size(Cweekly_vals, 1))), Cweekly_nms)

Cweekly[Cweekly[:, :Date] .> dates.model[end], :provenance] .= "projected"

CSV.write(outdir("Cweekly.csv"), Cweekly)
Cweekly

# (✓) `Bweekly.csv`
Bs_with_project = float.(cat(Bs, repeat(Bs[:, end:end, :], inner=(1, num_project, 1)); dims=2))
Bsweekly = similar(Bs_with_project, (size(Bs_with_project, 1), size(Bs_with_project, 2) ÷ 7, 1))
for t = 1:7:size(Bs_with_project, 2)
    Bsweekly[:, t ÷ 7 + 1, :] .= dropdims(median(sum(Bs_with_project[:, t:t + 6, :]; dims=2); dims=3); dims=2)
end
Bsweekly = repeat(Bsweekly, inner=(1, 7, 1));
Bweekly_vals = flatten_by_area(Bsweekly);
Bweekly_nms = make_colnames("C_", ["weekly"]);
Bweekly = DataFrame(hcat(Bweekly_vals, repeat(["inferred"], size(Bweekly_vals, 1))), Bweekly_nms)

Bweekly[Bweekly[:, :Date] .> dates.model[end], :provenance] .= "projected"

CSV.write(outdir("Bweekly.csv"), Bweekly)
Bweekly
