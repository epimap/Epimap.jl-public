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

rundir = let tmp = parsed_args["rundir"]
    # Might have been provided with a trailing `/`, which we want to remove.
    isdirpath(tmp) ? dirname(tmp) : tmp
end
verbose && @info "" rundir

# Ensure that we're using the correct version of the package.
if !parsed_args["ignore-commit"]
    interactive_checkout_maybe(rundir)
end

using Printf
using Serialization
using ProgressMeter
using DataFrames, CSV
using Epimap
using TuringUtils
using StatsFuns
using NNlib

# Quantiles we're going to compute.
qs = [0.025, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.975];

intermediatedir(args...) = joinpath(rundir, args...)

figdir(args...) = intermediatedir("figures", args...)
mkpath(figdir())

outdir(args...) = intermediatedir("out", args...)
mkpath(outdir())

# Setup
data = Rmap.load_data();

# Run-related information.
dates = deserialize(intermediatedir("dates.jls"));
args = deserialize(intermediatedir("args.jls"));
T = eltype(args.D)

# Some useful constants.
num_cond = size(args.X_cond, 2)
num_regions = size(args.X_cond, 1)
num_steps = size(args.K_time, 1)

# Need to resolve the names.
debiased = data.debiased
area_names_all = deserialize(intermediatedir("area_names.jls"))
area_names_debiased = unique(debiased[:, :ltla])
area_names = intersect(area_names_all, area_names_debiased)

# Useful to compare against recorded cases.
cases = data.cases
cases = let
    row_mask = cases[:, "Area name"] .∈ Ref(area_names)
    col_mask = names(cases) .∈ Ref(Dates.format.(dates.model, "yyyy-mm-dd"))
    cases[row_mask, col_mask]
end
cases = Array(cases)

# Instantiate model.
m = if "model.jls" ∈ readdir(intermediatedir())
    @info "Loading model from $(intermediatedir())"
    _m = deserialize(intermediatedir("model.jls"))
    # Due to limitations of serialization we need to ensure that we're still
    # working with the correct evaluator.
    # @assert typeof(_m.f) == DynamicPPLUtils.evaluatortype(Rmap.rmap_debiased)
    _m
else
    Rmap.rmap_debiased(
        args.logitπ, args.σ_debias, args.populations,
        args.D, args.W,
        args.F_id, args.F_out, args.F_in,
        args.K_time, args.K_spatial, args.K_local,
        args.days_per_step, args.X_cond_means, args.T;
        ρ_spatial=args.ρ_spatial, ρ_time=args.ρ_time,
        σ_ξ=args.σ_ξ,
    )
end;
logπ, logπ_unconstrained, b, θ_init = @trynumerical Epimap.make_logjoint(m);
binv = inv(b);

# Create example `var_info`.
var_info = @trynumerical DynamicPPL.VarInfo(m);

# Load the samples.
samples = deserialize(intermediatedir("chain.jls"));
adapt_end = findlast(t -> t.stat.is_adapt, samples);
samples_adapt = samples[1:adapt_end];
samples = samples[adapt_end + 1:end];

# Set the converters so we can use `TuringUtils.fast_predict` and
# `TuringUtils.fast_generated_quantities` instead of `predict` and `generated_quantities`.
chain = AbstractMCMC.bundle_samples(samples, var_info, MCMCChains.Chains);
chain = MCMCChainsUtils.setconverters(chain, m);

# # HACK: Probably not needed if we just update TuringUtils.jl
# maybevec(x) = x
# maybevec(x::AbstractArray) = vec(x)

# function DynamicPPLUtils.fast_setval_and_resample_inner!!(vi::DynamicPPL.TypedVarInfo, md, val)
#     vns = md.vns
#     ranges = md.ranges
#     vals = md.vals
#     idcs = md.idcs

#     for vn in vns
#         idx = idcs[vn]
#         r = ranges[idx]

#         if DynamicPPLUtils.hasindex(val, vn)
#             # `_getindex` should be using `view`.
#             vals[r] .= maybevec(DynamicPPL._getindex(val, vn.indexing))
#             DynamicPPL.settrans!!(vi, false, vn)
#         else
#             # Set to be sampled.
#             DynamicPPL.set_flag!!(vi, vn, "del")
#         end
#     end

#     return vi
# end

# Thining used for `predict` and `generated_quantities`.
thin = parsed_args["thin"]

# Predict.
parameters = MCMCChains.get_sections(chain, :parameters);
m_predict = DynamicPPLUtils.replace_args(m, C = missing);
predictions = @trynumerical TuringUtils.fast_predict(m_predict, parameters[1:thin:end]);

# logitπ_pred = reshape(Array(predictions), length(predictions), num_regions, :)
# π_pred = StatsFuns.logistic.(logitπ_pred)
# π_pred_daily = repeat(π_pred, inner=(1, 1, 7))

# # Compute prediction for each weeky by repeating the prevalence across the week
# # and then divinding
# prevalence_pred = π_pred .* reshape(args.populations, 1, :, 1)
# prevalence_pred_daily = repeat(
#     prevalence_pred,
#     inner=(1, 1, 7)
# )

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

results = @trynumerical TuringUtils.fast_generated_quantities(m, parameters[1:thin:end]);

Rs = extract_results(results, :R)
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
            X_cond = m.args.X_cond

            X_full = hcat(X_cond, res.X)
            F = Rmap.compute_flux(args.F_id, args.F_in, args.F_out, β, ρₜ, args.days_per_step)
            Z = Epimap.conv(X_full, args.W)[:, num_cond:end - 1]
            Z̃s[:, :, i] = NNlib.batched_vec(F, Z)

            expected_prevalence[:, :, i] = Epimap.conv(X_full, D)[:, num_cond + 1:end]
        end

        Z̃s, expected_prevalence
    end
end

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

num_regions = length(unique(unique(Xpred[:, :area])))

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
