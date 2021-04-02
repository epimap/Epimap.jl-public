using Epimap
using Epimap.Turing


@testset "Rmap" begin
    data = let rmapdir = get(ENV, "EPIMAP_RMAP_DATADIR", "")
        if !isempty(rmapdir)
            Rmap.load_data(rmapdir)
        else
            Rmap.load_data()
        end
    end

    @testset "filter_areas_by_distance" begin
        # If we allow number of regions to be all of them, then we should recover
        # the original dataset perfectly.
        epidemic_start = 241
        num_regions = size(data.cases, 1)
        filtered_data = Rmap.filter_areas_by_distance(
            data;
            num_main_regions = num_regions,
            epidemic_start = epidemic_start,
            radius = Inf
        )

        for k in keys(data)
            @test filtered_data[k] == data[k]
        end

        # Verify that the radius specified is respected.
        # Manchester and Birmingham are slightly more than 112km apart.
        radius = 1.12100
        filtered_data = Rmap.filter_areas_by_distance(
            data, ["Manchester"];
            radius = radius
        )
        @test "Birmingham" ∉ filtered_data.area_names

        radius = 1.12300
        filtered_data = Rmap.filter_areas_by_distance(
            data, ["Manchester"];
            radius = radius
        )
        @test "Birmingham" ∈ filtered_data.area_names
    end

    @testset "model" begin
        rng = StableRNG(42);
        num_repeats = 100

        data = Rmap.filter_areas_by_distance(
            data, "Manchester",
            radius=0.11
        )

        # Construct the model arguments from data
        setup_args = Rmap.setup_args(Rmap.rmap_naive, data; num_cond = 10)

        # Arguments not related to the data which are to be set up
        default_args = (
            ρ_spatial = 10.0,
            ρ_time = 0.1,
            σ_spatial = 0.1,
            σ_local = 0.1,
            σ_ξ = 1.0
        )

        args = merge(setup_args, default_args)

        # Instantiate model
        m = Rmap.rmap_naive(args...);

        # `make_logjoint`
        logπ = Epimap.make_logjoint(Rmap.rmap_naive, args...)

        # Verify that they have received the same arguments
        # Remove the type-parameters from the model
        for k in filter(k -> !(m.args[k] isa Type), keys(m.args))
            @test m.args[k] == getfield(logπ, k)
        end

        # Check average difference
        spl = DynamicPPL.SampleFromPrior()
        results = []

        for i = 1:num_repeats
            var_info = DynamicPPL.VarInfo(rng, m);
            θ = var_info[spl]

            # Get something we can pass to `make_logjoint`
            num_regions = size(data.cases, 1);
            θ_nt = map(DynamicPPL.tonamedtuple(var_info)) do (v, ks)
                if startswith(string(first(ks)), "X")
                    # Add back in the first column since it's not inferred
                    reshape(v, (num_regions, :))
                elseif length(v) == 1
                    first(v)
                else
                    v
                end
            end

            diff = DynamicPPL.getlogp(var_info) ≈ logπ(θ_nt)
            push!(results, abs(diff))
        end

        # Pretty "high" `atol` since we're in log-space + it's likely that `logπ` is numerically more accurate
        @test mean(results) ≤ 10
    end
end
