@testset "Rmap" begin
    data = let rmapdir = get(ENV, "EPIMAP_RMAP_DATADIR", "")
        if !isempty(rmapdir)
            Rmap.load_data(rmapdir)
        else
            get_test_data(Rmap.rmap_naive)
        end
    end

    function make_default_args(
        data, T = Float64;
        num_steps = 10,
        num_condition_days = 3,
        timestep = Day(1),
        condition_observations = true
    )
        # Construct the model arguments from data
        setup_args = Rmap.setup_args(
            Rmap.rmap_naive, data, T;
            num_condition_days = num_condition_days,
            num_steps = num_steps,
            timestep = timestep,
            condition_observations = condition_observations
        )

        # Arguments not related to the data which are to be set up
        default_args = (
            ρ_spatial = 10.0,
            ρ_time = 0.1,
            σ_spatial = 0.1,
            σ_local = 0.1,
            σ_ξ = 1.0
        )
        args = merge(setup_args, default_args)

        return adapt(Epimap.FloatMaybeAdaptor{T}(), args)
    end

    @testset "setup_args" begin
        # Make sure that the sizes and whatnot seem reasonable.
        num_steps = 10
        num_condition_days = 3
        args = make_default_args(
            data;
            num_condition_days = num_condition_days,
            num_steps = num_steps
        )
        @test size(args.X_cond, 2) == num_condition_days
        @test size(args.C, 2) == num_condition_days + num_steps

        # If `condition_observation`, we will simply make `X_cond` the same
        # as the number of cases on the conditioning days.
        num_condition_days = 3
        args = make_default_args(
            data;
            num_condition_days = num_condition_days,
            condition_observations = true
        )
        @test size(args.X_cond, 2) == num_condition_days
        @test size(args.C, 2) == num_condition_days + num_steps
        @test args.X_cond == args.C[:, 1:num_condition_days]

        # If not `condition_observations` and we have a sufficient number of
        # conditioning steps, e.g. > 10, we'll compute an estimate which won't
        # be equal to the number of cases during those days.
        num_steps = 20
        num_condition_days = 10
        args = make_default_args(
            data;
            num_steps = num_steps,
            num_condition_days = num_condition_days,
            condition_observations = false
        )
        @test size(args.X_cond, 2) == num_condition_days
        @test size(args.C, 2) == num_condition_days + num_steps
        @test args.X_cond != args.C[:, 1:num_condition_days]
    end

    @testset "filter_areas_by_distance" begin
        # If we allow number of regions to be all of them, then we should recover
        # the original dataset perfectly.
        epidemic_start = 10
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
        radius = 0.105
        filtered_data = Rmap.filter_areas_by_distance(
            data, ["Manchester"];
            radius = radius
        )
        @test "Trafford" ∉ filtered_data.area_names

        radius = 0.11
        filtered_data = Rmap.filter_areas_by_distance(
            data, ["Manchester"];
            radius = radius
        )
        @test "Trafford" ∈ filtered_data.area_names
    end

    @testset "model" begin
        for T ∈ [Float32, Float64]
            adaptor = Epimap.FloatMaybeAdaptor{T}()
            # Use different threshold if `Float32` since we're comparing to use of `Float64`.
            threshold = (T === Float32) ? 5 : 1
            num_repeats = 10

            # Instantiate model
            args = make_default_args(data, T)
            m = Rmap.rmap_naive(args...);

            # `make_logjoint`
            logπ, logπ_unconstrained, b, θ_init = Epimap.make_logjoint(
                Rmap.rmap_naive,
                args...,
                Matrix{T}
            )

            # Verify that they have received the same arguments
            # Remove the type-parameters from the model
            for k in filter(k -> !(m.args[k] isa Type), keys(m.args))
                @test m.args[k] == getfield(logπ, k)
            end

            # Check average difference
            spl = DynamicPPL.SampleFromPrior()

            for i = 1:num_repeats
                # Constrained space
                var_info = DynamicPPL.VarInfo(m);
                θ = var_info[spl]
                m(var_info)

                θ_ca = adapt(adaptor, ComponentArray(var_info))
                @test abs(DynamicPPL.getlogp(var_info) - logπ(θ_ca)) ≤ threshold

                # Unconstrained space
                DynamicPPL.link!(var_info, spl, Val(keys(θ_ca)))
                ϕ = var_info[spl]
                m(var_info)

                ϕ_ca = adapt(adaptor, ComponentArray(var_info))
                @test abs(DynamicPPL.getlogp(var_info) - logπ_unconstrained(ϕ_ca)) ≤ threshold

                # Ensure that precision is preserved
                @test logπ(θ_ca) isa T
                @test logπ_unconstrained(ϕ_ca) isa T
            end
        end
    end

    @testset "ComponentArrays" begin
        spl = DynamicPPL.SampleFromPrior()
        args = make_default_args(data)
        m = Rmap.rmap_naive(args...);
        var_info = DynamicPPL.VarInfo(m);
        θ = var_info[spl]
        θ_ca = ComponentArray(var_info)

        # Verify that we indeed have the correct parameters.
        md = var_info.metadata
        for vn in keys(md)
            @test md[vn].vals == θ_ca[vn]
        end
    end
end
