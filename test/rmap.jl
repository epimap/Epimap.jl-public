using Epimap, Dates, Adapt, Test, Zygote, ForwardDiff, ComponentArrays, UnPack

include("test_data.jl")

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
        args = Rmap.setup_args(
            Rmap.rmap_naive, data, T;
            num_condition_days = num_condition_days,
            num_steps = num_steps,
            timestep = timestep,
            condition_observations = condition_observations
        )

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
        @test "Trafford" ??? filtered_data.area_names

        radius = 0.11
        filtered_data = Rmap.filter_areas_by_distance(
            data, ["Manchester"];
            radius = radius
        )
        @test "Trafford" ??? filtered_data.area_names
    end

    @testset "model" begin
        for T ??? [Float32, Float64]
            adaptor = Epimap.FloatMaybeAdaptor{T}()
            # Use different threshold if `Float32` since we're comparing to use of `Float64`.
            threshold = (T === Float32) ? 1 : 1
            num_repeats = 10

            # Instantiate model
            args = make_default_args(data, T)
            m = Rmap.rmap_naive(args..., T, ??_time=T(100.0), ??_spatial=T(0.1));

            # `make_logjoint`
            log??, log??_unconstrained, b, ??_init = Epimap.make_logjoint(m)

            # Check average difference
            spl = DynamicPPL.SampleFromPrior()

            i = 1
            max_num_repeats = 30
            attempts = 0
            while i ??? num_repeats
                attempts += 1
                if attempts > max_num_repeats
                    error("couldn't sample $(num_repeats) samples from the prior with finite logjoint in $(max_num_repeats)")
                end

                # Constrained space
                var_info = DynamicPPL.VarInfo(m);
                ?? = var_info[spl]
                m(var_info)

                # `ComponentArray` impl
                ??_ca = adapt(adaptor, ComponentArray(var_info))
                # Due to possible lower numerical precision when using `Float32`
                # we have have samples from the prior which are `-Inf32` for
                # `log??` but finite when using `Float64`.
                # In these cases we just skip this particular sample from the prior.
                # Note that we only increment `i` once we've performed all tests,
                # hence the tests still need to pass `num_repeats` before progressing.
                isfinite(log??(??_ca)) || continue
                @test abs(DynamicPPL.getlogp(var_info) - log??(??_ca)) ??? threshold
                # Raw array impl
                ??_ca_raw = ComponentArrays.getdata(??_ca)
                @test log??(??_ca) == log??(??_ca_raw)

                # Gradients
                ???_zy = Zygote.gradient(log??, ??_ca)[1]
                ???_fd = ForwardDiff.gradient(log??, ??_ca)
                @test ???_zy ??? ???_fd

                # Unconstrained space
                DynamicPPL.link!!(var_info, spl, Val(keys(??_ca)))
                ?? = var_info[spl]
                m(var_info)

                # `ComponentArray` impl
                ??_ca = adapt(adaptor, ComponentArray(var_info))
                isfinite(log??_unconstrained(??_ca)) || continue
                @test abs(DynamicPPL.getlogp(var_info) - log??_unconstrained(??_ca)) ??? threshold
                # Raw array impl
                ??_ca_raw = ComponentArrays.getdata(??_ca)
                @test log??_unconstrained(??_ca) == log??_unconstrained(??_ca_raw)

                # Gradients
                ???_zy = Zygote.gradient(log??_unconstrained, ??_ca)[1]
                ???_fd = ForwardDiff.gradient(log??_unconstrained, ??_ca)
                @test ???_zy ??? ???_fd

                # Ensure that precision is preserved
                @test log??(??_ca) isa T
                @test log??(??_ca_raw) isa T
                @test log??_unconstrained(??_ca) isa T
                @test log??_unconstrained(??_ca_raw) isa T

                # Increment
                i += 1
            end
        end
    end

    @testset "flux" begin
        for T ??? [Float32, Float64]
            adaptor = Epimap.FloatMaybeAdaptor{T}()
            # Instantiate model
            args = make_default_args(data, T)

            # `make_logjoint`
            log??, log??_unconstrained, b, ??_init = Epimap.make_logjoint(Rmap.rmap_naive(args..., T))

            @unpack F_id, F_in, F_out = args
            @unpack ??, ????? = ??_init
            F = Rmap.compute_flux(F_id, F_in, F_out, ??, ?????)

            @test all(sum(F, dims=2) .??? 1)
        end
    end

    @testset "ComponentArrays" begin
        spl = DynamicPPL.SampleFromPrior()
        args = make_default_args(data)
        m = Rmap.rmap_naive(args...);
        var_info = DynamicPPL.VarInfo(m);
        ?? = var_info[spl]
        ??_ca = ComponentArray(var_info)

        # Verify that we indeed have the correct parameters.
        md = var_info.metadata
        for vn in keys(md)
            @test md[vn].vals == ??_ca[vn]
        end
    end
end
