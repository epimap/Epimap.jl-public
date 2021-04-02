using Epimap

@testset "Rmap" begin
    data = Rmap.load_data()

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
        # Manchester and Birmingham are 112150.26 apart in `Haversine` distance.
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
end

