@testset "split_dates" begin
    dates = Date(2020, 1, 1):Day(1):Date(2021, 1, 1)

    first_date_cands = [nothing, Date(2020, 10, 1)]
    last_date_cands = [nothing, Date(2020, 11, 1)]
    num_steps_cands = [nothing, 10]
    num_condition_days_cands = [1, 10, 30]
    timestep_cands = [Day(1), Week(1)]

    for first_date in first_date_cands, last_date in last_date_cands, num_steps in num_steps_cands, timestep in timestep_cands, num_condition_days in num_condition_days_cands
        dates_condition, dates_model = Rmap.split_dates(
            dates;
            first_date = first_date,
            last_date = last_date,
            num_steps = num_steps,
            num_condition_days = num_condition_days,
            timestep = timestep
        )

        # Do the endpoints make sense?
        @test Dates.days(dates_model[end] - dates_model[1] + Day(1)) % Dates.days(timestep) == 0
        # Do we have the correct number of dates?
        @test length(dates_model) % Dates.days(timestep) == 0
        # Is `dates_condition` of right length?
        @test length(dates_condition) == num_condition_days
    end
end
