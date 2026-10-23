-- MART: one row per month. This feeds the trend charts on the dashboard.
-- All months are included (a business dashboard shows full history), and the split column says
-- which part of the project timeline each month belongs to.

with monthly as (

    select
        date_trunc('month', date_received)                        as month,
        min(split)                                                as split,
        count(*)                                                  as complaints,
        sum(case when target_eligible then 1 else 0 end)          as eligible_complaints,
        sum(coalesce(got_relief, 0))                              as relief_cases,
        sum(coalesce(got_monetary_relief, 0))                     as monetary_relief_cases,
        sum(case when timely_response = 'No' then 1 else 0 end)   as untimely_cases
    from {{ ref('int_complaints_labeled') }}
    group by 1

),

rates as (

    select
        *,
        round(relief_cases * 1.0 / nullif(eligible_complaints, 0), 4)   as relief_rate,
        round(monetary_relief_cases * 1.0 / nullif(relief_cases, 0), 4)  as monetary_share_of_relief,
        round(untimely_cases * 1.0 / complaints, 4)                      as untimely_rate,
        -- 3-month rolling relief rate: smooths month-to-month noise
        round(
            sum(relief_cases) over (order by month rows between 2 preceding and current row) * 1.0
            / nullif(sum(eligible_complaints) over (order by month rows between 2 preceding and current row), 0)
        , 4)                                                             as relief_rate_3m
    from monthly

)

select
    *,
    round(relief_rate - lag(relief_rate) over (order by month), 4)       as relief_rate_change_vs_prev_month
from rates
order by month
