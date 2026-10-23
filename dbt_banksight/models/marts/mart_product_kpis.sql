-- MART: one row per product group, with its product family from the seed table.
-- IMPORTANT: only train and validation rows are used. The test period stays sealed until the final evaluation,
-- so no analysis table should show test-period rates by product.

select
    m.product_family,
    c.product_group,
    count(*)                                                          as complaints,
    sum(case when c.target_eligible then 1 else 0 end)                as eligible_complaints,
    sum(coalesce(c.got_relief, 0))                                    as relief_cases,
    round(sum(coalesce(c.got_relief, 0)) * 1.0
          / nullif(sum(case when c.target_eligible then 1 else 0 end), 0), 4)   as relief_rate,
    round(sum(coalesce(c.got_monetary_relief, 0)) * 1.0
          / nullif(sum(coalesce(c.got_relief, 0)), 0), 4)             as monetary_share_of_relief,
    round(sum(case when c.timely_response = 'No' then 1 else 0 end) * 1.0 / count(*), 4)  as untimely_rate
from {{ ref('int_complaints_labeled') }} as c
left join {{ ref('product_group_map') }} as m
    on c.product_group = m.product_group
where c.split in ('train', 'validation')
group by 1, 2
order by complaints desc
