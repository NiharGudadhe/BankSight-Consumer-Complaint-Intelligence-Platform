-- MART: one row per (product group, issue), ranked by size inside each product.
-- Same rule as the product mart: train and validation only.

with issue_stats as (

    select
        product_group,
        issue,
        count(*)                                                  as complaints,
        sum(case when target_eligible then 1 else 0 end)          as eligible_complaints,
        sum(coalesce(got_relief, 0))                              as relief_cases
    from {{ ref('int_complaints_labeled') }}
    where split in ('train', 'validation')
    group by 1, 2

)

select
    product_group,
    issue,
    complaints,
    eligible_complaints,
    relief_cases,
    round(relief_cases * 1.0 / nullif(eligible_complaints, 0), 4)            as relief_rate,
    row_number() over (partition by product_group order by complaints desc)  as rank_in_product
from issue_stats
order by product_group, rank_in_product
