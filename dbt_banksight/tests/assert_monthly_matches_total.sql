-- Reconciliation: the monthly mart must add up to the same number of eligible complaints as the base table.
-- Returns one row (a failure) only if the two totals differ.

with from_mart as (
    select sum(eligible_complaints) as total from {{ ref('mart_monthly_kpis') }}
),
from_base as (
    select count(*) as total from {{ ref('int_complaints_labeled') }} where target_eligible
)

select from_mart.total as mart_total, from_base.total as base_total
from from_mart, from_base
where from_mart.total <> from_base.total
