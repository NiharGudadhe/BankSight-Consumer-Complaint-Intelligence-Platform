-- A rate must be between 0 and 1. This test returns the bad rows; the test PASSES when it returns nothing.

select 'monthly' as mart, cast(month as varchar) as row_key, relief_rate as bad_value
from {{ ref('mart_monthly_kpis') }}
where relief_rate not between 0 and 1

union all

select 'product', product_group, relief_rate
from {{ ref('mart_product_kpis') }}
where relief_rate not between 0 and 1

union all

select 'issue', product_group || ' / ' || issue, relief_rate
from {{ ref('mart_issue_kpis') }}
where relief_rate not between 0 and 1
