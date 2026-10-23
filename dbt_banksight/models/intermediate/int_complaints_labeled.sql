-- INTERMEDIATE: each complaint plus the label (got_relief) and the time split (train / validation / test).
-- Marts build on this table so that every KPI uses the same definition of "relief".

select
    c.*,
    l.split,
    l.target_eligible,
    l.got_relief,
    l.got_monetary_relief
from {{ ref('stg_complaints') }} as c
join {{ source('features', 'complaint_labels') }} as l
    on c.complaint_id = l.complaint_id
