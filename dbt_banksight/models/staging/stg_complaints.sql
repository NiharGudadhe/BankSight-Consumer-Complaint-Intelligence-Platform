-- STAGING: one clean row per complaint.
-- Only renaming, casting and tidying here. No business logic.
-- We leave out the long narrative text (only its length is kept) and the unusable company column.

select
    complaint_id,
    cast(date_received as date)  as date_received,
    product_group,
    coalesce(issue, 'Unknown')   as issue,
    state,
    tags,
    timely_response,
    company_response,
    length(narrative)            as narrative_chars
from {{ source('raw', 'cfpb_complaints') }}
