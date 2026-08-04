"""Hand-written data contract for the LinkedIn Job Postings `companies.csv` reference dataset.

Field types and nullability below are derived directly from an inspection of
data/linkedin-job-postings/companies/companies.csv (24,473 rows) — see the
project README for the inspection notes.
"""


from pydantic import BaseModel, Field, HttpUrl, field_validator

# The source data uses the literal string "0" as a sentinel for "unknown" in
# country/state instead of an empty cell. Treated as null, not a real value.
_NULL_SENTINELS = {"0"}


class Company(BaseModel):
    company_id: int
    name: str | None = None
    description: str | None = None
    # LinkedIn's bucketed company-size code (1-7), not a literal headcount.
    company_size: int | None = Field(default=None, ge=1, le=7)
    state: str | None = None
    country: str | None = None
    city: str | None = None
    zip_code: str | None = None
    address: str | None = None
    url: HttpUrl

    @field_validator("country", "state", mode="before")
    @classmethod
    def _sentinel_to_none(cls, value):
        if value is None:
            return None
        if str(value).strip() in _NULL_SENTINELS:
            return None
        return value
