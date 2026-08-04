import pytest
from pydantic import ValidationError

from lucidflow.validation.pydantic_models import Company

VALID_ROW = {
    "company_id": 1009,
    "name": "IBM",
    "description": "Technology company.",
    "company_size": 7,
    "state": "NY",
    "country": "US",
    "city": "Armonk, New York",
    "zip_code": "10504",
    "address": "International Business Machines Corp.",
    "url": "https://www.linkedin.com/company/ibm",
}


def test_valid_row_passes():
    company = Company.model_validate(VALID_ROW)
    assert company.company_id == 1009
    assert company.state == "NY"


def test_sentinel_zero_country_becomes_none():
    row = {**VALID_ROW, "country": "0"}
    company = Company.model_validate(row)
    assert company.country is None


def test_sentinel_zero_state_becomes_none():
    row = {**VALID_ROW, "state": "0"}
    company = Company.model_validate(row)
    assert company.state is None


def test_company_size_out_of_range_fails():
    row = {**VALID_ROW, "company_size": 9}
    with pytest.raises(ValidationError):
        Company.model_validate(row)


def test_missing_company_id_fails():
    row = {k: v for k, v in VALID_ROW.items() if k != "company_id"}
    with pytest.raises(ValidationError):
        Company.model_validate(row)


def test_all_failures_are_reported_not_just_the_first():
    row = {**VALID_ROW, "company_size": 9, "url": "not-a-url"}
    del row["company_id"]

    with pytest.raises(ValidationError) as exc_info:
        Company.model_validate(row)

    failing_fields = {error["loc"][0] for error in exc_info.value.errors()}
    assert failing_fields == {"company_id", "company_size", "url"}


def test_optional_fields_may_be_null():
    row = {**VALID_ROW, "name": None, "description": None, "company_size": None}
    company = Company.model_validate(row)
    assert company.name is None
    assert company.company_size is None
