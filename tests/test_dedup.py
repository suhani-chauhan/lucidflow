import polars as pl

from lucidflow.cleaning.dedup import remove_exact_duplicates


def test_removes_exact_duplicate_rows():
    df = pl.DataFrame(
        {
            "company_id": [1, 2, 1, 3],
            "name": ["Acme", "Globex", "Acme", "Initech"],
        }
    )

    deduped, removed_count = remove_exact_duplicates(df)

    assert removed_count == 1
    assert deduped.height == 3
    assert sorted(deduped["company_id"].to_list()) == [1, 2, 3]


def test_no_duplicates_removes_nothing():
    df = pl.DataFrame({"company_id": [1, 2, 3], "name": ["Acme", "Globex", "Initech"]})

    deduped, removed_count = remove_exact_duplicates(df)

    assert removed_count == 0
    assert deduped.height == 3


def test_does_not_dedup_rows_that_differ_by_one_column():
    df = pl.DataFrame(
        {
            "company_id": [1, 1],
            "name": ["Acme", "Acme Corp"],
        }
    )

    deduped, removed_count = remove_exact_duplicates(df)

    assert removed_count == 0
    assert deduped.height == 2
