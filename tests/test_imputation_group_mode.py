import pandas as pd

from lucidflow.models.imputation_selector.group_mode import apply_group_mode, fit_group_mode


def test_fills_missing_with_the_mode_of_its_own_group():
    df = pd.DataFrame({
        "country": ["US", "US", "US", "US", "GB", "GB"],
        "state": ["CA", "CA", "NY", None, "LDN", None],
    })
    lookup = fit_group_mode(df, "state", ["country"])

    missing = df[df["state"].isna()]
    filled = apply_group_mode(missing, "state", lookup)

    assert filled.loc[3] == "CA"  # US group's mode
    assert filled.loc[5] == "LDN"  # GB group's mode


def test_falls_back_to_global_mode_for_a_group_never_seen_during_fit():
    known = pd.DataFrame({"country": ["US", "US", "US"], "state": ["CA", "CA", "NY"]})
    lookup = fit_group_mode(known, "state", ["country"])

    unseen_group = pd.DataFrame({"country": ["FR"], "state": [None]})
    filled = apply_group_mode(unseen_group, "state", lookup)

    assert filled.iloc[0] == lookup["global_mode"] == "CA"


def test_supports_multi_column_group_keys():
    df = pd.DataFrame({
        "country": ["US", "US", "US"],
        "state": ["CA", "CA", "NY"],
        "city": ["LA", "LA", None],
    })
    lookup = fit_group_mode(df, "city", ["state", "country"])

    missing = df[df["city"].isna()]
    filled = apply_group_mode(missing, "city", lookup)

    # NY/US group has no known city -> falls back to the global mode ("LA")
    assert filled.iloc[0] == "LA"


def test_leaves_already_present_values_untouched():
    df = pd.DataFrame({"country": ["US", "US"], "state": ["CA", "NY"]})
    lookup = fit_group_mode(df, "state", ["country"])

    filled = apply_group_mode(df, "state", lookup)

    assert list(filled) == ["CA", "NY"]
