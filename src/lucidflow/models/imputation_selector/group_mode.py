"""Group-mode fallback imputation: fill a column's nulls with the most
frequent known value within a group (e.g. state's mode within country),
falling back to the column's global mode when the group itself has no
known value.

Used for columns that are recoverable via geographic correlation but not
worth a full KNN/MICE/LightGBM benchmark (state, city, zip_code) — see
imputation_selector's README/report for why.
"""

import pandas as pd


def fit_group_mode(df: pd.DataFrame, target_col: str, group_cols: list[str]) -> dict:
    known = df[df[target_col].notna()]
    global_mode = known[target_col].mode().iloc[0]
    group_modes = known.groupby(group_cols)[target_col].agg(lambda s: s.mode().iloc[0]).to_dict()
    return {"group_cols": group_cols, "group_modes": group_modes, "global_mode": global_mode}


def _group_key(row: pd.Series, group_cols: list[str]):
    if len(group_cols) == 1:
        return row[group_cols[0]]
    return tuple(row[c] for c in group_cols)


def apply_group_mode(df: pd.DataFrame, target_col: str, lookup: dict) -> pd.Series:
    group_cols = lookup["group_cols"]
    group_modes = lookup["group_modes"]
    global_mode = lookup["global_mode"]

    def fill(row):
        if pd.notna(row[target_col]):
            return row[target_col]
        key = _group_key(row, group_cols)
        return group_modes.get(key, global_mode)

    return df.apply(fill, axis=1)
