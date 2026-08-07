import pandas as pd

from lucidflow.models.imputation_selector.benchmark import benchmark_column
from lucidflow.models.imputation_selector.strategies import (
    MedianStrategy,
    ModeStrategy,
    build_strategies,
)

# Unequal group sizes so frequency-encoding predictors actually differ between
# groups (equal-sized groups collapse to the same encoded value and destroy
# the very signal the test is trying to give KNN/MICE).
_KNOWN = pd.DataFrame({"region": ["north"] * 120 + ["south"] * 80, "code": [1] * 120 + [5] * 80})


def test_median_strategy_rounds_to_nearest_code():
    strategy = MedianStrategy()
    strategy.fit(pd.DataFrame({"code": [1, 1, 3, 5, 5]}), "code", [])

    preds = strategy.predict(pd.DataFrame(index=range(3)), [])

    assert (preds == 3).all()


def test_mode_strategy_predicts_most_frequent_category():
    strategy = ModeStrategy()
    strategy.fit(pd.DataFrame({"cat": ["a", "b", "b", "b", "c"]}), "cat", [])

    preds = strategy.predict(pd.DataFrame(index=range(3)), [])

    assert (preds == "b").all()


def test_knn_and_lightgbm_recover_a_clean_signal_the_baseline_cannot():
    strategies = {s.name: s for s in build_strategies("ordinal")}
    train, test = _KNOWN.iloc[:150], _KNOWN.iloc[150:]

    for strategy in strategies.values():
        strategy.fit(train, "code", ["region"])

    baseline_preds = strategies["median"].predict(test, ["region"])
    knn_preds = strategies["knn"].predict(test, ["region"])
    lgb_preds = strategies["lightgbm"].predict(test, ["region"])

    # baseline can only ever predict one constant value across a 2-class split
    assert baseline_preds.nunique() == 1
    assert (knn_preds == test["code"]).all()
    assert (lgb_preds == test["code"]).all()


def test_benchmark_column_selects_the_highest_scoring_strategy():
    result = benchmark_column(_KNOWN, "code", ["region"], "ordinal")

    assert result["scores"]["median"] < result["scores"][result["winner"]]
    assert result["winner"] == max(result["scores"], key=result["scores"].get)


def test_class_coverage_separates_evaluated_from_singleton_classes():
    # two classes with enough rows to test, one singleton that can only ever
    # be trained on (a class with 1 example can't also be held out).
    known = pd.DataFrame({
        "region": ["north"] * 10 + ["south"] * 10 + ["east"] * 1,
        "code": [1] * 10 + [5] * 10 + [9] * 1,
    })

    result = benchmark_column(known, "code", ["region"], "ordinal")
    coverage = result["class_coverage"]

    assert coverage["total_classes"] == 3
    assert coverage["evaluated_classes"] == 2
    assert coverage["singleton_classes"] == 1
    assert coverage["coverage_ratio"] == 2 / 3


def test_class_coverage_is_full_when_every_class_has_multiple_rows():
    result = benchmark_column(_KNOWN, "code", ["region"], "ordinal")
    coverage = result["class_coverage"]

    assert coverage["singleton_classes"] == 0
    assert coverage["coverage_ratio"] == 1.0


def test_build_strategies_uses_median_for_ordinal_and_mode_for_categorical():
    ordinal_names = {s.name for s in build_strategies("ordinal")}
    categorical_names = {s.name for s in build_strategies("categorical")}

    assert "median" in ordinal_names and "mode" not in ordinal_names
    assert "mode" in categorical_names and "median" not in categorical_names
    assert {"knn", "mice", "lightgbm"} <= ordinal_names
    assert {"knn", "mice", "lightgbm"} <= categorical_names
