from lucidflow.drift.metrics import (
    ks_severity,
    ks_test,
    population_stability_index,
    psi_severity,
)


def test_psi_is_zero_for_identical_distributions():
    counts = {"a": 100, "b": 200, "c": 50}
    assert population_stability_index(counts, counts) == 0.0


def test_psi_is_positive_and_grows_with_larger_shift():
    baseline = {"a": 500, "b": 500}
    small_shift = {"a": 450, "b": 550}
    large_shift = {"a": 100, "b": 900}

    psi_small = population_stability_index(baseline, small_shift)
    psi_large = population_stability_index(baseline, large_shift)

    assert psi_small > 0
    assert psi_large > psi_small


def test_psi_handles_a_category_missing_from_one_side():
    baseline = {"a": 500, "b": 500}
    batch = {"a": 500, "b": 400, "c": 100}  # "c" unseen in baseline, "b" unseen in batch's absence handled too

    psi = population_stability_index(baseline, batch)

    assert psi > 0


def test_psi_severity_bands():
    assert psi_severity(0.05) == "none"
    assert psi_severity(0.1) == "moderate"
    assert psi_severity(0.2) == "moderate"
    assert psi_severity(0.25) == "significant"
    assert psi_severity(0.5) == "significant"


def test_ks_test_p_value_high_for_identical_samples():
    sample = [1.0, 2.0, 3.0, 4.0, 5.0] * 20
    _, p_value = ks_test(sample, sample)

    assert p_value == 1.0


def test_ks_test_p_value_low_for_clearly_different_samples():
    baseline = list(range(100))
    shifted = list(range(500, 600))

    _, p_value = ks_test(baseline, shifted)

    assert p_value < 0.01


def test_ks_severity_bands():
    assert ks_severity(0.5) == "none"
    assert ks_severity(0.02) == "moderate"
    assert ks_severity(0.005) == "significant"
