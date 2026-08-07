from lucidflow.models.column_type_classifier.split import stratified_min1_split


def test_small_class_of_three_always_gets_at_least_one_test_example():
    # Mirrors the real training set: boolean has exactly 3 examples, and a
    # standard stratified split can round its test share down to zero.
    labels = (
        ["boolean"] * 3
        + ["categorical"] * 15
        + ["identifier"] * 10
        + ["numeric_continuous"] * 11
        + ["geographic"] * 9
        + ["free_text"] * 7
        + ["date"] * 5
        + ["url"] * 4
    )
    items = list(range(len(labels)))

    for seed in range(10):
        _, _, train_labels, test_labels = stratified_min1_split(
            items, labels, test_size=0.25, random_state=seed
        )
        assert test_labels.count("boolean") >= 1
        assert train_labels.count("boolean") >= 1


def test_split_preserves_every_item_exactly_once():
    labels = ["a"] * 4 + ["b"] * 6
    items = list(range(len(labels)))

    train_items, test_items, train_labels, test_labels = stratified_min1_split(
        items, labels, test_size=0.25, random_state=42
    )

    assert sorted(train_items + test_items) == items
    assert len(train_labels) == len(train_items)
    assert len(test_labels) == len(test_items)


def test_class_with_single_member_goes_entirely_to_train():
    labels = ["rare"] * 1 + ["common"] * 8
    items = list(range(len(labels)))

    _, _, train_labels, test_labels = stratified_min1_split(
        items, labels, test_size=0.25, random_state=42
    )

    assert train_labels.count("rare") == 1
    assert test_labels.count("rare") == 0


def test_every_class_with_at_least_two_members_appears_in_both_splits():
    labels = ["x"] * 2 + ["y"] * 5 + ["z"] * 20
    items = list(range(len(labels)))

    _, _, train_labels, test_labels = stratified_min1_split(
        items, labels, test_size=0.25, random_state=42
    )

    for label in ("x", "y", "z"):
        assert train_labels.count(label) >= 1
        assert test_labels.count(label) >= 1
