"""Stratified train/test split that guarantees minimum class representation.

`sklearn.model_selection.train_test_split(..., stratify=y)` doesn't strictly
guarantee that a very small class (e.g. n=3) lands at least one example in
both train and test — for small n it can round a class's test share down to
zero. This is a hard requirement for the boolean class here (n=3), so we
split per-class explicitly instead of relying on sklearn's rounding.
"""

import random


def stratified_min1_split(
    items: list, labels: list[str], test_size: float = 0.25, random_state: int = 42
) -> tuple[list, list, list, list]:
    """Split `items`/`labels` so every class with >=2 members contributes
    at least one example to both the train and test sets.

    A class with exactly 1 member goes entirely to train (it can't be split).
    """
    rng = random.Random(random_state)

    by_label: dict[str, list[int]] = {}
    for idx, label in enumerate(labels):
        by_label.setdefault(label, []).append(idx)

    train_idx: list[int] = []
    test_idx: list[int] = []

    for label, idxs in by_label.items():
        shuffled = idxs[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        if n < 2:
            train_idx.extend(shuffled)
            continue
        n_test = max(1, round(n * test_size))
        n_test = min(n_test, n - 1)  # always leave >=1 in train
        test_idx.extend(shuffled[:n_test])
        train_idx.extend(shuffled[n_test:])

    train_items = [items[i] for i in train_idx]
    test_items = [items[i] for i in test_idx]
    train_labels = [labels[i] for i in train_idx]
    test_labels = [labels[i] for i in test_idx]

    return train_items, test_items, train_labels, test_labels
