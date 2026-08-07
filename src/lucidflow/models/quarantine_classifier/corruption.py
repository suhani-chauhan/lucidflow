"""Synthetic corruption injection for the quarantine classifier's training set.

Scoped to four types only (named e-h from the original 8-type proposal;
a-d were dropped): encoding corruption, truncated/garbled text, cross-field
zip/state mismatch, and null-storm. The dropped types (missing required
field, malformed URL, out-of-range company_size, type corruption) are
already caught deterministically by the Pydantic contract with perfect
precision — training on them would either teach the classifier nothing new
or let it shortcut on a single trivial feature. These four are all things
the contract structurally cannot check, so above-chance recall on them is
real evidence the model adds value beyond the existing gate.

Every corruption function takes a real, valid row (never a fabricated one)
and returns (corrupted_row, corruption_type) — corruption never mutates the
input row in place, and the type label always travels with the row so it's
never blended into training data as if it were real.
"""

import random

CORRUPTION_TYPES = ["encoding", "truncation", "zip_state_mismatch", "null_storm"]

_OPTIONAL_FIELDS = ["company_size", "state", "country", "city", "zip_code", "address", "description"]

_MOJIBAKE_SUBSTITUTIONS = {
    "'": "â€™",  # UTF-8 right-single-quote misread as Latin-1 ("â€™")
    '"': "â€œ",  # left double quote misread ("â€œ")
    "-": "â€“",  # en dash misread ("â€“")
    "e": "Ã©",  # "é" misread as two Latin-1 chars ("Ã©")
    " ": "Â ",  # non-breaking space misread ("Â ")
}

_GARBLE_SUFFIX_CHARS = "kjfhqxzKJFHQXZ#@%&*0198"


def corrupt_encoding(row: dict, rng: random.Random) -> tuple[dict, str]:
    """Injects mojibake-style substitutions into description (or name if description is null) —
    simulates text that was written as UTF-8 and misread as Latin-1/CP1252 somewhere in the pipeline.
    """
    corrupted = dict(row)
    field = "description" if row.get("description") else "name"
    text = corrupted[field]

    candidate_positions = [i for i, c in enumerate(text) if c in _MOJIBAKE_SUBSTITUTIONS]
    n_hits = min(len(candidate_positions), rng.randint(1, 3))

    if n_hits == 0:
        # no substitutable characters at all (rare, very short/plain text) -- fall back to
        # inserting a bare mojibake fragment so the row is still corrupted.
        insert_at = rng.randint(0, len(text))
        corrupted[field] = text[:insert_at] + "Ã©â€™" + text[insert_at:]
        return corrupted, "encoding"

    replace_positions = set(rng.sample(candidate_positions, n_hits))
    corrupted[field] = "".join(
        _MOJIBAKE_SUBSTITUTIONS[c] if i in replace_positions else c for i, c in enumerate(text)
    )
    return corrupted, "encoding"


def corrupt_truncation(row: dict, rng: random.Random) -> tuple[dict, str]:
    """Cuts description short mid-word and appends a garbled suffix -- simulates a
    fixed-width column truncation bug combined with corrupted trailing bytes.
    """
    corrupted = dict(row)
    field = "description" if row.get("description") else "name"
    text = corrupted[field]

    cut_frac = rng.uniform(0.15, 0.5)
    cut_at = max(1, int(len(text) * cut_frac))
    garble_len = rng.randint(3, 8)
    garble = "".join(rng.choice(_GARBLE_SUFFIX_CHARS) for _ in range(garble_len))

    corrupted[field] = text[:cut_at] + garble
    return corrupted, "truncation"


def corrupt_zip_state_mismatch(
    row: dict, rng: random.Random, zip3_to_state: dict[str, str], zip3_pool: dict[str, list[str]]
) -> tuple[dict, str]:
    """Swaps in a real zip code from a DIFFERENT state than the row's actual state --
    caller must only pass rows from `zip_state_reference.eligible_rows` (known ground-truth state).
    """
    corrupted = dict(row)
    true_state = str(row["state"]).strip().upper()

    wrong_zip3_choices = [z3 for z3, state in zip3_to_state.items() if state != true_state]
    wrong_zip3 = rng.choice(wrong_zip3_choices)
    corrupted["zip_code"] = rng.choice(zip3_pool[wrong_zip3])
    return corrupted, "zip_state_mismatch"


def corrupt_null_storm(row: dict, rng: random.Random) -> tuple[dict, str]:
    """Blanks out most optional fields -- a barely-populated record, distinct from a
    single missing field. company_id, name, and url are left intact (still identifiable).
    """
    corrupted = dict(row)
    n_to_null = rng.randint(5, len(_OPTIONAL_FIELDS))
    fields_to_null = rng.sample(_OPTIONAL_FIELDS, n_to_null)
    for field in fields_to_null:
        corrupted[field] = None
    return corrupted, "null_storm"
