"""Task 1: given a CSV, runs Model 1 on its columns and generates a working
Pydantic contract from the predictions.

    python -m lucidflow.contract_generation.generate_contract <csv_path> [class_name]

Writes the generated contract source (a real, readable .py file) under
contract_generation/generated/ (gitignored -- like the imputation selector's
fitted artifacts, this is derived fresh from whatever CSV you point it at,
not a durable trained artifact) and prints a one-line confidence/risk summary.

Scope, stated plainly (see package README for the full writeup): this makes
the *schema-inference* layer generalize to new tabular data. It does not make
the imputation selector or quarantine classifier generalize -- those stay
tuned to companies.csv's specific columns, same as any trained model is tuned
to what it was trained on.
"""

import datetime
import keyword
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, create_model

from lucidflow.contract_generation.predict import TypePrediction, classify_columns
from lucidflow.contract_generation.type_mapping import FieldSpec, build_field_spec
from lucidflow.ingestion.loader import load_file

GENERATED_DIR = Path(__file__).parent / "generated"

_PY_TYPES = {
    "int": int,
    "str": str,
    "float": float,
    "bool": bool,
}


@dataclass
class GeneratedContract:
    class_name: str
    field_specs: list[FieldSpec]
    aliases: dict[str, str]  # sanitized field name -> original CSV header
    source: str
    model: type[BaseModel]


def _sanitize_identifier(name: str, taken: set[str]) -> str:
    candidate = re.sub(r"\W", "_", name.strip())
    if not candidate or candidate[0].isdigit():
        candidate = f"col_{candidate}"
    if keyword.iskeyword(candidate):
        candidate = f"{candidate}_"
    base = candidate
    i = 2
    while candidate in taken:
        candidate = f"{base}_{i}"
        i += 1
    taken.add(candidate)
    return candidate


def analyze_columns(columns: dict[str, list[str | None]]) -> dict[str, FieldSpec]:
    """Task 1's core step: Model 1 predictions -> field specs, per column."""
    predictions: dict[str, TypePrediction] = classify_columns(columns)
    return {
        column: build_field_spec(column, pred.predicted_type, pred.confidence, columns[column])
        for column, pred in predictions.items()
    }


def _build_dynamic_model(class_name: str, field_specs: list[FieldSpec], aliases: dict[str, str]) -> type[BaseModel]:
    """A real, runtime-usable Pydantic model built via `create_model` -- no
    string parsing or exec() of generated text, so a hostile CSV header or
    cell value can't inject code into this process. `render_source` below
    produces the human-readable .py text separately, purely for the file
    output and the Task 2 comparison report.
    """
    fields: dict[str, tuple] = {}
    for spec in field_specs:
        py_type = _PY_TYPES[spec.python_type] if spec.python_type != "date" else datetime.date
        annotation = py_type if spec.required else (py_type | None)
        alias = aliases.get(spec.column)
        field_kwargs = {"alias": alias} if alias else {}
        field_kwargs.update(spec.constraints)
        if spec.required:
            fields[spec.column] = (annotation, Field(**field_kwargs) if field_kwargs else ...)
        else:
            fields[spec.column] = (annotation, Field(default=None, **field_kwargs))

    model_config = ConfigDict(populate_by_name=True)
    return create_model(class_name, __config__=model_config, **fields)


def _render_field_line(spec: FieldSpec, alias: str | None) -> list[str]:
    lines = [f"    # Model 1 prediction: {spec.predicted_type} (confidence {spec.confidence:.2f})"]
    for comment in spec.comments:
        lines.append(f"    # {comment}")
    for risk in spec.risk_flags:
        lines.append(f"    # RISK: {risk}")

    py_type = "datetime.date" if spec.python_type == "date" else spec.python_type
    annotation = py_type if spec.required else f"{py_type} | None"

    field_args = []
    if alias:
        field_args.append(f"alias={alias!r}")
    for key, value in spec.constraints.items():
        field_args.append(f"{key}={value!r}")

    if spec.required:
        if field_args:
            lines.append(f"    {spec.column}: {annotation} = Field({', '.join(field_args)})")
        else:
            lines.append(f"    {spec.column}: {annotation}")
    else:
        field_args = ["default=None", *field_args]
        lines.append(f"    {spec.column}: {annotation} = Field({', '.join(field_args)})")

    return lines


def render_source(class_name: str, field_specs: list[FieldSpec], aliases: dict[str, str]) -> str:
    """Human-readable generated contract source -- meant to be read/edited by
    a person (or saved to a file), same style as the hand-written Phase 1
    contract. Not exec'd by this codebase; `_build_dynamic_model` is the
    runtime-usable path.
    """
    lines = [
        '"""Auto-generated Pydantic contract.',
        "",
        "Generated by lucidflow.contract_generation from Model 1 (column-type classifier)",
        "predictions -- NOT hand-verified. Read the RISK comments before trusting this",
        "as-is; low-confidence predictions and Model 1's documented blind spots (text",
        "booleans, possible ordinal codes) are flagged inline, not silently resolved.",
        '"""',
        "",
        "import datetime",
        "",
        "from pydantic import BaseModel, ConfigDict, Field",
        "",
        "",
        f"class {class_name}(BaseModel):",
        "    model_config = ConfigDict(populate_by_name=True)",
        "",
    ]
    for spec in field_specs:
        alias = aliases.get(spec.column)
        lines.extend(_render_field_line(spec, alias))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def generate_contract(columns: dict[str, list[str | None]], class_name: str = "GeneratedContract") -> GeneratedContract:
    field_specs_by_column = analyze_columns(columns)

    taken: set[str] = set()
    aliases: dict[str, str] = {}
    renamed_specs: list[FieldSpec] = []
    for original_column, spec in field_specs_by_column.items():
        sanitized = _sanitize_identifier(original_column, taken)
        if sanitized != original_column:
            aliases[sanitized] = original_column
        spec.column = sanitized
        renamed_specs.append(spec)

    source = render_source(class_name, renamed_specs, aliases)
    model = _build_dynamic_model(class_name, renamed_specs, aliases)

    return GeneratedContract(
        class_name=class_name, field_specs=renamed_specs, aliases=aliases, source=source, model=model
    )


def generate_from_csv(path: str | Path, class_name: str = "GeneratedContract") -> GeneratedContract:
    df = load_file(path)
    columns = {column: df[column].to_list() for column in df.columns}
    return generate_contract(columns, class_name=class_name)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m lucidflow.contract_generation.generate_contract <csv_path> [class_name]")
        raise SystemExit(1)
    csv_path = Path(sys.argv[1])
    class_name = sys.argv[2] if len(sys.argv) > 2 else "GeneratedContract"

    result = generate_from_csv(csv_path, class_name=class_name)

    GENERATED_DIR.mkdir(exist_ok=True)
    out_path = GENERATED_DIR / f"{class_name}.py"
    out_path.write_text(result.source)
    print(f"Wrote generated contract to {out_path}")

    n_low_confidence = sum(1 for spec in result.field_specs if spec.confidence < 0.6)
    n_flagged = sum(1 for spec in result.field_specs if spec.risk_flags)
    print(f"{len(result.field_specs)} columns -- {n_low_confidence} low-confidence, {n_flagged} with risk flags")


if __name__ == "__main__":
    main()
