"""Rescue actions that re-derive from the ORIGINAL table instead of the failing pipeline's
intermediate artefacts.

Two actions, same input (original question + coordinate-indexed original table):
  RAW-PoT : the strong model writes Python against the raw rows.
  PLAN    : the strong model emits a restricted JSON computation plan; the CPU reads the
            original cells and computes.

Why: escalation that reuses the weak model's decomposition/sanitised table scores 0.5194 on
the 276 TAT-QA first-pass-missing items, while a full strong rerun scores 0.7150. The gap is
the corrupted intermediate, so the fix is to go back to the source table.
"""
import ast
import json
import re

# ---------------------------------------------------------------- table serialisation


def _cell(v):
    return str(v).replace("\n", " ").replace("|", "/").strip()


def serialise(table, with_coords=True):
    """Row + column indexed view of the ORIGINAL table. Header row stays as data (row 0),
    because TAT-QA style tables carry their header inside the rows."""
    cols = [_cell(c) for c in (table.get("columns") or [])]
    rows = table.get("data") or []
    out = []
    if with_coords:
        out.append("COLUMNS (refer to a cell as [row, col], both 0-based):")
        out.append(" | ".join(f"C{c}={cols[c]}" for c in range(len(cols))))
        out.append("ROWS:")
        for r, row in enumerate(rows):
            out.append(f"R{r}: " + " | ".join(_cell(v) for v in row))
    else:
        out.append(" | ".join(cols))
        for row in rows:
            out.append(" | ".join(_cell(v) for v in row))
    return "\n".join(out)


# ---------------------------------------------------------------- number parsing


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def parse_number(text):
    """Parse a raw table cell into a float. Returns (value, is_percent) or (None, False).

    Handles: leading currency symbols, thousands separators, accounting negatives "(1,234)",
    trailing percent signs, and cells that merely contain a number.
    """
    if text is None:
        return None, False
    s = str(text).strip()
    if not s or s in {"-", "—", "–", "N/A", "n/a", "NA", "nan", "NaN", "None"}:
        return None, False
    negative = s.startswith("(") and s.endswith(")")
    if negative:
        s = s[1:-1]
    is_pct = "%" in s
    s = s.replace("%", "").replace("$", "").replace("¥", "").replace("€", "").strip()
    s = s.replace(",", "")
    m = _NUM.search(s)
    if not m:
        return None, False
    try:
        v = float(m.group(0))
    except ValueError:
        return None, False
    if negative:
        v = -abs(v)
    return v, is_pct


# ---------------------------------------------------------------- plan execution

OPS = {"value", "text", "sum", "mean", "diff", "ratio", "pct_change", "compare", "abs_diff"}


def _get(rows, rc):
    r, c = int(rc[0]), int(rc[1])
    if r < 0 or r >= len(rows):
        raise ValueError(f"row {r} out of range (0..{len(rows) - 1})")
    row = rows[r]
    if c < 0 or c >= len(row):
        raise ValueError(f"col {c} out of range (0..{len(row) - 1})")
    v, pct = parse_number(row[c])
    if v is None:
        raise ValueError(f"cell [{r},{c}] is not numeric: {row[c]!r}")
    return v, pct


def execute_plan(plan, table):
    """Run one plan step. Returns (value, note). Raises ValueError on any invalid step."""
    if not isinstance(plan, dict):
        raise ValueError("plan step is not an object")
    op = plan.get("op")
    if op not in OPS:
        raise ValueError(f"unsupported op {op!r}")
    rows = table.get("data") or []

    def cell(key):
        rc = plan.get(key)
        if not (isinstance(rc, (list, tuple)) and len(rc) == 2):
            raise ValueError(f"'{key}' must be [row, col]")
        return _get(rows, rc)

    if op == "value":
        v, pct = cell("a")
        return v, "percent" if pct else ""

    if op == "text":
        rc = plan.get("a")
        if not (isinstance(rc, (list, tuple)) and len(rc) == 2):
            raise ValueError("'a' must be [row, col]")
        r, c = int(rc[0]), int(rc[1])
        if r < 0 or r >= len(rows):
            raise ValueError(f"row {r} out of range")
        row = rows[r]
        if c < 0 or c >= len(row):
            raise ValueError(f"col {c} out of range")
        return str(row[c]).strip(), "text"

    if op in ("sum", "mean"):
        cells = plan.get("cells") or []
        if not cells:
            raise ValueError("'cells' is empty")
        vals = [_get(rows, rc)[0] for rc in cells]
        return (sum(vals) if op == "sum" else sum(vals) / len(vals)), ""

    if op in ("diff", "abs_diff"):
        a, _ = cell("a")
        b, _ = cell("b")
        return (a - b if op == "diff" else abs(a - b)), ""

    if op == "ratio":
        a, _ = cell("a")
        b, _ = cell("b")
        if b == 0:
            raise ValueError("division by zero")
        return a / b, ""

    if op == "pct_change":
        a, _ = cell("a")   # old
        b, _ = cell("b")   # new
        if a == 0:
            raise ValueError("division by zero in pct_change")
        return (b - a) / a * 100.0, "percent"

    if op == "compare":
        a, _ = cell("a")
        b, _ = cell("b")
        return ("higher" if a > b else "lower" if a < b else "equal"), ""

    raise ValueError(f"unhandled op {op!r}")


_PLAN_RE = re.compile(r"\{.*\}", re.S)


def parse_plan(text):
    """Extract the JSON plan from a model response. Accepts a single step or {'steps': [...]}."""
    if not text:
        raise ValueError("empty response")
    t = re.sub(r"```(?:json)?", " ", str(text))
    m = _PLAN_RE.search(t)
    if not m:
        raise ValueError("no JSON object found")
    obj = json.loads(m.group(0))
    if isinstance(obj, dict) and "steps" in obj:
        steps = obj["steps"]
    elif isinstance(obj, list):
        steps = obj
    else:
        steps = [obj]
    if not isinstance(steps, list) or not steps:
        raise ValueError("plan has no steps")
    return steps


def format_value(v, note=""):
    """Render a computed value the way the scorers expect (no thousands separators)."""
    if isinstance(v, str):
        return v
    if abs(v - round(v)) < 1e-9:
        s = str(int(round(v)))
    else:
        s = f"{v:.4f}".rstrip("0").rstrip(".")
    return s


def run_plan(text, table):
    """Execute a plan response end to end. Returns (answer_string, detail)."""
    steps = parse_plan(text)
    outs = []
    for st in steps:
        v, note = execute_plan(st, table)
        outs.append(format_value(v, note))
    return ", ".join(outs), {"steps": len(steps), "ok": True}


# ---------------------------------------------------------------- prompts

SYS_RAW = ("You are a careful financial-table analyst. You write short Python that computes "
           "the answer directly from the table rows given to you.")

USER_RAW = """Answer the question using the table below.

The table is provided as zero-based rows. `table[r][c]` is the raw text of the cell at row r,
column c. Cell values are strings; strip currency symbols and thousands separators before
doing arithmetic, and treat values in parentheses as negative.

<<TABLE>>

Question: <<QUESTION>>

Write Python that:
1. reads the needed cells from `table`,
2. computes the answer,
3. prints the final answer with `print(...)`.
Only print the final answer. Do not print explanations.
"""

SYS_PLAN = ("You are a careful financial-table analyst. You do NOT write code. You select the "
            "cells needed and choose one arithmetic operation; a deterministic interpreter "
            "reads the original cells and computes the result.")

USER_PLAN = """Answer the question using the table below.

The table is provided as zero-based rows and columns. A cell is referenced as [row, col].

<<TABLE>>

Question: <<QUESTION>>

Return ONE JSON object describing how to compute the answer, using exactly one of these ops:

  {"op":"value",      "a":[r,c]}          # numeric cell
  {"op":"text",       "a":[r,c]}          # return the cell text as-is (names, labels)
  {"op":"sum",        "cells":[[r,c],[r,c],...]}
  {"op":"mean",       "cells":[[r,c],[r,c],...]}
  {"op":"diff",       "a":[r,c], "b":[r,c]}          # a - b
  {"op":"abs_diff",   "a":[r,c], "b":[r,c]}          # |a - b|
  {"op":"ratio",      "a":[r,c], "b":[r,c]}          # a / b
  {"op":"pct_change", "a":[r,c], "b":[r,c]}          # a is OLD, b is NEW -> (b-a)/a*100
  {"op":"compare",    "a":[r,c], "b":[r,c]}          # higher | lower | equal

If several answers are needed, return {"steps":[ ...one object per answer... ]}.

Rules:
- Only use cell coordinates that exist in the table.
- Do not compute the arithmetic yourself; the interpreter does it.
- Return the JSON only, no prose.

JSON:"""


def render(template, table, question):
    """Fill a prompt template without str.format (they contain literal JSON braces)."""
    return template.replace("<<TABLE>>", serialise(table)).replace("<<QUESTION>>", question)
