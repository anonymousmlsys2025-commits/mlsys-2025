import re, time, pathlib, pandas as pd, ast, json
from openai import OpenAI

# ───────────────────────── CONFIG ─────────────────────────
MODEL_ID = "gpt-4.1-mini-2025-04-14"
TIMEOUT  = 600

API_KEY  = ""
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set.")
INPUT_CSV = "satbench_test.csv"

client = OpenAI(api_key=API_KEY)

suffix   = MODEL_ID.replace("-", "_")
CSV_FILE = pathlib.Path(f"nl2pl_{suffix}_summary.csv")

# ───────────────── PROMPT (LLM extracts pairs as JSON) ─────────────────
NL2PL_PROMPT = """### Task
Translate the English constraints into a single propositional-logic formula.

### How to treat variable names and indices
- Read both the Variable Mapping block and the Scenario text.
- If the Scenario or Constraints explicitly define label↔number pairs, EXTRACT them and treat them as part of the mapping.
  Extract ONLY explicit pairs in these forms:
  • label (number)      e.g., diurnal (0), prime (2)
  • number = label      e.g., 0 = ore
  • number is label     e.g., 2 is tech
- Inside x(...), ALWAYS use numeric indices (digits only). Never place words inside x(...).

### Critical Rules
- Variable Naming (strict): x(<int>[, <int>…]) only.
  Valid: x(2,), x(2,0), x(3,1,4)
  Invalid: x(2, diurnal), x(planet=2,0), x(A)
- Symbols: Only ¬ (NOT), ∧ (AND), ∨ (OR), and parentheses ().
- Form: Prefer CNF: (x(0,) ∨ ¬x(2,)) ∧ (…)
- No extras: Do not use → or ↔. No LaTeX, no commentary. Output the final formula on one line.

### Naming Example
Bad: (x(0, prime) ∨ x(0, midyear)) ∧ ¬x(0, prime)
Good: (x(0, 0) ∨ x(0, 1)) ∧ ¬x(0, 0)

### Scenario (you may extract explicit pairs from here)
{scenario}

### Variable Mapping (authoritative seed)
{variable_mapping}

### English Constraints
{constraints}

### Output format (exact)
Variable Mapping (JSON):
<valid JSON object mapping each label string to its numeric index as a string, e.g. {{"orchid":"0","fern":"1"}}>

Logical Formula:
<one-line CNF formula using only numeric variable names>
"""

# ──────────────────────── HELPERS ────────────────────────

def call_openai(prompt: str):
    t0 = time.perf_counter()
    rsp = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        timeout=TIMEOUT,
        temperature=0,
    )
    dt = round(time.perf_counter() - t0, 3)
    content = rsp.choices[0].message.content.strip()
    usage = getattr(rsp, "usage", None)
    pt = getattr(usage, "prompt_tokens", 0) if usage else 0
    ct = getattr(usage, "completion_tokens", 0) if usage else 0
    tt = getattr(usage, "total_tokens", 0) if usage else 0
    return content, dt, MODEL_ID, pt, ct, tt

def extract_mapping_and_formula(block: str):
    """Return (mapping_json_str, formula_line) as-is (no normalization)."""
    mapping_json = "{}"
    mm = re.search(r"Variable Mapping \(JSON\):\s*({.*?})\s*Logical Formula:", block, re.S | re.I)
    if mm:
        maybe = mm.group(1).strip()
        try:
            json.loads(maybe)
            mapping_json = maybe
        except Exception:
            mapping_json = "{}"

    formula = ""
    ml = re.split(r"Logical Formula:\s*", block, flags=re.I, maxsplit=1)
    if len(ml) == 2:
        after = ml[1]
        after = re.split(r"\n\s*(?:Variable Mapping|English Constraints?)\s*:\s*", after, flags=re.I, maxsplit=1)[0]
        for raw in after.splitlines():
            s = raw.strip().strip("`").strip()
            if s:
                formula = s
                break
    return mapping_json, formula

def parse_conditions_cell(cell):
    """'conditions' may be a Python-list-like string; convert to list[str]."""
    if isinstance(cell, list):
        return [str(x) for x in cell]
    if isinstance(cell, str):
        s = cell.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                val = ast.literal_eval(s)
                if isinstance(val, list):
                    return [str(x) for x in val]
            except Exception:
                pass
        return [s]
    return [str(cell)]

# ────────────────────────── MAIN ─────────────────────────

def main():
    df = pd.read_csv(INPUT_CSV)

    if "conditions" not in df.columns:
        raise ValueError(f"'conditions' column not found. Columns: {list(df.columns)}")

    has_mapping  = "variable_mapping" in df.columns
    has_scenario = "scenario" in df.columns

    rows = []

    for idx, row in df.iterrows():
        cond_list = parse_conditions_cell(row["conditions"])
        input_conditions = " ".join(x.strip() for x in cond_list if str(x).strip())

        input_mapping = ""
        if has_mapping and isinstance(row.get("variable_mapping"), str):
            input_mapping = row["variable_mapping"].strip()

        scenario_text = ""
        if has_scenario and isinstance(row.get("scenario"), str):
            scenario_text = row["scenario"].strip()

        prompt = NL2PL_PROMPT.format(
            scenario=scenario_text,
            variable_mapping=input_mapping,
            constraints=input_conditions if input_conditions else "(none provided)"
        )

        try:
            block, dur, mdl, pt, ct, tt = call_openai(prompt)
        except Exception as e:
            print(f"[NL→PL] row {idx} ERROR: {e}")
            continue

        mapping_json, formula = extract_mapping_and_formula(block)

        label = ""
        if "satisfiable" in df.columns:
            val = row["satisfiable"]
            if isinstance(val, (bool, int)):
                label = "SAT" if bool(val) else "UNSAT"
            else:
                s = str(val).strip().lower()
                label = "SAT" if s in {"1","true","t","yes","y"} else "UNSAT"

        original_formula = row["readable"] if "readable" in df.columns else ""

        out = {
            "row_index": idx,
            "model": mdl,
            "input_mapping": input_mapping,
            "scenario": scenario_text,
            "input_conditions": input_conditions,
            "label": label,
            "original_formula": original_formula,
            "generated_formula": formula,              # as returned
            "runtime_nl2pl_sec": dur,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": tt,
            "generated_mapping": mapping_json,         # JSON string
        }
        rows.append(out)

        print(f"[NL→PL] row {idx} | model={mdl} time={dur}s")

    cols = [
        "row_index","model","input_mapping","scenario","input_conditions",
        "label","original_formula","generated_formula",
        "runtime_nl2pl_sec","prompt_tokens","completion_tokens","total_tokens",
        "generated_mapping",
    ]
    pd.DataFrame(rows)[cols].to_csv(CSV_FILE, index=False)
    print("CSV →", CSV_FILE.resolve())

if __name__ == "__main__":
    main()
