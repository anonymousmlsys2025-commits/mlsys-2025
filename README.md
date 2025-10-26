# VeriTrans

Lightweight, reproducible pipeline for turning **natural language (NL)** specs into **propositional logic (PL)**, validating via **CNF + SAT**, and checking round-trip fidelity with **PL→NL**.

## Pipeline

1. **Stage 1 — NL→PL (deterministic)**
   - Structured prompt enumerates the symbol inventory `𝒱`.
   - LLM (`temperature=0`) returns **Variable Mapping (JSON)** + **Logical Formula** (one line).
   - Script extracts the first valid JSON block and first non-empty formula line; logs time/tokens.

2. **Stage 2 — PL→NL (deterministic)**
   - Uses the mapping + formula to reconstruct English (“Reconstructed Conditions:”).
   - Computes **TF–IDF cosine** vs the original NL constraints.

3. **Stage 3 — PL→CNF→SAT (symbolic)**
   - Canonicalizes `x(i,j,k)→x_i_j_k`, normalizes operators (`¬ ∧ ∨ → ↔` → `!, &, |, ->, <->`).
   - Tokenize → shunting-yard (RPN) → AST → Tseitin CNF → DIMACS.
   - Solves with MiniSAT22; outputs SAT/UNSAT and CNF.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install "openai>=1.0.0" pandas python-sat
```
1. **Hardcode your OpenAI API key** (Stages 1–2)
Edit both `nl_pl.py` and `pl_nl.py`:

```python
from openai import OpenAI

MODEL_ID = "gpt-4.1-mini-2025-04-14"
TIMEOUT  = 600

API_KEY  = "sk-REPLACE_ME"  # <-- paste your key here
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set.")

client = OpenAI(api_key=API_KEY)
```
2. **Run the pipeline**
**Stage 1** — NL→PL
```bash
# Stage 1 expects: satbench_test.csv (or your INPUT_CSV) with at least a `conditions` column.
python nl_pl.py
# outputs: nl2pl_<model>_summary.csv
```
**Stage 2** — PL→NL
```bash
python pl_nl.py
# reads:   nl2pl_<model>_summary.csv
# outputs: pl2nl_<model>_summary.csv
#          pl2nl_<model>_log.txt
```
**Stage 3** — PL→CNF→SAT
```bash
python pl_sat.py
# reads:   pl2nl_<model>_summary.csv
# outputs: pl2cnf_<tag>_annotated.csv
#          pl2cnf_<tag>_incorrect.csv
```
## Key Outputs

- **Stage 1:** `generated_formula` (one line), `generated_mapping` (JSON), `input_conditions`, `scenario`, `runtime_nl2pl_sec`, token counts, `label` (if provided), `model`
- **Stage 2:** `reconstructed_nl`, `similarity_percent`, `runtime_pl2nl_sec`, combined totals with Stage 1, `model_pl2nl`, log file
- **Stage 3:** `pred_from_script` (SAT/UNSAT), `match_original`, `cnf_dimacs`, `pct_correct_sat_only`, `pct_correct_unsat_only`, `pct_correct_overall`
