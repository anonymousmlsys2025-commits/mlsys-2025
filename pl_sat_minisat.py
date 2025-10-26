#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from pysat.solvers import Minisat22


# =========================
# ======== CONFIG =========
# =========================
INPUT_CSV   = "pl2nl_gpt_4.1_mini_2025_04_14_summary.csv"
OUTPUT_CSV  = None
FORMULA_COL = "generated_formula"
LABEL_COL   = "label"
VERBOSE     = True
# =========================


# ---------- Canonicalization & normalization ----------
INDEXED_VAR_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\(\s*([^)]+?)\s*\)')

def canonicalize_indexed_vars(s: str) -> str:
    def repl(m: re.Match) -> str:
        name = m.group(1)
        args = [a.strip().replace(" ", "") for a in m.group(2).split(",") if a.strip()]
        if not args:
            return name
        safe = [re.sub(r"[^A-Za-z0-9_-]", "_", a) for a in args]
        return f"{name}_{'_'.join(safe)}"
    prev = None
    while s != prev:
        prev = s
        s = INDEXED_VAR_RE.sub(repl, s)
    return s

def normalize_symbols(s: str) -> str:
    repls = {
        "¬": " ! ", "~": " ! ", " NOT ": " ! ", " not ": " ! ",
        "∧": " & ", " AND ": " & ", " and ": " & ",
        "∨": " | ", " OR ": " | ", " or ": " | ",
        "↔": " <-> ", "<=>": " <-> ", "<->": " <-> ",
        "=>": " -> ", "IMPLIES": " -> ", " implies ": " -> ",
    }
    s = s.replace("(", " ( ").replace(")", " ) ")
    s = f" {s} "
    for k, v in repls.items():
        s = s.replace(k, v)
    return re.sub(r"\s+", " ", s).strip()


# ---------- Tokenizer ----------
IDENT_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')

def tokenize(s: str) -> List[str]:
    return s.split()

def is_ident(tok: str) -> bool:
    return bool(IDENT_RE.match(tok))


# ---------- Shunting-yard (to RPN) ----------
PREC = {"!": 5, "&": 4, "|": 3, "->": 2, "<->": 1}
ASSOC = {"!": "right", "&": "left", "|": "left", "->": "right", "<->": "left"}

def to_rpn(tokens: List[str]) -> List[str]:
    out: List[str] = []
    ops: List[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if is_ident(t):
            out.append(t)
        elif t in ("!", "&", "|", "->", "<->"):
            if t == "!" and (i == 0 or tokens[i-1] in ("(", "&", "|", "->", "<->", "!")):
                while ops and ops[-1] != "(" and (PREC[ops[-1]] > PREC["!"] or
                      (PREC[ops[-1]] == PREC["!"] and ASSOC["!"] == "left")):
                    out.append(ops.pop())
                ops.append("!")
            else:
                while ops and ops[-1] != "(" and (PREC[ops[-1]] > PREC[t] or
                      (PREC[ops[-1]] == PREC[t] and ASSOC[t] == "left")):
                    out.append(ops.pop())
                ops.append(t)
        elif t == "(":
            ops.append(t)
        elif t == ")":
            while ops and ops[-1] != "(":
                out.append(ops.pop())
            if not ops:
                raise ValueError("Mismatched ')'")
            ops.pop()
        else:
            raise ValueError(f"Unknown token: {t}")
        i += 1
    while ops:
        op = ops.pop()
        if op in ("(", ")"):
            raise ValueError("Mismatched parentheses")
        out.append(op)
    return out


# ---------- AST nodes ----------
@dataclass
class Node:
    pass

@dataclass
class Var(Node):
    name: str

@dataclass
class Not(Node):
    a: Node

@dataclass
class And(Node):
    a: Node
    b: Node

@dataclass
class Or(Node):
    a: Node
    b: Node

@dataclass
class Impl(Node):
    a: Node
    b: Node

@dataclass
class Iff(Node):
    a: Node
    b: Node


def rpn_to_ast(rpn: List[str]) -> Node:
    st: List[Node] = []
    for t in rpn:
        if is_ident(t):
            st.append(Var(t))
        elif t == "!":
            if not st: raise ValueError("NOT missing operand")
            st.append(Not(st.pop()))
        elif t in ("&", "|", "->", "<->"):
            if len(st) < 2: raise ValueError(f"Operator {t} missing operands")
            b = st.pop(); a = st.pop()
            if t == "&": st.append(And(a, b))
            elif t == "|": st.append(Or(a, b))
            elif t == "->": st.append(Impl(a, b))
            elif t == "<->": st.append(Iff(a, b))
        else:
            raise ValueError(f"Unknown RPN token {t}")
    if len(st) != 1:
        raise ValueError("Bad expression")
    return st[0]


# ---------- Tseitin CNF ----------
def tseitin_cnf(expr: Node) -> Tuple[List[List[str]], str]:
    clauses: List[List[str]] = []
    counter = [0]

    def fresh() -> str:
        counter[0] += 1
        return f"_aux_{counter[0]}"

    def encode(node: Node) -> str:
        if isinstance(node, Var):
            return node.name
        if isinstance(node, Not):
            a = encode(node.a)
            v = fresh()
            clauses.append([f"!{v}", f"!{a}"])
            clauses.append([f"{v}", f"{a}"])
            return v
        if isinstance(node, And):
            a = encode(node.a); b = encode(node.b)
            v = fresh()
            clauses.append([f"!{v}", f"{a}"])
            clauses.append([f"!{v}", f"{b}"])
            clauses.append([f"!{a}", f"!{b}", f"{v}"])
            return v
        if isinstance(node, Or):
            a = encode(node.a); b = encode(node.b)
            v = fresh()
            clauses.append([f"!{a}", f"{v}"])
            clauses.append([f"!{b}", f"{v}"])
            clauses.append([f"!{v}", f"{a}", f"{b}"])
            return v
        if isinstance(node, Impl):
            a = encode(node.a); b = encode(node.b)
            v = fresh()
            clauses.append([f"{a}", f"{v}"])
            clauses.append([f"!{b}", f"{v}"])
            clauses.append([f"!{v}", f"!{a}", f"{b}"])
            return v
        if isinstance(node, Iff):
            a = encode(node.a); b = encode(node.b)
            v = fresh()
            clauses.append([f"!{v}", f"!{a}", f"{b}"])
            clauses.append([f"!{v}", f"!{b}", f"{a}"])
            clauses.append([f"!{a}", f"!{b}", f"{v}"])
            clauses.append([f"{a}", f"{b}", f"{v}"])
            return v
        raise ValueError("Unknown node type")
    top = encode(expr)
    clauses.append([top])
    return clauses, top


# ---------- Map string literals to ints (DIMACS-style) ----------
def lit_to_int_mapper():
    sym2id: Dict[str, int] = {}
    def convert(lit: str) -> int:
        neg = lit.startswith("!")
        name = lit[1:] if neg else lit
        if name not in sym2id:
            sym2id[name] = len(sym2id) + 1
        v = sym2id[name]
        return -v if neg else v
    return convert, sym2id


def solve_with_minisat(clauses: List[List[int]]) -> bool:
    with Minisat22(bootstrap_with=clauses) as solver:
        result = solver.solve()
    return bool(result)


MODEL_COL_CANDS = ["model_pl2nl", "model_nl2pl", "model_name", "model"]

def normalize_gt(v: Any) -> Optional[bool]:
    """
    Normalize ground truth label to boolean.
    Returns True for SAT, False for UNSAT, None for unknown/invalid.
    """
    if v is None or pd.isna(v):
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0

    s = str(v).strip().lower()

    if s in {"sat", "satisfiable", "true", "1"}:
        return True
    if s in {"unsat", "unsatisfiable", "false", "0"}:
        return False

    if "unsat" in s:
        return False
    if "sat" in s:
        return True

    return None


def infer_model_name(df: pd.DataFrame) -> Optional[str]:
    for cand in MODEL_COL_CANDS:
        for col in df.columns:
            if col.lower() == cand:
                value = df[col].dropna().astype(str).head(1)
                if not value.empty:
                    name = value.iloc[0].strip()
                    if name:
                        return name
    for col in df.columns:
        if "model" in col.lower():
            value = df[col].dropna().astype(str).head(1)
            if not value.empty:
                name = value.iloc[0].strip()
                if name:
                    return name
    return None


def sanitize_filename_component(name: str) -> str:
    cleaned = re.sub(r"\s+", "_", name.strip())
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "unknown"

# ---------- End-to-end per-row processing ----------
def formula_to_clauses(formula: str, verbose: bool = False) -> Tuple[List[List[int]], Dict[str, int]]:
    if not isinstance(formula, str):
        raise ValueError("Formula must be a string")
    # 1) canonicalize indexed vars (prevent collapsing)
    canon = canonicalize_indexed_vars(formula)
    if verbose:
        print(f"  Canonicalized: {canon}")
    # 2) normalize symbols/words
    norm = normalize_symbols(canon)
    if verbose:
        print(f"  Normalized: {norm}")
    # 3) tokenize → RPN → AST
    tokens = tokenize(norm)
    rpn = to_rpn(tokens)
    ast_root = rpn_to_ast(rpn)
    # 4) Tseitin → clauses over string literals
    clauses_str, _top = tseitin_cnf(ast_root)
    # 5) map to integers
    to_int, sym2id = lit_to_int_mapper()
    clauses_int = [[to_int(lit) for lit in cl] for cl in clauses_str]
    if verbose:
        print(f"  Variables mapped: {sym2id}")
        print(f"  Number of clauses: {len(clauses_int)}")
    return clauses_int, sym2id


def clauses_to_dimacs(clauses: List[List[int]], sym2id: Dict[str, int]) -> str:
    num_vars = len(sym2id)
    num_clauses = len(clauses)
    lines = [f"p cnf {num_vars} {num_clauses}"]
    for name, idx in sorted(sym2id.items(), key=lambda kv: kv[1]):
        lines.append(f"c {idx} {name}")
    for clause in clauses:
        if clause:
            lines.append(" ".join(str(lit) for lit in clause) + " 0")
        else:
            lines.append("0")
    return "\n".join(lines)


# ---------- Main (hard-coded IO) ----------
def main():
    t0 = time.time()
    in_path = Path(INPUT_CSV)
    if not in_path.exists():
        raise SystemExit(f"ERROR: INPUT_CSV not found: {in_path}")

    df = pd.read_csv(in_path)

    if OUTPUT_CSV:
        out_path = Path(OUTPUT_CSV)
        incorrect_path = out_path.with_name(out_path.stem + "_incorrect.csv")
    else:
        inferred = infer_model_name(df) or in_path.stem
        model_tag = sanitize_filename_component(inferred)
        prefix = f"pl2cnf_{model_tag}"
        out_path = in_path.with_name(f"{prefix}_annotated.csv")
        incorrect_path = in_path.with_name(f"{prefix}_incorrect.csv")

    if FORMULA_COL not in df.columns:
        raise SystemExit(f"ERROR: Formula column '{FORMULA_COL}' not found in input CSV.")
    col_formula = FORMULA_COL

    col_gt = LABEL_COL if LABEL_COL in df.columns else None

    if VERBOSE:
        print({"formula_col": col_formula, "gt_col": col_gt})

    preds: List[Optional[str]] = []
    matches: List[Optional[bool]] = []
    gt_bools: List[Optional[bool]] = []
    dimacs_texts: List[Optional[str]] = []
    errors = 0

    for i, row in df.iterrows():
        s = row[col_formula]
        gt = normalize_gt(row[col_gt]) if col_gt else None
        gt_bools.append(gt)

        if not isinstance(s, str) or not s.strip():
            preds.append(None); matches.append(None); dimacs_texts.append(None); errors += 1
            if VERBOSE:
                print(f"[Row {i}] ERROR: Empty/missing formula")
            continue

        try:
            if VERBOSE:
                print(f"\n[Row {i}] Processing formula:")
                print(f"  Original: {s[:100]}..." if len(s) > 100 else f"  Original: {s}")
            clauses, sym2id = formula_to_clauses(s, verbose=VERBOSE)
            dimacs_str = clauses_to_dimacs(clauses, sym2id)
            sat = solve_with_minisat(clauses)
            pred_result = "SAT" if sat else "UNSAT"
            preds.append(pred_result)
            matches.append(None if gt is None else (sat == gt))
            dimacs_texts.append(dimacs_str)
            if VERBOSE:
                gt_str = "SAT" if gt else "UNSAT" if gt is not None else "N/A"
                match_str = "✓" if matches[-1] else "✗" if matches[-1] is not None else "N/A"
                print(f"  Result: {pred_result} | Ground truth: {gt_str} | Match: {match_str}")
        except ValueError as e:
            preds.append(None); matches.append(None); dimacs_texts.append(None); errors += 1
            if VERBOSE:
                print(f"[Row {i}] PARSE ERROR: {e}")
                print(f"  Formula was: {s}")
        except Exception as e:
            preds.append(None); matches.append(None); dimacs_texts.append(None); errors += 1
            if VERBOSE:
                print(f"[Row {i}] UNEXPECTED ERROR ({type(e).__name__}): {e}")
                print(f"  Formula was: {s}")

    # Build annotated DF
    out = df.copy()
    out["pred_from_script"] = preds
    out["match_original"]   = matches
    out["cnf_dimacs"]       = dimacs_texts

    overall_pairs = [(m, g) for m, g in zip(matches, gt_bools) if g is not None and m in (True, False)]

    def pct(values: List[bool]) -> Optional[float]:
        return round(sum(values) / len(values), 6) if values else None

    if overall_pairs:
        sat_pct = pct([m for m, g in overall_pairs if g is True])
        uns_pct = pct([m for m, g in overall_pairs if g is False])
        overall_pct = pct([m for m, _ in overall_pairs])
    else:
        sat_pct = uns_pct = overall_pct = None

    out["pct_correct_sat_only"]   = sat_pct
    out["pct_correct_unsat_only"] = uns_pct
    out["pct_correct_overall"]    = overall_pct

    # Reorder columns so predictions sit directly before the original label.
    base_cols = [c for c in df.columns if c != col_gt]
    ordered = base_cols + ["pred_from_script"]
    if col_gt:
        ordered.append(col_gt)
    ordered.extend(["match_original", "cnf_dimacs",
                    "pct_correct_sat_only", "pct_correct_unsat_only", "pct_correct_overall"])
    remaining = [c for c in out.columns if c not in ordered]
    out = out[ordered + remaining]

    out.to_csv(out_path, index=False)

    try:
        incorrect = out[out["match_original"] == False].copy()
        if col_gt:
            incorrect[col_gt] = pd.Categorical(incorrect[col_gt], categories=["SAT","UNSAT"], ordered=True)
            incorrect = incorrect.sort_values([col_gt], kind="stable")
        incorrect.to_csv(incorrect_path, index=False)
    except Exception:
        pass

    total = len(out)
    correct = int(sum(1 for m, _ in overall_pairs if m is True))
    gt_rows = int(sum(1 for g in gt_bools if g is not None))
    accuracy = overall_pct
    print({
        "rows_total": total,
        "rows_with_error": errors,
        "gt_rows": gt_rows,
        "correct_matches": correct,
        "accuracy_on_gt_rows": accuracy,
        "out": str(out_path),
        "formula_col": col_formula,
        "gt_col": col_gt,
        "runtime_sec": round(time.time() - t0, 3),
    })


if __name__ == "__main__":
    main()
