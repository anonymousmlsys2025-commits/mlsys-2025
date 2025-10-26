import time, re, math, pathlib, pandas as pd, json
from collections import Counter
from openai import OpenAI

# ───────────────────────── CONFIG ─────────────────────────
MODEL_ID = "gpt-4.1-mini-2025-04-14"
TIMEOUT  = 600

API_KEY  = "sk-REPLACE_ME"
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set. Export it or put it in openai.env and load before running.")

client = OpenAI(api_key=API_KEY)
suffix = MODEL_ID.replace("-", "_")

CSV_IN   = pathlib.Path(f"nl2pl_{suffix}_summary.csv")
LOG_FILE = pathlib.Path(f"pl2nl_{suffix}_log.txt")
CSV_OUT  = pathlib.Path(f"pl2nl_{suffix}_summary.csv")

# ───────────────────── PL→NL PROMPT ──────────────────────
PL2NL_PROMPT = """### Task
Convert the propositional-logic statement below—together with its variable mapping—into clear English sentences.

### Guidelines
- Use the mapping to expand each variable name exactly (e.g., x(0,) → "Whiskers visits the porch").
- Respect the operators and parentheses.
- Split the formula at top-level conjunctions (∧): output one short sentence per conjunct.
- Render disjunctions (∨) with “either … or …” inside that sentence; do not merge separate conjuncts into one sentence.
- Preserve clause order from the formula.
- Use the same wording as much as possible from the input conditions.
- Do not paraphrase beyond substituting variable names with their mapped labels.
- No LaTeX, no math markup.

### Output (use exactly this heading)
Reconstructed Conditions:
1. <first clause in English>
2. <second clause in English>
...

Variable Mapping (lines or JSON expanded to lines):
{mapping}

Logical Formula:
{formula}
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


def extract_final_sentence(block: str) -> str:
    key = "Reconstructed Conditions:"
    if key not in block:
        return ""
    after = block.split(key, 1)[1]
    lines = []
    for ln in after.splitlines():
        s = ln.strip()
        if not s:
            if lines: break
            continue
        if s.lower().startswith(("variable mapping:", "logical formula:")):
            break
        lines.append(s)
    return " ".join(lines).strip()

# ────────────── Similarity ──────────────
STOPWORDS = {
    "the","a","an","and","or","but","if","then","else","for","to","of","in","on","at",
    "by","with","as","is","are","was","were","be","been","being","it","its","that",
    "this","these","those","from","over","under","into","out","about","so","such",
    "than","not","no","do","does","did","done","can","could","would","should","might",
    "may","will","shall","you","your","yours","we","our","ours","they","their","theirs",
    "he","him","his","she","her","hers","i","me","my","mine","us","them","there","here"
}
WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)

def tokenize(text: str):
    toks = [t.lower() for t in WORD_RE.findall(str(text))]
    return [t for t in toks if t not in STOPWORDS]

def build_idf(corpus_tokens):
    N = len(corpus_tokens)
    df = Counter()
    for toks in corpus_tokens:
        df.update(set(toks))
    # +1 smoothing
    return {t: math.log((1 + N) / (1 + df[t])) + 1.0 for t in df}

def tfidf_vec(tokens, idf):
    tf = Counter(tokens)
    vec = {t: tf[t] * idf.get(t, 0.0) for t in tf}
    norm = math.sqrt(sum(w*w for w in vec.values())) or 1.0
    return {t: w / norm for t, w in vec.items()}

def cosine(u, v):
    if len(u) < len(v):
        u, v = v, u
    return sum(w * v.get(t, 0.0) for t, w in u.items())

# ────────────────────────── MAIN ─────────────────────────
def main():
    if not CSV_IN.exists():
        raise FileNotFoundError(f"Input CSV not found: {CSV_IN}")

    df = pd.read_csv(CSV_IN)

    has_gen = "generated_formula" in df.columns
    has_old = "formula" in df.columns
    if not (has_gen or has_old):
        raise ValueError(f"Missing required column 'generated_formula' or 'formula'. Present: {list(df.columns)}")

    formula_col = "generated_formula" if has_gen else "formula"
    df = df[df[formula_col].astype(str).str.strip().ne("")].reset_index(drop=True)

    has_input_mapping    = "input_mapping" in df.columns
    has_input_conditions = "input_conditions" in df.columns
    has_parsed_mapping   = "mapping" in df.columns

    log_lines, rows = [], []

    for i, r in df.iterrows():
        model_nl2pl = str(r.get("model", ""))  # rename on output as model_nl2pl
        generated_formula = str(r.get("generated_formula", r.get("formula", "")) or "").strip()
        original_formula  = str(r.get("original_formula", r.get("gold_formula", "")) or "").strip()
        pt_nl2pl = int(r.get("prompt_tokens", 0) or 0)
        ct_nl2pl = int(r.get("completion_tokens", 0) or 0)
        tt_nl2pl = int(r.get("total_tokens", 0) or 0)
        rt_nl2pl = float(r.get("runtime_nl2pl_sec", 0.0) or 0.0)
        formula = generated_formula
        mapping = ""
        mapping = ""
        if has_input_mapping and isinstance(r.get("input_mapping"), str):
            mapping = r["input_mapping"].strip()
        elif has_parsed_mapping and isinstance(r.get("mapping"), str):
            mapping = r["mapping"].strip()
        genmap_json = str(r.get("generated_mapping", "") or "").strip()
        pairs_text = ""
        if genmap_json:
            try:
                d = json.loads(genmap_json)
                items = sorted(
                    d.items(),
                    key=lambda kv: int(kv[1]) if str(kv[1]).isdigit() else 10**9
                )
                pairs_text = "\n".join(f"{v} = {k}" for k, v in items)
            except Exception:
                pairs_text = genmap_json

        if pairs_text:
            mapping = (mapping + ("\n" if mapping else "") + pairs_text).strip()

        input_conditions = str(r.get("input_conditions", "") or "").strip()

        short_preview = generated_formula.replace("\n", " ")[:60] + ("…" if len(generated_formula) > 60 else "")
        print(f"[PL→NL] row {i} | '{short_preview}'")

        try:
            resp, t_rev, model_used, pt_pl2nl, ct_pl2nl, tt_pl2nl = call_openai(
                PL2NL_PROMPT.format(mapping=mapping, formula=formula)
            )
            recon_nl = extract_final_sentence(resp)
            print(f"  Model: {model_used}  Time: {t_rev}s")

            # Two-step combined totals
            runtime_total_sec        = rt_nl2pl + t_rev
            prompt_tokens_total      = pt_nl2pl + pt_pl2nl
            completion_tokens_total  = ct_nl2pl + ct_pl2nl
            total_tokens_total       = tt_nl2pl + tt_pl2nl

        except Exception as e:
            print(f"  ERROR: {e}")
            log_lines.append(f"[ERROR] row={i}|{e}\n")
            continue

        log_lines.append(
            f"==== row {i} | {model_used} ====\n"
            f"INPUT MAPPING:\n{mapping}\n"
            f"INPUT FORMULA:\n{formula}\n"
            f"Elapsed: {t_rev}s\n"
            f"RAW OUTPUT:\n{resp}\n"
            f"Reconstructed NL:\n{recon_nl}\n"
            + "-"*40 + "\n"
        )

        out = {
            "row_index": r.get("row_index", i),
            "model_nl2pl": model_nl2pl,
            "model_pl2nl": model_used,
            "input_mapping": mapping,
            "input_conditions": input_conditions,
            "label": r.get("label", ""),
            "original_formula": original_formula,
            "generated_formula": generated_formula,
            "reconstructed_nl": recon_nl,
            "similarity_percent": None,  # filled later
            "runtime_nl2pl_sec": rt_nl2pl,
            "runtime_pl2nl_sec": t_rev,
            "runtime_total_sec": runtime_total_sec,
            "prompt_tokens_total": prompt_tokens_total,
            "completion_tokens_total": completion_tokens_total,
            "total_tokens_total": total_tokens_total,
        }
        rows.append(out)
    COLS = [
        "row_index",
        "model_nl2pl",
        "model_pl2nl",
        "input_mapping",
        "input_conditions",
        "label",
        "original_formula",
        "generated_formula",
        "reconstructed_nl",
        "similarity_percent",
        "runtime_nl2pl_sec",
        "runtime_pl2nl_sec",
        "runtime_total_sec",
        "prompt_tokens_total",
        "completion_tokens_total",
        "total_tokens_total",
    ]

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        df_out = pd.DataFrame(columns=COLS)
    if "input_conditions" in df_out.columns and "reconstructed_nl" in df_out.columns:
        tok_orig = [tokenize(t) for t in df_out["input_conditions"].fillna("").astype(str)]
        tok_recon= [tokenize(t) for t in df_out["reconstructed_nl"].fillna("").astype(str)]
        idf = build_idf(tok_orig + tok_recon)
        sims = [round(cosine(tfidf_vec(o,idf), tfidf_vec(r,idf)) * 100.0, 2) for o, r in zip(tok_orig, tok_recon)]
        df_out["similarity_percent"] = sims

    df_out = df_out[COLS]
    totals = {
        "row_index": "TOTALS",
        "model_nl2pl": "",
        "model_pl2nl": "",
        "input_mapping": "",
        "input_conditions": "",
        "label": "",
        "original_formula": "",
        "generated_formula": "",
        "reconstructed_nl": "",
        "similarity_percent": "",
        "runtime_nl2pl_sec": df_out["runtime_nl2pl_sec"].sum(),
        "runtime_pl2nl_sec": df_out["runtime_pl2nl_sec"].sum(),
        "runtime_total_sec": df_out["runtime_total_sec"].sum(),
        "prompt_tokens_total": df_out["prompt_tokens_total"].sum(),
        "completion_tokens_total": df_out["completion_tokens_total"].sum(),
        "total_tokens_total": df_out["total_tokens_total"].sum(),
    }
    df_out = pd.concat([df_out, pd.DataFrame([totals])], ignore_index=True)

    LOG_FILE.write_text("".join(log_lines), encoding="utf-8")
    df_out.to_csv(CSV_OUT, index=False)

    print(f"Done.\nLog  → {LOG_FILE.resolve()}\nCSV → {CSV_OUT.resolve()}")

if __name__ == "__main__":
    main()
