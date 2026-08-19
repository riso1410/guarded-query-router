"""Summarise oe4/results/results.csv into a markdown table.

usage: .venv/bin/python summarize.py [results/results.csv] [--md out.md] [--rules argmax,tau] [--variants oe4,ctrl3]
default: plain 4-class argmax rows only
"""
import json
import sys

import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "results/results.csv"
out_md = sys.argv[sys.argv.index("--md") + 1] if "--md" in sys.argv else None

df = pd.read_csv(path)
rules = sys.argv[sys.argv.index("--rules") + 1].split(",") if "--rules" in sys.argv else ["argmax"]
df = df[df.rule.isin(rules)]
if "--variants" in sys.argv:
    df = df[df.variant.isin(sys.argv[sys.argv.index("--variants") + 1].split(","))]
else:
    df = df[df.variant == "oe4"]
# keep the latest row per (model_key, embedding, variant, rule, seed)
df = df.sort_values("timestamp").drop_duplicates(
    ["model_key", "embedding", "variant", "rule", "seed"], keep="last")

order_m = ["xgb", "svm", "fasttext", "widemlp", "bert", "modernbert"]
order_e = ["baai", "mini", "tf_idf", "own"]
df["_m"] = df.model_key.map({k: i for i, k in enumerate(order_m)})
df["_e"] = df.embedding.map({k: i for i, k in enumerate(order_e)})
df = df.sort_values(["_m", "_e", "variant", "rule"])

lines = ["| model | embedding | variant | rule | ID acc | OOD acc | **GQR** | τ | aux-val rej | latency ms | train s |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for _, r in df.iterrows():
    lines.append(f"| {r.model} | {r.embedding} | {r.variant} | {r.rule} | {r.id_acc:.4f} | {r.ood_acc:.4f} | "
                 f"**{r.gqr_score:.4f}** | {'' if pd.isna(r.tau) else f'{r.tau:.3f}'} | "
                 f"{'' if pd.isna(r.aux_val_reject) else f'{r.aux_val_reject:.3f}'} | "
                 f"{r.avg_latency_s * 1e3:.2f} | {r.train_time_s:.0f} |")

# best-per-model comparison: oe4 (best rule) vs ctrl3/msp — only when the control was requested
if (df.variant == "ctrl3").any():
  lines += ["", "### Best rule per model: 4th background class (oe4) vs 3-class control (ctrl3, MSP reject)", "",
          "| model | embedding | ctrl3/msp GQR | oe4 best GQR (rule) | Δ GQR | oe4 ID | oe4 OOD |", "|---|---|---|---|---|---|---|"]
for (mk, em), g in (df.groupby(["model_key", "embedding"], sort=False) if (df.variant == "ctrl3").any() else []):
    c = g[(g.variant == "ctrl3") & (g.rule == "msp")]
    o = g[g.variant == "oe4"]
    if o.empty:
        continue
    ob = o.loc[o.gqr_score.idxmax()]
    cg = float(c.gqr_score.iloc[0]) if len(c) else float("nan")
    lines.append(f"| {ob.model} | {em} | {cg:.4f} | **{ob.gqr_score:.4f}** ({ob.rule}) | "
                 f"{ob.gqr_score - cg:+.4f} | {ob.id_acc:.4f} | {ob.ood_acc:.4f} |")

# per-OOD-dataset accuracy for the oe4 best rows
lines += ["", "### Per-OOD-dataset accuracy (oe4, best rule)", ""]
first = True
for (mk, em), g in df.groupby(["model_key", "embedding"], sort=False):
    o = g[g.variant == "oe4"]
    if o.empty:
        continue
    ob = o.loc[o.gqr_score.idxmax()]
    ds = json.loads(ob.dataset_acc)
    if first:
        lines.append("| model | embedding | rule | " + " | ".join(ds.keys()) + " |")
        lines.append("|---|---|---|" + "---|" * len(ds))
        first = False
    lines.append(f"| {ob.model} | {em} | {ob.rule} | " + " | ".join(f"{v:.3f}" for v in ds.values()) + " |")

text = "\n".join(lines)
print(text)
if out_md:
    open(out_md, "w").write(text + "\n")
