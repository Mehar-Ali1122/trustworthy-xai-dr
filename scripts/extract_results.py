#!/usr/bin/env python3
"""Extract every reported number from the saved outputs of the four executed notebooks.

Nothing is re-computed: the script only parses the text that the notebooks printed
when they were run, so the CSV files it writes are a direct transcript of the
executed cells.

Usage:
    python scripts/extract_results.py            # writes results/*.csv
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
OUT_DIR = ROOT / "results"

RUNS = {
    "Swin-T_IDRiD": "01_SwinT_IDRiD.ipynb",
    "ViT-B16_IDRiD": "02_ViT-B16_IDRiD.ipynb",
    "Swin-T_APTOS": "03_SwinT_APTOS.ipynb",
    "ViT-B16_APTOS": "04_ViT-B16_APTOS.ipynb",
}

NUM = r"-?\d+(?:\.\d+)?"


def cell_outputs(nb: dict) -> list[tuple[int, str]]:
    """Return (cell_number_1_based, concatenated_text_output) for every code cell."""
    out = []
    for i, cell in enumerate(nb["cells"], start=1):
        text = ""
        for o in cell.get("outputs", []):
            if "text" in o:
                text += "".join(o["text"])
            elif "data" in o and "text/plain" in o["data"]:
                text += "".join(o["data"]["text/plain"])
        out.append((i, text))
    return out


def find(outputs, marker: str) -> tuple[int, str]:
    hits = [(i, t) for i, t in outputs if marker in t]
    if not hits:
        raise ValueError(f"marker not found: {marker!r}")
    return hits[-1]


def parse_grading(outputs):
    c_tr, t_tr = find(outputs, "Best validation QWK")
    m = re.search(rf"Best validation QWK = ({NUM}) \(epoch (\d+)\)", t_tr)
    c_te, t_te = find(outputs, "TEST SET")
    acc = re.search(rf"Accuracy\s*:\s*({NUM})", t_te).group(1)
    qwk = re.search(rf"QWK\s*:\s*({NUM})", t_te).group(1)
    n = re.search(r"\bn\s*:\s*(\d+)", t_te).group(1)
    per_class = re.findall(rf"Class (\d)\s+({NUM})\s+({NUM})\s+({NUM})\s+(\d+)", t_te)
    return ({"best_val_qwk": m.group(1), "best_epoch": m.group(2), "test_acc": acc,
             "test_qwk": qwk, "n_test": n, "cell_train": c_tr, "cell_test": c_te},
            [{"grade": g, "precision": p, "recall": r, "f1": f, "support": s, "cell": c_te}
             for g, p, r, f, s in per_class])


ROW = re.compile(
    rf"^(?P<name>\S.*?)\s+(?P<ins>{NUM})\s+(?P<dele>{NUM})\s+(?P<frank>\d+)\s+"
    rf"(?P<rho>{NUM})\s+(?P<rho_full>{NUM})\s+(?P<sanity>YES|NO)"
    rf"(?:\s+(?P<plaus>{NUM})\s+(?P<prank>\d+))?\s*$")


def parse_scorecard(outputs):
    c, text = find(outputs, "TRUSTWORTHINESS SCORECARD")
    block = text[text.index("TRUSTWORTHINESS SCORECARD"):]
    rows = []
    for line in block.splitlines():
        m = ROW.match(line.rstrip())
        if m:
            d = m.groupdict()
            rows.append({"explainer": d["name"].strip(), "insertion_auc": d["ins"],
                         "deletion_auc": d["dele"], "mean_abs_rho": d["rho"],
                         "abs_rho_full_randomization": d["rho_full"], "sanity_pass": d["sanity"],
                         "plausibility_auprc_ratio_union": d["plaus"] or "N/A", "cell": c})
    if not rows:
        raise ValueError("scorecard rows not parsed")
    return rows


def parse_per_lesion(outputs):
    try:
        c, text = find(outputs, "PLAUSIBILITY — per lesion type")
    except ValueError:
        return []
    block = text[text.index("PLAUSIBILITY — per lesion type"):]
    rows = []
    for line in block.splitlines():
        m = re.match(rf"^(\S.*?)\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})\s*$", line.rstrip())
        if m:
            rows.append({"explainer": m.group(1).strip(), "MA": m.group(2), "HE": m.group(3),
                         "EX": m.group(4), "SE": m.group(5), "mean": m.group(6), "cell": c})
    return rows


def write(name, rows):
    if not rows:
        return
    path = OUT_DIR / name
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path.relative_to(ROOT)} ({len(rows)} rows)")


def main():
    grading, per_class, scores, lesions = [], [], [], []
    for run, fname in RUNS.items():
        nb = json.loads((NB_DIR / fname).read_text(encoding="utf-8"))
        outs = cell_outputs(nb)
        g, pc = parse_grading(outs)
        grading.append({"run": run, "notebook": fname, **g})
        per_class += [{"run": run, **r} for r in pc]
        scores += [{"run": run, **r} for r in parse_scorecard(outs)]
        lesions += [{"run": run, **r} for r in parse_per_lesion(outs)]
    write("grading.csv", grading)
    write("per_class_test.csv", per_class)
    write("scorecards.csv", scores)
    write("plausibility_per_lesion.csv", lesions)


if __name__ == "__main__":
    main()
