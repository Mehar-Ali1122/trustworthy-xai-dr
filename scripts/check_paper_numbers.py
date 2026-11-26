#!/usr/bin/env python3
"""Check every value in Tables 3-6 of the manuscript against the executed notebooks.

Run `python scripts/extract_results.py` first. The expected values below are copied
from the manuscript tables. A value passes when it equals the notebook value after
the rounding used in the paper (a tolerance of half a unit in the paper's last digit,
plus half a unit in the notebook's printed last digit, covers printed-then-rounded values).

Exit code 0 = every value matches.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

NAME = {"Grad-CAM": "GradCAM", "Integrated Grad.": "IntegratedGradients",
        "Transf. Attr.": "TransformerAttribution", "Attn. Rollout": "AttentionRollout",
        "Input-Grad.": "InputGrad (baseline)", "TokenNorm": "TokenNorm (Swin surrogate)"}

# ---- Table 4: grading performance (Val kappa, Epoch, Acc., Test kappa) ----
TABLE_IV = {
    "Swin-T_IDRiD": ("0.7781", "11", "0.5049", "0.6002"),
    "ViT-B16_IDRiD": ("0.8563", "34", "0.5631", "0.6454"),
    "Swin-T_APTOS": ("0.9204", "26", "0.8415", "0.8858"),
    "ViT-B16_APTOS": ("0.9083", "14", "0.8415", "0.9070"),
}
# ---- Table 5: insertion AUC, mean |rho|, sanity verdict ----
TABLE_V = {
    "ViT-B16_IDRiD": [("Grad-CAM", "0.531", "0.125", "pass"), ("Integrated Grad.", "0.601", "0.673", "FAIL"),
                      ("Transf. Attr.", "0.511", "0.125", "pass"), ("Attn. Rollout", "0.487", "0.765", "FAIL"),
                      ("Input-Grad.", "0.473", "0.420", "FAIL")],
    "Swin-T_IDRiD": [("Grad-CAM", "0.528", "0.195", "pass"), ("Integrated Grad.", "0.491", "0.647", "FAIL"),
                     ("Transf. Attr.", "0.427", "0.163", "pass"), ("TokenNorm", "0.459", "0.166", "pass"),
                     ("Input-Grad.", "0.356", "0.298", "pass")],
    "ViT-B16_APTOS": [("Integrated Grad.", "0.642", "0.710", "FAIL"), ("Transf. Attr.", "0.579", "0.122", "pass"),
                      ("Grad-CAM", "0.576", "0.179", "pass"), ("Attn. Rollout", "0.536", "0.795", "FAIL"),
                      ("Input-Grad.", "0.500", "0.396", "FAIL")],
    "Swin-T_APTOS": [("Integrated Grad.", "0.613", "0.686", "FAIL"), ("Grad-CAM", "0.610", "0.184", "pass"),
                     ("TokenNorm", "0.555", "0.207", "pass"), ("Transf. Attr.", "0.468", "0.168", "pass"),
                     ("Input-Grad.", "0.388", "0.318", "FAIL")],
}
# ---- Table 5, deletion AUC column ----
TABLE_5_DEL = {'ViT-B16_IDRiD': {'Grad-CAM': '0.259', 'Integrated Grad.': '0.316', 'Transf. Attr.': '0.318', 'Attn. Rollout': '0.364', 'Input-Grad.': '0.406'}, 'Swin-T_IDRiD': {'Grad-CAM': '0.310', 'Integrated Grad.': '0.312', 'Transf. Attr.': '0.378', 'TokenNorm': '0.393', 'Input-Grad.': '0.420'}, 'ViT-B16_APTOS': {'Integrated Grad.': '0.278', 'Transf. Attr.': '0.317', 'Grad-CAM': '0.356', 'Attn. Rollout': '0.454', 'Input-Grad.': '0.365'}, 'Swin-T_APTOS': {'Integrated Grad.': '0.320', 'Grad-CAM': '0.419', 'TokenNorm': '0.470', 'Transf. Attr.': '0.487', 'Input-Grad.': '0.461'}}
# ---- Table 6: plausibility, union AUPRC ratio and MA / EX ratios (IDRiD only) ----
TABLE_VI = {
    "Swin-T_IDRiD": [("Integrated Grad.", "3.631", "2.06", "6.68"), ("TokenNorm", "1.670", "1.49", "2.38"),
                     ("Grad-CAM", "1.654", "1.69", "1.89"), ("Input-Grad.", "1.199", "1.86", "1.53"),
                     ("Transf. Attr.", "1.128", "1.35", "1.31")],
    "ViT-B16_IDRiD": [("Integrated Grad.", "9.428", "3.52", "20.34"), ("Input-Grad.", "3.135", "3.90", "4.79"),
                      ("Transf. Attr.", "2.564", "1.59", "8.04"), ("Attn. Rollout", "1.582", "1.07", "1.96"),
                      ("Grad-CAM", "1.232", "1.25", "1.29")],
}
# ---- Table 6, "Held-out" column: union AUPRC ratio on the 25 segmentation photographs that are not
#      training-split images (results/plausibility_heldout.csv, from scripts/seg_identity_and_heldout_colab.py) ----
TABLE_VI_HELD = {
    "Swin-T_IDRiD": {"Integrated Grad.": "3.069", "TokenNorm": "1.508", "Grad-CAM": "1.467",
                     "Input-Grad.": "1.117", "Transf. Attr.": "1.191"},
    "ViT-B16_IDRiD": {"Integrated Grad.": "8.321", "Input-Grad.": "3.119", "Transf. Attr.": "2.398",
                      "Attn. Rollout": "1.602", "Grad-CAM": "1.479"},
}
# ---- Table 3: per-grade image counts per split (train counts include augmentation) ----
TABLE_III = {
    ("01_SwinT_IDRiD.ipynb", "Restored training data:"): {
        "train": [120, 72, 122, 73, 73], "val": [14, 2, 14, 7, 5], "test": [34, 5, 32, 19, 13]},
    ("03_SwinT_APTOS.ipynb", "Restored APTOS data:"): {
        "train": [1434, 600, 808, 600, 600], "val": [172, 40, 104, 22, 28], "test": [199, 30, 87, 17, 33]},
}


def decimals(s: str) -> int:
    return len(s.split(".")[1]) if "." in s else 0


def close(paper: str, nb: str) -> bool:
    tol = 0.5 * 10 ** -decimals(paper) + 0.5 * 10 ** -decimals(nb) + 1e-9
    return abs(float(paper) - float(nb)) <= tol


def load(name):
    with (RES / name).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    bad, n = [], 0

    def check(label, paper, nb):
        nonlocal n
        n += 1
        if not close(paper, nb):
            bad.append(f"{label}: paper {paper} vs notebook {nb}")

    grading = {r["run"]: r for r in load("grading.csv")}
    for run, (vk, ep, acc, tk) in TABLE_IV.items():
        g = grading[run]
        check(f"Table 4 {run} val kappa", vk, g["best_val_qwk"])
        check(f"Table 4 {run} epoch", ep, g["best_epoch"])
        check(f"Table 4 {run} accuracy", acc, g["test_acc"])
        check(f"Table 4 {run} test kappa", tk, g["test_qwk"])

    sc = {(r["run"], r["explainer"]): r for r in load("scorecards.csv")}
    for run, rows in TABLE_V.items():
        for name, ins, rho, verdict in rows:
            r = sc[(run, NAME[name])]
            check(f"Table 5 {run} {name} insertion", ins, r["insertion_auc"])
            check(f"Table 5 {run} {name} mean|rho|", rho, r["mean_abs_rho"])
            check(f"Table 5 {run} {name} deletion", TABLE_5_DEL[run][name], r["deletion_auc"])
            n += 1
            if (verdict == "pass") != (r["sanity_pass"] == "YES"):
                bad.append(f"Table 5 {run} {name} sanity: paper {verdict} vs notebook {r['sanity_pass']}")

    les = {(r["run"], r["explainer"]): r for r in load("plausibility_per_lesion.csv")}
    for run, rows in TABLE_VI.items():
        for name, union, ma, ex in rows:
            check(f"Table 6 {run} {name} union", union, sc[(run, NAME[name])]["plausibility_auprc_ratio_union"])
            check(f"Table 6 {run} {name} MA", ma, les[(run, NAME[name])]["MA"])
            check(f"Table 6 {run} {name} EX", ex, les[(run, NAME[name])]["EX"])

    held = {(r["run"], r["explainer"]): r for r in load("plausibility_heldout.csv")}
    for run, vals in TABLE_VI_HELD.items():
        for name, v in vals.items():
            r = held[(run, NAME[name])]
            check(f"Table 6 held-out {run} {name}", v, r["union_auprc_ratio_mean"])
            n += 1
            if r["n_images"] != "25":
                bad.append(f"Table 6 held-out {run} {name}: n_images {r['n_images']} != 25")

    # Limitations (v): APTOS test kappa without the test images that duplicate training images
    ded = {(r["run"], r["test set"]): r for r in load("aptos_dedup_test_scores.csv")}
    for run, full, dedup in (("Swin-T_APTOS", "0.8858", "0.8833"), ("ViT-B16_APTOS", "0.9070", "0.9076")):
        check(f"re-evaluation reproduces Table 4 {run}", full, ded[(run, "all test images")]["QWK"])
        check(f"Limitations (v) {run} kappa without duplicates", dedup,
              ded[(run, "without duplicates of TRAIN images")]["QWK"])

    for (fname, marker), exp in TABLE_III.items():
        nb = json.loads((ROOT / "notebooks" / fname).read_text(encoding="utf-8"))
        text = ""
        for cell in nb["cells"]:
            for o in cell.get("outputs", []):
                t = "".join(o.get("text", ""))
                if marker in t:
                    text = t
        for split, counts in exp.items():
            m = re.search(rf"^{split}\s+((?:\d+\s+){{5}})\d+\s*$", text, re.M)
            got = [int(x) for x in m.group(1).split()] if m else None
            n += 1
            if got != counts:
                bad.append(f"Table 3 {fname} {split}: paper {counts} vs notebook {got}")

    print(f"checked {n} reported values (Tables 3-6 and Limitations)")
    if bad:
        print("MISMATCHES:")
        print("\n".join("  " + b for b in bad))
        return 1
    print("all values match the executed notebooks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
