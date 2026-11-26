# -*- coding: utf-8 -*-
# plausibility_eval.py — loaded by the two IDRiD notebooks (section 10).
# Code identical to cell 20 of preprocessing/00_IDRiD_preprocessing.ipynb.

"""
PLAUSIBILITY EVALUATION — the third trustworthiness axis
=========================================================
Drop-in module for the two IDRiD benchmark notebooks.

Consumes the step-5 plausibility set:
    IDRiD_plausibility/images/*.jpg          81 preprocessed fundus (224x224)
    IDRiD_plausibility/masks_union/*.png     union lesion masks
    IDRiD_plausibility/masks_{MA,HE,EX,SE}/  per-lesion-type masks

Interface matches the existing notebook convention exactly:
    explainer_fn(model, img_norm, target) -> HxW float array in [0, 1]

WHAT PLAUSIBILITY MEASURES
--------------------------
Faithfulness asks "does the attribution reflect what the model used?".
Sanity asks "does the attribution depend on the model at all?".
Plausibility asks a different question: "does the attribution land on ACTUAL
PATHOLOGY?" - agreement with expert pixel-level lesion annotation.

An explainer can be faithful yet implausible (the model genuinely used a
spurious cue), so the three axes are complementary, not redundant. Report
plausibility as evidence of clinical alignment, NOT as a correctness proof.

METRIC CHOICE — WHY FOUR, NOT ONE
---------------------------------
Lesion burden varies ~250x across the 81 images (union_frac 0.0009 - 0.216),
and lesions occupy a TINY share of the frame. Any single metric is misleading
here, so we report a panel:

  1. IoU@k        Intersection-over-Union of the top-k% attributed pixels with
                  the lesion mask. Interpretable but penalised by the huge
                  area mismatch when lesions are sparse.
  2. Precision@k  Of the top-k% attributed pixels, what fraction is lesion?
                  The clinically natural question: "is the model looking at
                  pathology?" Robust to lesion sparsity.
  3. Recall@k     Of the lesion pixels, what fraction is in the top-k%?
                  Complements precision; low when lesions are widespread.
  4. AUPRC        Threshold-free ranking quality: treats the attribution as a
                  per-pixel score for "is this pixel lesion?". Independent of
                  any k, and the fairest cross-explainer comparison.

A RANDOM BASELINE is computed for every image. Because lesion prevalence is
so low, raw precision looks tiny even for a good explainer; the ratio
(observed / random) is the honest effect size. ALWAYS report the ratio
alongside the raw value, or the numbers look uninterpretably small.
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import os
import numpy as np
import pandas as pd


# ======================================================================
# Core metric primitives
# ======================================================================
def topk_mask(attr: np.ndarray, k_frac: float) -> np.ndarray:
    """Boolean mask of the top ``k_frac`` fraction of attributed pixels."""
    if not (0 < k_frac <= 1):
        raise ValueError(f"k_frac must be in (0, 1], got {k_frac}")
    flat = attr.ravel()
    n_take = max(1, int(round(k_frac * flat.size)))
    # argpartition is O(n) vs O(n log n) for a full sort
    idx = np.argpartition(flat, -n_take)[-n_take:]
    out = np.zeros(flat.size, dtype=bool)
    out[idx] = True
    return out.reshape(attr.shape)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / union) if union else np.nan


def precision_recall(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    inter = float(np.logical_and(pred, gt).sum())
    p = inter / pred.sum() if pred.sum() else np.nan
    r = inter / gt.sum() if gt.sum() else np.nan
    return float(p), float(r)


def auprc(attr: np.ndarray, gt: np.ndarray) -> float:
    """Threshold-free AUPRC treating attribution as a per-pixel lesion score.

    Implemented directly (no sklearn dependency) via a sorted sweep.
    """
    y = gt.ravel().astype(bool)
    s = attr.ravel().astype(np.float64)
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == y.size:
        return np.nan
    order = np.argsort(-s)                      # descending score
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    # step-wise integration over recall (average precision)
    dr = np.diff(np.concatenate([[0.0], recall]))
    return float(np.sum(precision * dr))


def random_baseline_precision(gt: np.ndarray) -> float:
    """Expected precision of a random top-k selection = lesion prevalence."""
    return float(gt.mean())


# ======================================================================
# Mask loading
# ======================================================================
def load_mask(path: str) -> np.ndarray:
    """Load a binary mask as a boolean HxW array."""
    import cv2
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return m > 127


def build_plausibility_index(img_dir: str, union_dir: str,
                             per_type_dirs: Optional[Dict[str, str]] = None
                             ) -> pd.DataFrame:
    """Pair each plausibility image with its union mask (and per-type masks)."""
    from pathlib import Path
    rows = []
    for p in sorted(Path(img_dir).glob("*.jpg")):
        bid = p.stem
        um = Path(union_dir) / f"{bid}.png"
        if not um.is_file():
            continue
        rec = {"base_id": bid, "image_path": str(p), "union_mask": str(um)}
        if per_type_dirs:
            for lt, d in per_type_dirs.items():
                q = Path(d) / f"{bid}.png"
                rec[f"mask_{lt}"] = str(q) if q.is_file() else None
        rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(
            f"No image/mask pairs found.\n  images: {img_dir}\n  masks: {union_dir}")
    return df


# ======================================================================
# Per-image scoring
# ======================================================================
def score_attribution(attr: np.ndarray, gt: np.ndarray,
                      k_fracs: Sequence[float] = (0.01, 0.05, 0.10)) -> Dict:
    """All plausibility metrics for one attribution map against one mask."""
    if attr.shape != gt.shape:
        raise ValueError(f"shape mismatch: attr {attr.shape} vs mask {gt.shape}")

    # normalise defensively; explainers should already return [0,1]
    a = np.asarray(attr, dtype=np.float64)
    if not np.isfinite(a).all():
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    rng = a.max() - a.min()
    a = (a - a.min()) / rng if rng > 0 else np.zeros_like(a)

    prevalence = random_baseline_precision(gt)
    out: Dict[str, float] = {
        "lesion_frac": prevalence,
        "auprc": auprc(a, gt),
        # AUPRC of a random ranker equals prevalence
        "auprc_ratio": (auprc(a, gt) / prevalence) if prevalence > 0 else np.nan,
    }
    for k in k_fracs:
        pred = topk_mask(a, k)
        p, r = precision_recall(pred, gt)
        tag = f"{int(k*100)}"
        out[f"iou@{tag}"] = iou(pred, gt)
        out[f"prec@{tag}"] = p
        out[f"rec@{tag}"] = r
        # effect size vs a random selector of the same size
        out[f"prec_ratio@{tag}"] = (p / prevalence) if prevalence > 0 else np.nan
    return out


# ======================================================================
# Full evaluation loop
# ======================================================================
def run_plausibility(model,
                     explainers: Dict[str, Callable],
                     index: pd.DataFrame,
                     preprocess_fn: Callable,
                     device,
                     k_fracs: Sequence[float] = (0.01, 0.05, 0.10),
                     per_lesion: bool = True,
                     lesion_types: Sequence[str] = ("MA", "HE", "EX", "SE"),
                     min_lesion_frac: float = 0.0,
                     target_mode: str = "predicted",
                     log_every: int = 20,
                     logger=None) -> pd.DataFrame:
    """Score every explainer on every plausibility image.

    Parameters
    ----------
    model : the TRAINED grader. Inference only - never train on these images.
    explainers : name -> fn(model, img_norm, target) -> HxW map in [0,1]
    index : output of build_plausibility_index()
    preprocess_fn : path -> normalised 1x3xHxW tensor (must match training)
    min_lesion_frac : skip images whose union mask covers less than this
        fraction of the frame. Metrics on near-empty masks are extremely
        noisy; the default 0.0 keeps everything and reports lesion_frac so
        you can filter afterwards.
    target_mode : 'predicted' (explain what the model decided) or 'true'.

    Returns one row per (image, explainer, mask_type).
    """
    import torch

    records: List[Dict] = []
    n = len(index)
    for i, (_, r) in enumerate(index.iterrows(), 1):
        x = preprocess_fn(r["image_path"]).to(device)
        with torch.no_grad():
            logits = model(x)
            pred = int(logits.argmax(1).item())
        target = pred if target_mode == "predicted" else int(r.get("grade", pred))

        union = load_mask(r["union_mask"])
        if union.mean() < min_lesion_frac:
            continue

        masks: Dict[str, np.ndarray] = {"union": union}
        if per_lesion:
            for lt in lesion_types:
                col = f"mask_{lt}"
                if col in r and isinstance(r[col], str) and os.path.isfile(r[col]):
                    m = load_mask(r[col])
                    if m.any():
                        masks[lt] = m

        for ename, efn in explainers.items():
            attr = efn(model, x.cpu(), target)
            for mtype, gt in masks.items():
                sc = score_attribution(attr, gt, k_fracs=k_fracs)
                records.append({
                    "base_id": r["base_id"], "explainer": ename,
                    "mask_type": mtype, "pred": pred, "target": target, **sc,
                })

        if logger and (i % log_every == 0 or i == n):
            logger.info("  plausibility: %d/%d images", i, n)

    return pd.DataFrame(records)


# ======================================================================
# Aggregation / reporting
# ======================================================================
def summarise(df: pd.DataFrame, mask_type: str = "union",
              k_tag: str = "5") -> pd.DataFrame:
    """Mean +/- std per explainer for one mask type."""
    sub = df[df.mask_type == mask_type]
    if sub.empty:
        return pd.DataFrame()
    cols = ["auprc", "auprc_ratio", f"iou@{k_tag}",
            f"prec@{k_tag}", f"rec@{k_tag}", f"prec_ratio@{k_tag}"]
    cols = [c for c in cols if c in sub.columns]
    grouped = sub.groupby("explainer")
    means = grouped[cols].mean().round(4)
    stds = grouped[cols].std().round(4)
    stds.columns = [f"{c}_std" for c in cols]
    out = means.join(stds)
    out["n_images"] = grouped.size()
    sort_col = "auprc_ratio" if "auprc_ratio" in out.columns else out.columns[0]
    return out.sort_values(sort_col, ascending=False)


def summarise_per_lesion(df: pd.DataFrame, metric: str = "auprc_ratio",
                         lesion_types: Sequence[str] = ("MA", "HE", "EX", "SE")
                         ) -> pd.DataFrame:
    """Explainer x lesion-type table — where the interesting asymmetry lives."""
    sub = df[df.mask_type.isin(lesion_types)]
    if sub.empty or metric not in sub.columns:
        return pd.DataFrame()
    piv = sub.pivot_table(index="explainer", columns="mask_type",
                          values=metric, aggfunc="mean").round(3)
    piv = piv.reindex(columns=[c for c in lesion_types if c in piv.columns])
    piv["mean"] = piv.mean(axis=1).round(3)
    return piv.sort_values("mean", ascending=False)


def interpretation_notes(df: pd.DataFrame) -> str:
    """Guard-rail text to print beneath the tables."""
    lines = [
        "INTERPRETING THESE NUMBERS",
        "-" * 62,
        "* Raw precision looks small because lesions occupy a tiny share of the",
        "  frame. Use the *_ratio columns (observed / random baseline) as the",
        "  effect size. ratio = 1.0 means no better than chance.",
        "* AUPRC is threshold-free and the fairest cross-explainer comparison;",
        "  IoU/precision depend on the chosen top-k.",
        "* Plausibility is NOT correctness. A faithful explainer can be",
        "  implausible if the model genuinely used a non-lesion cue - that is a",
        "  finding about the MODEL, not a failure of the explainer.",
        "* Per-lesion results matter: microaneurysms occupy ~10x less area than",
        "  haemorrhages/hard exudates, so union-only scores are dominated by the",
        "  large bright lesions. Low MA scores with high EX scores means",
        "  explainers track late-stage pathology and miss early markers.",
        "* Evaluated on the IDRiD segmentation set (n=81), which is DISJOINT",
        "  from the grading data (verified: different photographs). APTOS-2019",
        "  has no pixel masks, so plausibility is IDRiD-only.",
    ]
    if not df.empty and "lesion_frac" in df.columns:
        lf = df.drop_duplicates("base_id")["lesion_frac"]
        lines.append(f"* Lesion burden across images: min={lf.min():.5f} "
                     f"median={lf.median():.5f} max={lf.max():.5f} "
                     f"({lf.max()/max(lf.min(),1e-9):.0f}x spread).")
    return "\n".join(lines)