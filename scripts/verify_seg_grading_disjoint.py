#!/usr/bin/env python3
"""Full check that the IDRiD Segmentation images are different photographs from the
Disease-Grading images.

The diagnostic in preprocessing/00_IDRiD_preprocessing.ipynb (cell 16) compared only the
first 12 segmentation images with their same-numbered grading images (24 pairs). This
script compares ALL 81 segmentation images with ALL 516 grading images:

  1. exact byte identity (MD5) over every pair;
  2. Pearson correlation of 64x64 grayscale thumbnails (same method as cell 16)
     over every pair, reporting the closest grading image for each segmentation image.

Usage (after unzipping the official IDRiD "A. Segmentation" and "B. Disease Grading" folders):
    python scripts/verify_seg_grading_disjoint.py --seg <segmentation_dir> --grade <grading_dir>

Requires numpy, pandas and opencv-python.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def thumb(path: Path, size: int = 64) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise ValueError(f"cannot read image: {path}")
    return cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA).astype(np.float64).ravel()


def zscore_rows(m: np.ndarray) -> np.ndarray:
    m = m - m.mean(axis=1, keepdims=True)
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg", required=True, help="folder containing the segmentation fundus images")
    ap.add_argument("--grade", required=True, help="folder containing the disease-grading images")
    ap.add_argument("--out", default="results/seg_grading_disjointness.csv")
    args = ap.parse_args()

    def ok(p: Path) -> bool:  # skip macOS archive metadata such as __MACOSX/._IDRiD_01.jpg
        return "__MACOSX" not in p.parts and not p.name.startswith("._")

    seg = sorted(p for p in Path(args.seg).rglob("*.jpg") if ok(p) and "groundtruth" not in str(p).lower())
    grade = sorted(p for p in Path(args.grade).rglob("*.jpg") if ok(p))
    print(f"segmentation images: {len(seg)}   grading images: {len(grade)}")

    seg_md5 = {p: md5(p) for p in seg}
    grade_md5 = {}
    for p in grade:
        grade_md5.setdefault(md5(p), []).append(p)
    md5_hits = [(s, g) for s, h in seg_md5.items() for g in grade_md5.get(h, [])]

    S = zscore_rows(np.stack([thumb(p) for p in seg]))
    G = zscore_rows(np.stack([thumb(p) for p in grade]))
    C = S @ G.T                      # Pearson correlation, all 81 x 516 pairs

    best = C.argmax(axis=1)
    rows = [{"seg_file": seg[i].name, "closest_grading_file": grade[j].name,
             "closest_grading_folder": grade[j].parent.name, "max_thumb_corr": round(float(C[i, j]), 4)}
            for i, j in enumerate(best)]
    df = pd.DataFrame(rows).sort_values("max_thumb_corr", ascending=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"pairs compared: {C.size}")
    print(f"MD5 matches: {len(md5_hits)}")
    for s, g in md5_hits:
        print(f"  identical bytes: {s.name} == {g}")
    print("highest thumbnail correlation per segmentation image:")
    print(f"  min {df.max_thumb_corr.min():.4f}  median {df.max_thumb_corr.median():.4f}  "
          f"max {df.max_thumb_corr.max():.4f}")
    print(f"  segmentation images with a grading image above 0.98: {(df.max_thumb_corr > 0.98).sum()}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
