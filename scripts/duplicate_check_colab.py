# ======================================================================
# Colab cell, run on 26 Nov 2026 (pairs found are listed in results/duplicates_across_splits.csv).
# Definitive duplicate check across splits (vessel/lesion pattern, like the segmentation check):
#   same photograph -> fine r near 1;  different photographs -> clearly lower
# ======================================================================
import os, numpy as np, pandas as pd, cv2
from pathlib import Path
IDRID = "/content/drive/MyDrive/22_07_2026_IDRiD_Dataset_IEEE"
ok = lambda p: "__MACOSX" not in p.parts and not p.name.startswith("._")
TOPK = 15

def thumb(p):
    v = cv2.resize(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), (64, 64), interpolation=cv2.INTER_AREA).astype(np.float64).ravel()
    v -= v.mean(); return v / (np.linalg.norm(v) + 1e-12)
def make_fine(size, sigma):
    cache = {}
    def fine(p):
        if p not in cache:
            g = cv2.resize(cv2.imread(str(p))[:, :, 1].astype(np.float32), size, interpolation=cv2.INTER_AREA)
            h = (g - cv2.GaussianBlur(g, (0, 0), sigma)).ravel().astype(np.float64)
            h -= h.mean(); cache[p] = h / (np.linalg.norm(h) + 1e-12)
        return cache[p]
    return fine
def fine_check(name, A, B, fine, label=lambda p: ""):
    C = np.stack([thumb(p) for p in A]) @ np.stack([thumb(p) for p in B]).T
    rows = []
    for a, p in enumerate(A):
        best, j = max((float(fine(p) @ fine(B[k])), k) for k in np.argsort(-C[a])[:TOPK])
        rows.append((p.name, B[j].name, round(best, 4), label(B[j])))
    r = pd.DataFrame(rows, columns=["image", "best_match", "fine_r", "match_info"]).sort_values("fine_r", ascending=False)
    print(f"\n{name}: {len(A)} vs {len(B)} images")
    print("  top 15 fine r:", r.fine_r.values[:15].round(3).tolist())
    print(f"  fine r >= 0.95 (same photograph): {(r.fine_r >= 0.95).sum()}")
    print(r[r.fine_r >= 0.95].to_string(index=False) if (r.fine_r >= 0.95).any() else "  none")
    return r

# ---- IDRiD: official test vs official train (full resolution) ----
G = Path("/content/idcheck/grade")
tr = sorted(p for p in G.rglob("*.jpg") if ok(p) and "Training" in str(p))
te = sorted(p for p in G.rglob("*.jpg") if ok(p) and "Testing" in str(p))
sp = pd.read_csv(f"{IDRID}/_meta/idrid_grading_splits.csv")
our_split = {i: s for o, i, s in zip(sp.orig_source, sp.image, sp.split) if o == "official_train"}
r1 = fine_check("IDRiD official test vs official train", te, tr, make_fine((512, 340), 6),
                label=lambda p: f"our split: {our_split.get(p.name)}")

# ---- APTOS: our splits (cropped 224 x 224 images) ----
A = Path("/content/aptos224")
imgs = [p for p in A.rglob("*") if p.suffix.lower() in (".png", ".jpg") and ok(p)]
part = lambda s: sorted(p for p in imgs if f"/{s}/" in str(p))
trA, vaA, teA = part("train"), part("validation"), part("test")
fA = make_fine((224, 224), 3)
r2 = fine_check("APTOS test vs train", teA, trA, fA, label=lambda p: p.parent.name)
r3 = fine_check("APTOS validation vs train", vaA, trA, fA, label=lambda p: p.parent.name)
r4 = fine_check("APTOS test vs validation", teA, vaA, fA, label=lambda p: p.parent.name)
os.makedirs(f"{IDRID}/identity_check", exist_ok=True)
for n, r in [("idrid_test_vs_train", r1), ("aptos_test_vs_train", r2), ("aptos_val_vs_train", r3), ("aptos_test_vs_val", r4)]:
    r.to_csv(f"{IDRID}/identity_check/dup_{n}.csv", index=False)
print("\nsaved CSVs to", f"{IDRID}/identity_check/")
