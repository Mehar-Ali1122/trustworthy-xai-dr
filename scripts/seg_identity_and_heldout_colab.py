# ======================================================================
# Colab cell. IDRiD segmentation-vs-grading identity check on the vessel/lesion pattern
# (green channel, high-pass, 512 x 340), then Table VI "Held-out" on the segmentation photographs
# that are not copies of training-split images, from the per-image plausibility_raw.csv saved in
# results_<RUN_TAG>.zip by notebooks 01 and 02.
# Run on 26 Nov 2026: best-match r >= 0.989 for 60 images, <= 0.783 for the other 21 (clean gap);
# 56 of the 60 are training-split images -> 25 held-out photographs. Printed tables are
# transcribed in results/plausibility_heldout.csv; per-image matches in results/seg_identity_fine.csv.
# Expects the official IDRiD zips unzipped to /content/idcheck/seg and /content/idcheck/grade.
# ======================================================================
import zipfile, io, numpy as np, pandas as pd, cv2
from pathlib import Path
D = "/content/drive/MyDrive/22_07_2026_IDRiD_Dataset_IEEE"
SEG, GRADE = "/content/idcheck/seg", "/content/idcheck/grade"
TOPK = 15                                   # thumbnail candidates checked per segmentation image
ok = lambda p: "__MACOSX" not in p.parts and not p.name.startswith("._")
seg = sorted(p for p in Path(SEG).rglob("*.jpg") if ok(p) and "groundtruth" not in str(p).lower())
grade = sorted(p for p in Path(GRADE).rglob("*.jpg") if ok(p))
print("segmentation:", len(seg), " grading:", len(grade))

def thumb(p):
    v = cv2.resize(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), (64, 64), interpolation=cv2.INTER_AREA).astype(np.float64).ravel()
    v -= v.mean(); return v / (np.linalg.norm(v) + 1e-12)
_cache = {}
def fine(p):                                # green channel, high-pass: keeps vessels and lesions
    if p not in _cache:
        g = cv2.imread(str(p))[:, :, 1].astype(np.float32)
        g = cv2.resize(g, (512, 340), interpolation=cv2.INTER_AREA)
        h = (g - cv2.GaussianBlur(g, (0, 0), 6)).ravel().astype(np.float64)
        h -= h.mean(); _cache[p] = h / (np.linalg.norm(h) + 1e-12)
    return _cache[p]

S = np.stack([thumb(p) for p in seg]); G = np.stack([thumb(p) for p in grade])
C = S @ G.T
sp = pd.read_csv(f"{D}/_meta/idrid_grading_splits.csv")
key = lambda p: ("official_test" if "test" in p.parent.name.lower() else "official_train", p.name)
split_of = {(o, i): s for o, i, s in zip(sp.orig_source, sp.image, sp.split)}
grade_of = {(o, i): g for o, i, g in zip(sp.orig_source, sp.image, sp.grade)}

rows = []
for a, p in enumerate(seg):
    cand = np.argsort(-C[a])[:TOPK]
    fs = [(float(fine(p) @ fine(grade[j])), j) for j in cand]
    best, j = max(fs)
    rows.append({"seg_file": p.name, "best_grading_file": grade[j].name, "folder": grade[j].parent.name,
                 "fine_corr": round(best, 4), "thumb_corr": round(float(C[a, j]), 4),
                 "split": split_of.get(key(grade[j])), "grade": grade_of.get(key(grade[j]))})
r = pd.DataFrame(rows).sort_values("fine_corr", ascending=False)
print("\nfine (vessel-pattern) correlation of the best match, sorted:")
print(np.round(r.fine_corr.values, 3))
THR = 0.95                                  # same photograph; the observed gap is 0.783 -> 0.989
r["same_photo"] = r.fine_corr > THR
r.to_csv(f"{D}/identity_check/seg_identity_fine.csv", index=False)
r.to_csv("/content/seg_identity_fine.csv", index=False)
print(f"\nsame photograph (fine r > {THR}): {r.same_photo.sum()} of {len(r)}")
print(pd.crosstab(r.split, r.same_photo).to_string())
print("\nlowest same:", r[r.same_photo].fine_corr.min(), " highest different:", r[~r.same_photo].fine_corr.max())
print("best grading match shared by several segmentation images (sign of NON-identity):")
print(r[~r.same_photo].best_grading_file.value_counts().loc[lambda v: v > 1].to_string())

held = set(r[~(r.same_photo & (r.split == "train"))].seg_file.str.rsplit(".", n=1).str[0])
print(f"\nHELD-OUT photographs (not a training-split photograph): {len(held)}")
for tag in ["Swin-T_IDRiD", "ViT-B16_IDRiD"]:
    with zipfile.ZipFile(f"{D}/results_{tag}.zip") as z:
        raw = pd.read_csv(io.BytesIO(z.read("plausibility_raw.csv")))
    x = raw[raw.base_id.isin(held)]
    u = x[x.mask_type == "union"].groupby("explainer").auprc_ratio.agg(["mean", "std", "count"]).round(3)
    les = x[x.mask_type.isin(["MA", "EX"])].pivot_table(index="explainer", columns="mask_type",
                                                        values="auprc_ratio", aggfunc="mean").round(2)
    print(f"\n===== {tag} | held-out =====")
    print(u.join(les).sort_values("mean", ascending=False).to_string())
