# ======================================================================
# Colab cell, run on 26 Nov 2026 (output transcribed in results/aptos_dedup_test_scores.csv).
# APTOS test score with and without the duplicated images (no retraining):
# reloads each saved best checkpoint, reproduces Table IV, then re-scores without
# the test images that are copies of training (or validation) images.
# ======================================================================
from google.colab import drive
import os
if not os.path.ismount("/content/drive"):
    drive.mount("/content/drive")

import io, json, zipfile, numpy as np, pandas as pd, torch, timm
from pathlib import Path
from torchvision import datasets, transforms
from sklearn.metrics import cohen_kappa_score
IDRID = "/content/drive/MyDrive/22_07_2026_IDRiD_Dataset_IEEE"
APTOS = "/content/drive/MyDrive/11_07_2026_APTOS_2019_Dataset"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# test images that are copies of training / validation images (from scripts/duplicate_check_colab.py;
# listed in results/duplicates_across_splits.csv)
dup_train = {"f6f3ea0d2693", "e740af6ac6ea", "fe0fc67c7980", "fcc6aa6755e6", "e76a9cbb2a8c", "f920ccd926db",
             "f9ecf1795804", "ff0740cb484a", "fea14b3d44b0", "e4e343eaae2a", "e7a7187066ad", "f7edc074f06b",
             "f4d3777f2710", "ee2c2a5f7d0e", "ea05c22d92e9", "ea9e0fb6fb0b", "f066db7a2efe", "f03d3c4ce7fb",
             "eadc57064154", "ed3a0fc5b546", "f9e1c439d4c8", "ea15a290eb96", "ff1e940105f9", "e79e10907295"}
dup_any = dup_train | {"fda39982a810", "f0f89314e860"}
stem = lambda s: str(s).rsplit(".", 1)[0]
print("test images duplicated in train:", len(dup_train), "| in train or validation:", len(dup_any))

# the exact preprocessed images the benchmark notebooks used
root = Path("/content/aptos_clahe")
if not root.exists():
    zipfile.ZipFile(f"{APTOS}/APTOS2019_classfolders_Cropped_224_Aug_CLAHE.zip").extractall(root)
test_dir = [p for p in root.rglob("test") if p.is_dir() and (p / "Class0").is_dir()][0]
print("test folder:", test_dir)

RUNS = {"Swin-T_APTOS": "swin_tiny_patch4_window7_224", "ViT-B16_APTOS": "vit_base_patch16_224"}
rows = []
for tag, arch in RUNS.items():
    with zipfile.ZipFile(f"{IDRID}/results_{tag}.zip") as z:
        if "best_model.pt" not in z.namelist():
            print(f"{tag}: best_model.pt is not in the results zip -> cannot re-evaluate"); continue
        stats = json.loads(z.read("norm_stats.json"))
        state = torch.load(io.BytesIO(z.read("best_model.pt")), map_location="cpu")
    model = timm.create_model(arch, pretrained=False, num_classes=5)
    model.load_state_dict(state); model.to(dev).eval()
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize(stats["mean"], stats["std"])])
    ds = datasets.ImageFolder(str(test_dir), transform=tf)
    ld = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)
    preds = []
    with torch.no_grad():
        for x, _ in ld:
            preds.extend(model(x.to(dev)).argmax(1).cpu().numpy())
    preds = np.array(preds); trues = np.array([y for _, y in ds.samples])
    names = np.array([stem(Path(p).name) for p, _ in ds.samples])
    for label, keep in [("all test images", np.ones(len(names), bool)),
                        ("without duplicates of TRAIN images", ~np.isin(names, list(dup_train))),
                        ("without duplicates of TRAIN or VAL images", ~np.isin(names, list(dup_any)))]:
        t, p = trues[keep], preds[keep]
        rows.append({"run": tag, "test set": label, "n": int(keep.sum()),
                     "accuracy": round(float((t == p).mean()), 4),
                     "QWK": round(float(cohen_kappa_score(t, p, weights="quadratic")), 4)})
    pd.DataFrame({"image": names, "true": trues, "pred": preds,
                  "dup_of_train": np.isin(names, list(dup_train)),
                  "dup_of_train_or_val": np.isin(names, list(dup_any))}
                 ).to_csv(f"{IDRID}/identity_check/aptos_test_predictions_{tag}.csv", index=False)
    del model; torch.cuda.empty_cache() if dev.type == "cuda" else None

res = pd.DataFrame(rows)
print("\n" + res.to_string(index=False))
print("\n(The 'all test images' rows should equal Table IV: Swin acc 0.8415 / QWK 0.8858, ViT acc 0.8415 / QWK 0.9070.)")
res.to_csv(f"{IDRID}/identity_check/aptos_dedup_test_scores.csv", index=False)
