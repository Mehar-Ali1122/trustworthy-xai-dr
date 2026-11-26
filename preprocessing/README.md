# Offline preprocessing

The four benchmark notebooks start from preprocessed archives (`DATA_PREPROCESSED = True`).

| Dataset | Preprocessing code | Status |
|---|---|---|
| IDRiD (grading set and plausibility set) | `00_IDRiD_preprocessing.ipynb` (executed, outputs kept) | included |
| IDRiD plausibility metrics | `../src/plausibility_eval.py` (same code as cell 20 of the notebook above) | included |
| APTOS-2019 | `00_APTOS_preprocessing.ipynb` (executed, outputs kept) | included |

## IDRiD: `00_IDRiD_preprocessing.ipynb`

Cell numbers count every cell from the top, starting at 1.

| Cell | Step | Output |
|---|---|---|
| 3 | Step 0/1/3: read the official labels, stratified 90/10 train/validation carve of the official training set (`random_state = 42`), leakage audit | `_meta/idrid_grading_splits.csv`; 371 / 42 / 103 images |
| 6 | Step 1b: label analysis and class-folder layout | `IDRiD_organized/` |
| 8 | Step 2: three-tier retinal ROI detection (Hough circle → largest bright contour → fallback), crop, resize to 224 × 224 | `IDRiD_224/`, `crop_boxes.csv` |
| 10 | Step 3: CLAHE on the LAB L-channel, `clipLimit = 2.0`, `tileGridSize = (8, 8)`, on all splits | `IDRiD_224_clahe/`, `clahe_log.csv` |
| 12 | Step 4: augmentation of the training split only (seed 42) | `IDRiD_final/`, `augmentation_log.csv`, `class_weights.json` |
| 14 | Step 4b: archive | `IDRiD_final_dataset.zip` |
| 16 | Diagnostic: are the Segmentation and Grading images the same photographs? | printed verdict |
| 17 | Step 5: plausibility set; each of the 81 segmentation images gets the same three-tier ROI detection, one crop box shared by the image and all its masks, resize to 224 × 224, and CLAHE with the step-3 settings on the image | `IDRiD_plausibility/`, `plausibility_manifest.csv` |
| 19 | Step 5b: archive | `IDRiD_plausibility_set.zip` |
| 20 | Plausibility module (identical to `src/plausibility_eval.py`) | — |

Augmentation settings in cell 12 (`CFG`): class floor 60 % of the majority class, at most 4× a
class's originals; horizontal flip with p = 0.5, vertical flip with p = 0.2, rotation within ±12°,
zoom 0.92–1.08, translation up to 4 % (`TRANS_FRAC = 0.04`), `BORDER_REFLECT_101`.
This produced 89 augmented images: 371 → 460 training images (per grade 120 / 72 / 122 / 73 / 73).

The `class_weights.json` written by step 4 (plain inverse frequency) is not what the benchmark
notebooks use: they recompute tempered weights from the training counts (benchmark notebooks,
cell 13).

### Segmentation vs grading photographs (cell 16 and the full check)

Cell 16 compared only the first 12 segmentation images with their **same-numbered** grading
images (24 pairs) and found no identical files. The two IDRiD parts in fact share photographs under
different file numbers. Full check on 26 Nov 2026 (all 81 segmentation vs all 516 grading images):

1. **Screening** (`../scripts/verify_seg_grading_disjoint.py`, MD5 + 64 × 64 thumbnail correlation;
   output `../results/seg_grading_disjointness.csv`): 3 byte-identical pairs; 60 images with
   thumbnail r ≥ 0.9998. Thumbnail correlation is not decisive below that level (fundus framing alone
   gives r ≈ 0.98, e.g. segmentation images matched to grade-0 grading images).
2. **Decision** (`../scripts/seg_identity_and_heldout_colab.py`, correlation of the high-pass green
   channel at 512 × 340 over the 15 best thumbnail candidates): the best-match correlation is
   ≥ 0.989 for 60 images and ≤ 0.783 for the other 21, with nothing in between. The lower values
   come from shared field-of-view geometry (one grade-0 test image is the "best match" of three
   different segmentation images).

Result: **60 of the 81 segmentation photographs are grading photographs** (3 byte-identical);
**56 are in our training split**, 3 in test, 1 in validation. The graders were therefore trained on
56 of the 81 plausibility photographs (after the same crop and CLAHE). The manuscript reports Table VI
on all 81 images and, in a "Held-out" column, on the 25 photographs that are not training-split
images (`../results/plausibility_heldout.csv`).

## APTOS-2019: `00_APTOS_preprocessing.ipynb`

| Cell | Step | Output |
|---|---|---|
| 1 | Download the source archive `APTOS-2019_Dataset.zip` from the author's Google Drive and sort the images into `train/validation/test/Class0…Class4` using the archive's `train.csv`, `valid.csv` and `test.csv` | `APTOS2019_classfolders/` (2,930 / 366 / 366) |
| 2 | Zip the class folders | `APTOS2019_classfolders.zip` |
| 3 | Three-tier retinal ROI crop (Hough circle → largest contour → original image, 2 % margin), resize to 224 × 224 | `APTOS2019_classfolders_Cropped_224/`; Hough 2,159, contour 1,503, original 0, failed 0 |
| 4 | Before/after crop figures | PNG figures |
| 5 | Augmentation of the training split, then CLAHE on all splits (seed 42) | `APTOS2019_classfolders_Cropped_224_Aug_CLAHE.zip` |

Settings in cell 5:

- Augmentation, training split only, grades 1, 3 and 4: raised to 600 images per grade, at most 4×
  the originals (grade 1: 300 → 600, grade 3: 154 → 600, grade 4: 234 → 600; 1,112 images generated,
  2,930 → 4,042). Each copy: horizontal flip p = 0.5, vertical flip p = 0.2, rotation within ±12°,
  translation up to ±6 %, zoom 1.00–1.12 (zoom in only), brightness ±10 % on the HSV V-channel,
  black constant border.
- CLAHE on the LAB L-channel, `clipLimit = 2.0`, `tileGridSize = (8, 8)`, applied **after**
  augmentation to every image (4,042 + 366 + 366 = 4,774; the cell asserts this count).
  For IDRiD, CLAHE was applied before augmentation; both datasets therefore receive the same
  CLAHE on every image, in a different order relative to augmentation.

**Split.** The train / validation / test partition (2,930 / 366 / 366, i.e. 80 / 10 / 10 % of the
3,662 labelled images) is the one published as the Kaggle dataset "APTOS-2019 dataset"
(<https://www.kaggle.com/datasets/mariaherrerot/aptos2019>); its `train_1.csv`, `valid.csv` and
`test.csv` match the CSV files in the source archive row for row (checked 26 Nov 2026). The notebook
does not create the split. The parts are consecutive blocks of the sorted image IDs, so the class
proportions are similar but not stratified (grade 0: 48.9 % / 47.0 % / 54.4 %). The benchmark
notebooks' markdown and `run_summary_results_*_APTOS.json` describe this split as "stratified
70/15/15"; that description is incorrect, and the counts above are the ones actually used.

**Redaction.** In this copy of the notebook the Google Drive file ID of the source archive (cell 1,
source and download log) has been replaced by `<YOUR_DRIVE_FILE_ID>`; nothing else was changed.
The notebook's figures show a few APTOS fundus images.

## Normalisation

Normalisation statistics are computed on the fly from the training split only (benchmark
notebooks, section 3); they are not part of the offline step.
