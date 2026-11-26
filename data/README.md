# Data

No images are stored in this repository. Both datasets are public; download them from the
official sources below and accept their terms of use.

## IDRiD (Indian Diabetic Retinopathy Image Dataset)

- Source: IEEE DataPort, <https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid>
  (DOI [10.21227/H25W98](https://doi.org/10.21227/H25W98)); challenge site <https://idrid.grand-challenge.org/>.
- Licence stated by the dataset authors: CC BY 4.0. Cite P. Porwal *et al.*, *Data*, 3(3):25, 2018, and
  the challenge paper (reference [36] in the manuscript).
- Parts used:
  - **B. Disease Grading**: 516 images (413 official train, 103 official test). Used for grading and for
    the faithfulness and sanity axes.
  - **A. Segmentation**: 81 images with pixel masks for MA, HE, EX and SE. Used only for the
    plausibility axis, at inference time. 60 of the 81 are the same photographs as grading images under
    different file numbers (56 of them in our training split); see `../preprocessing/README.md`.

## APTOS 2019 Blindness Detection

- Source: Kaggle competition, <https://www.kaggle.com/c/aptos2019-blindness-detection>
  (reference [38] in the manuscript). You must accept the competition rules to download.
- Part used: the 3,662 labelled training images. Kaggle withheld the test labels; the study uses
  the published 80/10/10 partition (2,930 / 366 / 366) of the Kaggle dataset "APTOS-2019 dataset",
  <https://www.kaggle.com/datasets/mariaherrerot/aptos2019> (`train_1.csv`, `valid.csv`, `test.csv`;
  `train_1.csv` was saved as `train.csv` in the source archive `APTOS-2019_Dataset.zip`). The three
  parts are consecutive blocks of the sorted image IDs (validation `000c…`–`1ae3…`, training
  `1ae8…`–`e4d3…`, test `e4dc…`–`ffec…`), so the split is not stratified by grade. APTOS has no pixel
  masks, so plausibility is not computed for it.

## Duplicate photographs across splits

Both datasets contain the same photograph under different file names. Checked on 26 Nov 2026 by
image content (`../scripts/duplicate_check_colab.py`; pairs in `../results/duplicates_across_splits.csv`):

- IDRiD: 1 of the 103 official test images (`IDRiD_064`) is byte-identical to official training
  image `IDRiD_118`, which is in our training split; the next most similar pair scores 0.86.
- APTOS: 24 of the 366 test images duplicate training images, 24 validation images duplicate
  training images, and 2 test images duplicate validation images. Within the duplicated pairs the
  grades often differ (7 of the 24 test–train pairs).
- Re-scoring the saved APTOS checkpoints without the duplicated test images
  (`../scripts/aptos_dedup_reeval_colab.py`, `../results/aptos_dedup_test_scores.csv`) changes test
  QWK from 0.8858 to 0.8833 (Swin-Tiny) and from 0.9070 to 0.9076 (ViT-B/16); the full-set rows
  reproduce Table IV exactly.

## Folder layout the notebooks expect (Google Colab)

The notebooks restore preprocessed archives from Google Drive and verify the counts before training.

```
/content/IDRiD_final/                      # from IDRiD_final_dataset.zip
    train/class0 … class4/*.jpg            # 460 images (371 unique + augmented copies)
    val/class0 … class4/*.jpg              # 42
    test/class0 … class4/*.jpg             # 103 (official IDRiD test set)
/content/IDRiD_plausibility/               # from IDRiD_plausibility_set.zip
    images/*.jpg                           # 81
    masks_union/*.png                      # 81 (union of MA, HE, EX, SE; optic disc excluded)
    masks_MA/ masks_HE/ masks_EX/ masks_SE/
/content/APTOS2019_classfolders_Cropped_224_Aug_CLAHE/   # from the APTOS archive
    train/Class0 … Class4/                 # 4,042 images (2,930 unique + augmented copies)
    validation/Class0 … Class4/            # 366
    test/Class0 … Class4/                  # 366
```

Drive folders used in the original runs (edit `DRIVE_ROOT` and `CFG` in each notebook to your own):

- IDRiD: `MyDrive/22_07_2026_IDRiD_Dataset_IEEE/` with `IDRiD_final_dataset.zip`,
  `IDRiD_plausibility_set.zip`, `plausibility_eval.py` and a `_meta/` folder.
- APTOS: `MyDrive/11_07_2026_APTOS_2019_Dataset/` with `APTOS2019_classfolders_Cropped_224_Aug_CLAHE.zip`.

The archives are produced by the offline preprocessing described in `../preprocessing/README.md`.
