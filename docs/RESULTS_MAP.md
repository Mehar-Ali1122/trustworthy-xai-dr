# Where each reported number comes from

Every table value in the manuscript is printed by a cell of one of the four executed notebooks.
Cell numbers count every cell (markdown and code) from the top of the notebook, starting at 1.
Each cell also begins with a capitalised comment you can search for.

`python scripts/extract_results.py` copies these printed values into `results/*.csv`, and
`python scripts/check_paper_numbers.py` compares 156 reported values (Tables 3–6, including the deletion AUC column, and the de-duplicated APTOS κ) with them.

## Notebooks

| File | Run tag | Backbone (`timm`) | Dataset | GPU recorded in notebook metadata |
|---|---|---|---|---|
| `notebooks/01_SwinT_IDRiD.ipynb` | Swin-T_IDRiD | `swin_tiny_patch4_window7_224` | IDRiD | T4 |
| `notebooks/02_ViT-B16_IDRiD.ipynb` | ViT-B16_IDRiD | `vit_base_patch16_224` | IDRiD | A100 |
| `notebooks/03_SwinT_APTOS.ipynb` | Swin-T_APTOS | `swin_tiny_patch4_window7_224` | APTOS-2019 | A100 |
| `notebooks/04_ViT-B16_APTOS.ipynb` | ViT-B16_APTOS | `vit_base_patch16_224` | APTOS-2019 | T4 |

## Manuscript item → notebook cell

| Manuscript item | Searchable cell title | 01 | 02 | 03 | 04 |
|---|---|---|---|---|---|
| Table 2 (unique / augmented / val / test counts) | `PROVENANCE BLOCK`, `RUN SUMMARY` | 11, 50 | 11, 50 | 11, 49 | 11, 48 |
| IDRiD split and augmentation counts at source (371 / 42 / 103; 371 → 460) | `preprocessing/00_IDRiD_preprocessing.ipynb`, cells 3 (Stage G) and 12 (Stages A–D) | — | — | — | — |
| Table 3 (per-grade counts per split) | `VERIFY the restored tree` | 10 | 10 | 10 | 10 |
| Table 4: validation κ and best epoch | training cell (`Best validation QWK = …`) | 18 | 18 | 18 | 18 |
| Table 4: test accuracy and test κ | `LOAD BEST CHECKPOINT AND EVALUATE` | 21 | 21 | 21 | 21 |
| Fig. 2 (per-grade recall) and the precision/recall values in Section 6.1 | same cell, per-class table | 21 | 21 | 21 | 21 |
| XAI subset size (53 per seed on IDRiD, 60 on APTOS; seeds 42, 1, 7) | `Balanced held-out XAI subset` | 34 | 34 | 35 | 34 |
| Faithfulness sweep (insertion/deletion AUC) | `FAITHFULNESS SWEEP` | 35 | 35 | 36 | 35 |
| Sanity (model randomization, 30 images, fractions 0.25/0.50/1.00); per-fraction values in Fig. 3 | `Cascading model-randomization` (printed table `Mean \|Spearman\|`) | 38 | 38 | 39 | 38 |
| Table 5 (insertion and deletion AUC, mean \|ρ\|, pass/fail) and the threshold-sensitivity statement in Section 7 | scorecard cell, output headed `TRUSTWORTHINESS SCORECARD` (columns `faith_insertion`, `sanity_mean_rho`, `sanity_at_full`, `sanity_pass`) | 47 | 47 | 46 | 45 |
| Table 6 union AUPRC ratio | `RUN PLAUSIBILITY` (union table) and the scorecard column `plaus_auprc_ratio` | 42, 47 | 42, 47 | — | — |
| Table 6 MA / EX ratios and Fig. 4 (MA, HE, EX, SE) | `RUN PLAUSIBILITY` (per-lesion table) | 42 | 42 | — | — |
| Table 6 "Held-out" column (25 segmentation photographs that are not training-split images) | `scripts/seg_identity_and_heldout_colab.py`, from `plausibility_raw.csv` saved by cell 42 of 01 and 02; values in `results/plausibility_heldout.csv` | — | — | — | — |
| Limitations (v): duplicated test images and test κ without them (0.8833, 0.9076) | `scripts/duplicate_check_colab.py`, `scripts/aptos_dedup_reeval_colab.py`; `results/duplicates_across_splits.csv`, `results/aptos_dedup_test_scores.csv` | — | — | — | — |
| Fig. 6 (qualitative panel, ViT-B/16 on IDRiD) | `def show_panel_for_grade` | — | 45 | — | — |
| Per-configuration trustworthiness scatter | `TRUSTWORTHINESS SCATTER` | 48 | 48 | 47 | 46 |

Figs. 2–5 of the manuscript combine the four configurations. Fig. 2 is drawn from `results/per_class_test.csv` and Figs. 3–4 from the sanity and per-lesion outputs listed above (`scripts/make_fig2_per_grade_recall.py`, `scripts/make_fig3_fig4_bspc.py`); Fig. 5 uses the scorecard values in `results/scorecards.csv`.

## Notes for readers of the code

- **Blurred baseline.** `CFG.BLUR_SIGMA = 8.0` is defined, but the faithfulness section
  (cell 32 in notebooks 01, 02, 04; cell 33 in 03) redefines `blurred_baseline()` as
  `torchvision` Gaussian blur with `kernel_size=51, sigma=15.0`. Python resolves the name when
  the function is called, so every explanation evaluated after that cell, including the
  Integrated Gradients baseline, uses kernel 51 and σ = 15.0, as stated in the manuscript.
- **Values set in the model cell.** `CFG` lists `HEAD_ONLY_EPOCHS = 5`, `DROP_RATE = 0.15` and
  `DROP_PATH_RATE = 0.1`, but the model cell (cell 15) builds the backbone with
  `drop_rate=0.1, drop_path_rate=0.1` and sets `HEAD_ONLY_EPOCHS = 3`. The training output
  confirms the 3-epoch warm-up ("unfroze full backbone after epoch 3 (lr 1e-03 -> 2e-05)", cell 18).
  The manuscript reports the values actually used (3 epochs, dropout 0.10, drop-path 0.10).
- **APTOS training diagnostic.** The overfitting diagnostic in notebooks 03 and 04 prints
  "val has only 366 images (Class 1: 2)". The "(Class 1: 2)" text is a fixed string carried over
  from the IDRiD notebook; the APTOS validation split has 40 grade-1 images (cell 10).
- **APTOS split description.** The APTOS notebooks and run summaries describe the split as
  "stratified 70/15/15". The split actually used is 2,930 / 366 / 366 (80/10/10), the published
  partition of the Kaggle "APTOS-2019 dataset" (`data/README.md`); the counts in Tables 2–3 are
  correct.
- **Table 6 rounding.** Three Table 6 values are rounded from the per-image results rather than
  from the 4-decimal printout: Swin Grad-CAM 1.654 (printed 1.6545), Swin Input-Gradient 1.199
  (printed 1.1995), ViT Attention Rollout EX 1.96 (printed 1.965).
- **Run summaries.** `results/run_summary_results_*.json` are the files written by the
  `RUN SUMMARY` cell of each notebook; their values match the printed outputs.
