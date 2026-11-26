# Trustworthy Explainable AI for Diabetic Retinopathy Grading
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24555312.svg)](https://doi.org/10.5281/zenodo.24555312)

Code and executed notebooks for the manuscript

> M. Ali, A. Ishaq, and S. Ullah, “Trustworthy explainable AI for diabetic retinopathy grading:
> A joint faithfulness, sanity, and plausibility benchmark across vision transformer
> > architectures,” manuscript, 2026. Code archived at https://doi.org/10.5281/zenodo.24555312.

Table and figure numbers below follow the manuscript (Tables 1–6, Figs. 1–6).

The study grades diabetic retinopathy (grades 0–4) with two ImageNet-pretrained vision
transformers (ViT-B/16 and Swin-Tiny) on two public datasets (IDRiD and APTOS-2019), then scores
five explanation methods per backbone on three axes:

| Axis | Test | Data |
|---|---|---|
| Faithfulness | Deletion / insertion AUC against a blurred baseline (50 steps) | IDRiD, APTOS |
| Sanity | Cascading model randomization (Adebayo *et al.*, 2018); pass if mean \|ρ\| < 0.30 over fractions 0.25 / 0.50 / 1.00 | IDRiD, APTOS |
| Plausibility | AUPRC ratio against expert lesion masks (IDRiD segmentation subset, n = 81) | IDRiD only (APTOS has no masks) |

Explainers: Input-Gradient (baseline), Grad-CAM, Integrated Gradients, Transformer Attribution,
plus Attention Rollout for ViT or a TokenNorm surrogate for Swin.

## Repository contents

```
notebooks/      the four executed benchmark notebooks, with all cell outputs kept (one per backbone × dataset)
preprocessing/  00_IDRiD_preprocessing.ipynb and 00_APTOS_preprocessing.ipynb (executed split, crop,
                CLAHE, augmentation; IDRiD plausibility set)
src/            plausibility_eval.py, loaded by the IDRiD notebooks for the plausibility axis
results/        run summaries written by the notebooks + CSV transcripts of every printed result
scripts/        extract_results.py (outputs → CSV), check_paper_numbers.py (CSV → manuscript tables),
                verify_seg_grading_disjoint.py (IDRiD segmentation-vs-grading screening),
                seg_identity_and_heldout_colab.py (identity decision + Table 6 held-out column),
                duplicate_check_colab.py and aptos_dedup_reeval_colab.py (duplicates across splits),
                make_fig2_per_grade_recall.py and make_fig3_fig4_bspc.py (re-draw Figs. 2–4 from results/)
docs/           RESULTS_MAP.md: which notebook cell prints each number in the paper
data/           download instructions and the folder layout the notebooks expect (no images)
```

The notebooks are the exact files that produced the manuscript results; their outputs have not
been edited or re-run. SHA-256 checksums are in `notebooks/SHA256SUMS` and `preprocessing/SHA256SUMS`.

## Check the reported numbers (no GPU or data needed)

```bash
python scripts/extract_results.py      # parse the saved notebook outputs into results/*.csv
python scripts/check_paper_numbers.py  # compare 136 reported values with those outputs
```

Both scripts use only the Python standard library.

## Re-run the experiments

The notebooks were written for Google Colab (Python 3.10, one GPU per run; see
`docs/RESULTS_MAP.md` for the GPU recorded in each notebook).

1. Download IDRiD and APTOS-2019 (`data/README.md`) and run the offline preprocessing
   (`preprocessing/README.md` and the two `preprocessing/00_*_preprocessing.ipynb` notebooks) to produce
   the archives the notebooks restore from Google Drive. For the IDRiD plausibility section, upload
   `src/plausibility_eval.py` to `/content/` or to the IDRiD Drive folder.
2. Install the dependencies: `pip install -r requirements.txt`.
3. Edit the Drive paths in the configuration cell (`CFG`) and the restore cell of each notebook.
4. Run all cells. The backbone is selected by `CFG.MODEL_NAME` alone; every later cell is
   architecture-aware.

Settings used in the reported runs (configuration cell `CFG`, cell 5, and model cell 15):
AdamW, weight decay 0.01, cosine annealing, mixed precision; 3-epoch head-only warm-up at
learning rate 1e-3, then full fine-tuning at 2e-5; up to 60 epochs, early stopping on validation
QWK (patience 15); batch size 16 (IDRiD) / 32 (APTOS); dropout 0.1 and drop-path 0.1;
label smoothing 0.05; ordinal penalty weight λ = 0.3;
class weights tempered with exponent 0.75 and a 1.6 boost for grade 1; seed 42 for the split and
training, seeds {42, 1, 7} for the explanation subsets (12 images per grade per seed);
Integrated Gradients with 32 steps; sanity check on 30 images.

## Known limitations of this release

- The APTOS split (2,930 / 366 / 366) is the published partition of the Kaggle "APTOS-2019 dataset"
  (`data/README.md`); the benchmark notebooks' text calls it "stratified 70/15/15", which is incorrect.
- 60 of the 81 IDRiD segmentation photographs are grading photographs, 56 of them in the training
  split; Table 6 therefore also reports a held-out column (`preprocessing/README.md`).
- Duplicate photographs across splits (1 IDRiD and 24 APTOS test images) are documented in
  `data/README.md`, with test scores recomputed without them.
- Package versions were not recorded when the notebooks ran; `requirements.txt` explains this.
- Trained model weights are not included.

## Licence and citation

Code: MIT (see `LICENSE`). Datasets keep their own licences and terms (see `data/README.md`).
If you use this code, please cite the paper above; citation metadata is in `CITATION.cff`.
