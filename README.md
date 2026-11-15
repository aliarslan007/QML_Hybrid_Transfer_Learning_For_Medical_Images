# Hybrid Transfer Learning for Medical Images

QML course, LUMS, Fall 2026, Project #11.

**Question:** Does a variational quantum circuit (VQC) head on top of a frozen pretrained CNN help on a small medical-image task, compared with a classical head of matched size?

```
PneumoniaMNIST (224x224) -> frozen ResNet-18 -> 512-d features -> PCA to d in {4, 8}
   -> scale to [-pi/2, pi/2] -> head (linear | parameter-matched MLP | VQC) -> pneumonia score
```

## Heads

| Head | d = 4 params | d = 8 params |
|---|---|---|
| Linear | 5 | 9 |
| MLP, d -> h -> 1 (h = 4 / 5) | 25 | 51 |
| VQC, 2 layers (RY RZ RY + CNOT ring), mean of all qubit expectations, then scale and bias | 26 | 50 |
| Reference only: logistic regression on all 512 features | 513 | 513 |

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest -q                       # sanity tests
python scripts/01_extract_features.py     # download data + cache ResNet-18 features (~2 min GPU / ~15 min CPU)
python scripts/02_make_folds.py           # fixed stratified subsamples -> folds/pneumoniamnist_folds.json
python scripts/05_gradient_check.py       # backprop vs parameter-shift gradients must agree
python scripts/03_run_classical.py        # linear + MLP heads, full grid, plus the 512-d reference
python scripts/04_run_vqc.py              # Week-1 VQC pilot: d=4, all n, fold 0, seed 0
python scripts/06_summarize.py            # summary table + learning curves
```

On Google Colab, open `notebooks/week1_colab.ipynb`.

All settings live in `configs/default.yaml`. Result files can be resumed: re-running a script skips runs that have already finished.

## Protocol (fixed in advance)

- **Data:** official MedMNIST v2 splits: 4,708 train / 524 validation / 624 test.
- **Training sizes:** n in {50, 100, 250, 500, 1000}, with 5 stratified subsamples each and 3 initialisation seeds. Every head is trained on the same subsamples.
- **Leakage control:** PCA and scaling are fit on the training split only. The test set is used only for final scoring.
- **Fairness:** every head uses the same loss (class-weighted BCE), full-batch Adam, early stopping on validation loss, and the same learning-rate x weight-decay grid. The best configuration is chosen by validation AUROC.
- **Metrics:** AUROC is primary. Also reported: AUPRC, accuracy, balanced accuracy, F1, sensitivity and specificity, using a threshold chosen on validation (Youden's J).

## Layout

```
configs/   experiment settings
src/       features, folds, reduction, heads, training, metrics, experiment runner
scripts/   numbered pipeline steps
tests/     pytest sanity checks
folds/     saved subsample indices (committed)
results/   CSV results and figures (committed)
report/    LaTeX report
```
