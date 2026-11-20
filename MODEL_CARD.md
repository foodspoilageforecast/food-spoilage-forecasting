# Model card: RCA-Gompertz spoilage-depth forecaster

This model card describes the forecasting model released in this repository. It follows
the Hugging Face model-card outline (https://huggingface.co/docs/hub/model-cards).

## Model details

- **Name:** Replicate-assisted cross-attention neural–Gompertz model (RCA-Gompertz)
- **Version:** 1.0.0 (released 2026-10-28)
- **Type:** Physics-constrained neural time-series forecaster (regression)
- **Architecture:** A shared single-layer GRU encoder (48 hidden units). Learned 48-dimensional
  food, temperature and grid-position embeddings. Four-head cross-attention over reference
  replicates: the query is the target's 0–24 h trajectory, the keys are the references'
  0–24 h trajectories, and the values are the references' 0–36 h trajectories. An anchored
  Gompertz decoder, `depth(h) = A·(d0/A)^exp(−k·h)`, with a bounded residual correction
  (at most 0.45 cm). The outputs are floored at the 24 h depth, capped at 3 cm and made
  non-decreasing.
- **Parameters:** 32,739 trainable parameters, trained from scratch. No pretrained weights
  are used.
- **Framework:** PyTorch 2.10.0, Python 3.12
- **Developers:** Dinghan Wang and co-authors of the accompanying article (Zhu, Pei, Wang,
  Wei, Sun, Wang, Liu & Liu)
- **License:** MIT
- **Contact:** chmlx@nus.edu.sg
- **Citation:** "AI-enabled spatiotemporal mapping and forecasting of internal food
  spoilage" (see `CITATION.cff`)

## Intended use

- **Primary use:** Research use. The model forecasts internal spoilage-penetration depth
  (cm) at each of 25 needle positions (5 × 5 array) for hours 25–36. Its inputs are a
  sample's 0–24 h needle-array measurements plus the complete 0–36 h trajectories of four
  reference replicates of the same food and storage temperature.
- **Covered conditions:** Beef, pork, chicken, duck, fish and minced shrimp, stored
  unpackaged at a constant 4 °C or 25 °C, measured with the needle-array protocol in the
  article.
- **Out of scope:**
  - Food-safety or regulatory decisions, or setting shelf-life or expiry dates.
  - Foods, temperatures, packaging, temperature fluctuations or sensing protocols that were
    not in the training data.
  - Use without reference trajectories.
  - Uses that need calibrated uncertainty. The model gives no prediction intervals.

## Training data

- `data/raw/*.xlsx`: one workbook per food, with sheets for 4 °C and 25 °C. Each condition
  has hourly measurements from 0 to 36 h (37 time points), a 5 × 5 needle grid and five
  biological replicates. See `data/README.md`.
- In total there are 12 conditions and 55,500 depth values (0–3 cm, 0.1 cm resolution).
  These represent 60 replicate trajectories, not 55,500 independent specimens.
- All data were generated in one laboratory with one protocol. Depth is the manually
  identified position of the red-to-green fluorescence transition on each needle.
- Preprocessing in this code: depth is divided by the fixed 3 cm sensing range. No outlier
  removal, imputation or smoothing is applied. The loader checks that the data are complete
  and lie within 0–3 cm.

## Training procedure

- **Data split:** Five-fold leave-one-replicate-out. In fold *j*, replicate index *j* is
  excluded from training for every food and temperature. Each of the other four replicates
  serves in turn as the query, with the remaining three as references. This gives 48
  episodes, or 1,200 needle-position series, per fold.
- **Optimisation:** Full-batch AdamW (learning rate 3e-3 with cosine annealing to 1.5e-4,
  weight decay 1e-4, gradient clipping at 1.0), 600 epochs, seed 42.
- **Loss:** Horizon-weighted `mean|w·e| + 0.3·mean((w·e)²)`, where *w* rises from 1.0 to 1.6
  across the 12 h horizon. Physical constraints are built into the architecture, not added
  as penalty terms.
- **No validation subset or early stopping:** all hyperparameters are the same in every
  fold.
- **Compute:** CPU only. On an Apple M3 laptop with one thread, training all five folds
  takes about 12.5 min (about 13 min including the comparison methods). Inference for all
  18,000 held-out forecasts takes about 4 s, and peak memory is about 1.1 GB.

## Evaluation

The target replicate's 25–36 h measurements are never used as input. They serve only for
scoring. Metrics are pooled over 5 folds × 6 foods × 25 positions × 12 horizons
(9,000 forecasts per temperature):

| Method | MAE 25 °C (cm) | R² 25 °C | MAE 4 °C (cm) | R² 4 °C |
| --- | ---: | ---: | ---: | ---: |
| Persistence (24 h value) | 0.273 | 0.03 | 0.514 | 0.05 |
| Per-cell linear extrapolation | 0.244 | −0.14 | 0.586 | −0.09 |
| Per-cell Gompertz | 0.112 | 0.77 | 0.199 | 0.80 |
| Replicate-informed Gompertz | 0.058 | 0.94 | 0.089 | 0.96 |
| Gradient boosting | 0.030 | 0.97 | 0.041 | 0.99 |
| **RCA-Gompertz** | **0.029** | **0.98** | **0.050** | **0.99** |

Source: `results/reference/summary_by_method_temp.csv`. Per-food and per-horizon metrics are
in `summary_by_meat_temp.csv` and `summary_by_horizon.csv`. Gradient boosting has a lower
MAE than RCA-Gompertz at 4 °C. RCA-Gompertz forecasts are always bounded and
non-decreasing; 578 of the 1,500 gradient-boosting forecast trajectories decrease at some
horizon.

## Limitations and biases

- Results show forecasting for unseen replicates *within* the studied foods, temperatures
  and sample batches. They do not establish generalization to new batches, products,
  packaging formats, fluctuating temperatures or laboratories.
- The reference replicates are parallel samples from the same experiment, and their
  complete futures are given as input. In deployment, references would have to come from
  earlier monitored samples of the same product and storage condition.
- Needle positions, horizons and folds are correlated. The 18,000 forecasts per method are
  not independent observations.
- Depth labels come from manual reading at 0.1 cm resolution and may vary between
  observers.
- The input format is fixed: six foods × two temperatures, 37 hourly points, a 5 × 5 grid
  and five replicates. Other designs need code changes (`MEATS`/`TEMPS` in
  `nn_replicate_attention.py`, and `validate_contexts` in `run_experiment.py`).

## Ethical considerations

The model uses no human or personal data. Its outputs describe chemical spoilage depth, not
microbiological safety, and must not replace food-safety testing or regulatory date labels.

## How to use

```bash
python -m pip install -r requirements.txt
python scripts/prediction/predict.py --checkpoint-dir models      # forecasts from released checkpoints (~4 s)
python scripts/prediction/run_experiment.py                         # full retraining + comparison methods (~13 min)
```

Checkpoints: `models/fold_1.pt` … `fold_5.pt`. Each contains the weights, the held-out
replicate index, the category order, the architecture settings and the depth range. Load
them with `torch.load(..., weights_only=True)`.
