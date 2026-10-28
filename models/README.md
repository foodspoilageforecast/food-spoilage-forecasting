# Release checkpoints

`fold_1.pt` through `fold_5.pt` were trained from the distributed workbooks
using the release implementation, seed 42, CPU execution, one numerical thread,
and 600 full-batch epochs per fold. Each checkpoint contains model weights,
the excluded replicate index, category order, architecture settings and the
physical depth range. Only model state dictionaries and basic metadata are
serialized; inference loads them with `weights_only=True`.

`fold_*_training.csv` records training loss and scheduled learning rate.
`run_metadata.json` records the validation environment and input checksums.
`predictions.csv` and `summary_by_method_temp.csv` record the forecasts and
temperature-level metrics produced by these checkpoints.

The recorded manuscript forecasts remain separately available in
`results/reference/`. These checkpoints are from the release verification run;
they are not represented as the original archived training state. Neural
training can differ slightly across library versions and numerical backends.
The observed per-cell differences from the manuscript outputs, and the
baseline re-computation check, are recorded in `docs/reproduction_check.json`.

To reproduce the checkpoint forecasts:

```bash
python scripts/prediction/predict.py --checkpoint-dir models
```

The target replicate's future is used only to populate evaluation columns and
metrics. The model receives its hours 0-24 and the full reference trajectories.
