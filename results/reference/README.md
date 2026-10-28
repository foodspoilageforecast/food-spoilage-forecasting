# Recorded analysis outputs

These files contain the per-cell forecasts and aggregate tables used for the
AI panels in the accompanying manuscript. They are retained as a numerical
reference and are never read by the training procedure.

`predictions.csv` contains 108,000 rows: 18,000 predictions for each of six
methods, including the additional gradient-boosting comparison. Depths are in
centimetres. The RCA-Gompertz method identifier is `replicate_attention`.

`summary_by_method.csv`, `summary_by_method_temp.csv`, `summary_by_meat_temp.csv`
and `summary_by_horizon.csv` contain metrics pooled from their corresponding
prediction rows. `fold_metrics.csv` resolves each method, food, temperature,
held-out replicate and forecast horizon. `trustworthy_horizon.csv` reports
the descriptive R² threshold specified in the Methods documentation.

`neural_model_info.json` records the parameter count and final training loss
for the original five models. The checkpoint files in `models/` are from the
documented release verification run, whose forecasts are stored separately.
Minor neural differences can arise from numerical libraries and hardware;
the original CSVs have not been replaced by newly trained predictions.
