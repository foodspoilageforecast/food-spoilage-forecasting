# Forecasting methods

## Information available to each model

The experiment predicts spoilage depth at hours 25-36 for a target replicate.
Its inputs contain the target's hours 0-24 and four reference replicates from
the same food-temperature condition over hours 0-36. The target's later
measurements are used only to score the forecasts.

Five folds exclude replicate indices 1-5 in turn. For the model associated
with index j, that index is absent from every training episode across all
foods and temperatures. Of the four remaining indices, each becomes a training
query in turn, with the other three as references. There are 48 episodes per
fold (6 foods × 2 temperatures × 4 query choices), each contributing 25 cells,
or 1,200 cell-level rows. Reference sets are padded to four entries and masked.
At evaluation, all four non-held-out pieces are valid references.

## Encoder and reference retrieval

Depth is divided by the fixed 3 cm sensing range; no sample-derived scaling
statistic is fitted. A shared, single-layer GRU encodes scalar trajectories
into 48 features. Its final recurrent output summarizes the trajectory.
Learned 48-dimensional food, temperature and cell embeddings are added to
the trajectory features. Food and temperature embedding orders are recorded
in each checkpoint.

Four-head cross-attention uses the target's 25-point early representation as
the query, reference 25-point early representations as keys, and reference
37-point complete representations as values. Matching therefore uses early
trajectories on both sides; the values carry later reference kinetics.
The attention output is added to the query and layer-normalized. Concatenating
the query and this combined representation supplies the decoder.

Each fold trains one model shared by all six foods and both temperatures.
The architecture treats cells as individual series conditioned on location.
It contains no neighbourhood convolution, spatial message passing or explicit
spatial continuity loss.

## Anchored decoder

For normalized anchor depth d0 at hour 24, the main decoder has the form

`depth(h) = A * (d0 / A) ** exp(-k * h)`.

A sigmoid head places the asymptote between the anchor and the normalized
depth ceiling. A numerical floor of d0 + 0.001 is applied to A, and d0 is
clamped to [0.001, 1]. This floor can place the internal A slightly above one
for a saturated anchor; final output clipping still enforces the ceiling.
A softplus head produces a positive rate, with a floor of 0.001.

The correction head receives the combined representation, exp(-k h) and h/12.
Its output is multiplied by 0.15, transformed through tanh, and multiplied
by 1 - exp(-k h), so the correction vanishes at the anchor and is bounded
by 0.15 normalized units, equivalent to 0.45 cm. Final forecasts are floored
at d0, capped at one, and passed through a cumulative maximum over horizons.
Thus final predictions are bounded and non-decreasing. Deceleration describes
the base Gompertz component; it is not a separate hard constraint on the
residual-corrected trajectory.

## Optimization

The random seed is 42. Each fold uses 600 full-batch epochs, AdamW, initial
learning rate 0.003, weight decay 0.0001, gradient-norm clipping at 1.0, and
cosine learning-rate annealing to 0.00015. There is no early stopping or
separate validation subset. The 32,739 parameters are freshly initialized
for each fold.

Horizon weights increase linearly from 1.0 to 1.6 over 12 hours. In normalized
depth units the implemented objective is

`mean(abs(w * (prediction - target))) + 0.3 * mean((w * (prediction - target)) ** 2)`.

The squared-error term therefore contains squared horizon weights. No additional
physics penalty is added to this loss: physical bounds and monotonicity are
applied in the forward computation. AdamW provides parameter weight decay.
The training history records the loss before each parameter update and the
learning rate after the scheduler step.

## Comparison methods

- Persistence repeats the target depth measured at hour 24.
- Linear extrapolation fits each target cell over hours 0-24.
- Per-cell Gompertz fits each target cell over hours 0-24 by bounded least squares.
- Replicate-informed Gompertz combines that early fit with median parameters
  fitted to the four complete reference trajectories. Fixed shrinkage weights
  are 0.7 for the asymptote and 0.5 for the rate and inflection time.
- Gradient boosting uses the target's early statistics, coordinates, target
  hour and reference-mean depth at the target hour. Training queries obtain
  their reference features from the other training pieces. The estimator is
  HistGradientBoostingRegressor with maximum depth 4, 250 iterations,
  learning rate 0.06, L2 regularization 1.0 and random seed 42.

All comparison outputs are clipped to [0, 3] cm. Failed Gompertz fits use the
fallback described in `baselines.py`. The complete result tables include
gradient boosting; the five-method paper plots exclude that additional method.

## Metrics and visualization

MAE, RMSE and signed bias use centimetres. R² is calculated as one minus the
sum of squared errors divided by the sum of squared deviations of the measured
depths from their pooled mean. R² is missing when that denominator is below
the numerical tolerance. Metrics in pooled tables are calculated from the
individual forecasts, not averaged from subgroup R² values.

Across five folds, there are 18,000 predictions per method: 5 folds × 6 foods ×
2 temperatures × 25 cells × 12 horizons. A derived "trustworthy horizon" table
reports the initial uninterrupted run of horizons with R² at least 0.8. This
threshold is descriptive and is not a validated food-safety criterion.

The renderer interpolates the 5 × 5 measured or predicted depth field for
display. A cross-section labels tissue above the depth boundary as spoiled
and deeper tissue as unspoiled. Shared material colours are #7a7751 and
#b39b89, respectively; lighting changes their displayed brightness. Smooth
rendering does not increase the underlying sensing resolution. The lateral
display coordinates are grid positions and should not be interpreted as a
separately calibrated physical length.

## Scope of validation

Completed reference futures are part of the defined input. Replicate and
spatial correlations reduce the number of independent observations. The
reported results do not establish generalization to new batches, foods,
temperatures or laboratories, nor do they provide prediction intervals.
