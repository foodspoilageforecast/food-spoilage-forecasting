# Forecasting measurements

`raw/` contains one Excel workbook for each of Beef, Chicken, Duck, Fish, Pork,
and Shrimp. Shrimp is labelled "Minced shrimp" in the manuscript.

Each workbook contains the 4 °C and 25 °C conditions, with hourly measurements
from 0 through 36 h. Each hour has 25 locations, indexed by `x` and `y` from 1
to 5. The five columns headed `z` identify five parallel replicate pieces.
Their values are spoilage penetration depths in centimetres.

The loader identifies each time block by its hour label followed by the `x`,
`y` and five `z` headers. Time blocks may appear side by side or in successive
row groups. The assembled tensor has shape `(37, 5, 5, 5)`, in the order
`(hour, x, y, replicate)`. Replicates are retained separately throughout training
and evaluation; they are not averaged into a depth axis.

There are 12 food-temperature conditions and 55,500 scalar depth measurements.
They represent 60 piece trajectories, rather than 55,500 independent specimens.
Depth is bounded by 3 cm. Values quantify the thickness of the spoiled layer:
0.2 cm means that material from the surface down to 0.2 cm is spoiled.

For every fold, reference pieces contribute all 37 time points. The target
piece contributes hours 0-24 as input and hours 25-36 only for evaluation.
Each output prediction row contains food, temperature, fold, reference block
identifiers, target block identifier, hour, forecast horizon, location, measured
depth, and predicted depth. Block and grid indices are one-based in the CSVs.

The `z=0` column retained in the historical reference prediction CSV is an unused
legacy placeholder. It is not a spatial coordinate or measurement; new result
exports omit it.

SHA-256 checksums of the distributed workbooks are recorded in
`input_checksums.json`. They establish which input files were used for the
release validation. Experimental collection and calibration are described in
the accompanying manuscript and Supplementary Methods.
