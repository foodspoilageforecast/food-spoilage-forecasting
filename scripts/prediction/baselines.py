"""Reference methods and forecast error summaries for the replicate experiment."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ANCHOR_HOUR = 24
MAX_HORIZON = 12
DEPTH_CEILING = 3.0
SHRINK_A, SHRINK_K, SHRINK_TC = 0.7, 0.5, 0.5

def gompertz(t, A, k, tc):
    return A * np.exp(-np.exp(-k * (t - tc)))


def fit_gompertz(hours: np.ndarray, depth: np.ndarray):
    """Robust bounded Gompertz least-squares. Returns (A,k,tc) or None."""
    depth = np.asarray(depth, dtype=float)
    if np.nanmax(depth) <= 1e-6 or np.allclose(depth, depth[0]):
        return None
    A0 = max(float(np.nanmax(depth)), 0.05)
    try:
        popt, _ = curve_fit(
            gompertz, hours, depth, p0=[A0, 0.3, float(np.median(hours))],
            bounds=([1e-3, 1e-3, -50.0], [DEPTH_CEILING * 1.15, 5.0, 200.0]), maxfev=8000,
        )
        return tuple(float(v) for v in popt)
    except (ValueError, RuntimeError, FloatingPointError):
        return None


def clamp(a):
    return np.clip(a, 0.0, DEPTH_CEILING)


def m_persistence(test_obs, hours, fut_hours):
    d0 = test_obs[ANCHOR_HOUR]                                  # (nx,ny)
    return clamp(np.repeat(d0[None], MAX_HORIZON, axis=0))


def m_linear(test_obs, hours, fut_hours):
    nx, ny = test_obs.shape[1:]
    obs_h = hours[: ANCHOR_HOUR + 1]                           # 0..24
    pred = np.zeros((MAX_HORIZON, nx, ny))
    for ix in range(nx):
        for iy in range(ny):
            y = test_obs[: ANCHOR_HOUR + 1, ix, iy]
            b1, b0 = np.polyfit(obs_h, y, 1)
            pred[:, ix, iy] = b0 + b1 * fut_hours
    return clamp(pred)


def m_per_cell_gompertz(test_obs, hours, fut_hours):
    nx, ny = test_obs.shape[1:]
    obs_h = hours[: ANCHOR_HOUR + 1]
    pred = np.zeros((MAX_HORIZON, nx, ny))
    for ix in range(nx):
        for iy in range(ny):
            p = fit_gompertz(obs_h, test_obs[: ANCHOR_HOUR + 1, ix, iy])
            if p is None:
                pred[:, ix, iy] = test_obs[ANCHOR_HOUR, ix, iy]      # fallback: persistence
            else:
                pred[:, ix, iy] = gompertz(fut_hours, *p)
    return clamp(pred)


def _train_block_priors(train_blocks, hours):
    """Per-cell median Gompertz (A,k,tc) from the 4 training blocks' FULL 0-36 h trajectories."""
    n_hours, nx, ny, nb = train_blocks.shape
    priors = np.full((nx, ny, 3), np.nan)
    for ix in range(nx):
        for iy in range(ny):
            params = []
            for b in range(nb):
                p = fit_gompertz(hours, train_blocks[:, ix, iy, b])
                if p is not None:
                    params.append(p)
            if params:
                priors[ix, iy] = np.median(np.array(params), axis=0)
    return priors                                                  # (nx,ny,3), NaN where none


def m_replicate_informed_gompertz(test_obs, hours, fut_hours, priors):
    """Held-out per-cell Gompertz fit on 0-24 h, shrunk toward the training-block prior
    (strong shrinkage on the under-determined asymptote A). Falls back to the prior when the
    held-out early fit fails. Fixed a-priori shrinkage weights; no test data used to tune."""
    nx, ny = test_obs.shape[1:]
    obs_h = hours[: ANCHOR_HOUR + 1]
    pred = np.zeros((MAX_HORIZON, nx, ny))
    for ix in range(nx):
        for iy in range(ny):
            prior = priors[ix, iy]
            ho = fit_gompertz(obs_h, test_obs[: ANCHOR_HOUR + 1, ix, iy])
            if np.any(np.isnan(prior)):
                params = ho                                         # no prior available
            elif ho is None:
                params = tuple(prior)                               # early fit failed -> prior
            else:
                A = (1 - SHRINK_A) * ho[0] + SHRINK_A * prior[0]
                k = (1 - SHRINK_K) * ho[1] + SHRINK_K * prior[1]
                tc = (1 - SHRINK_TC) * ho[2] + SHRINK_TC * prior[2]
                params = (A, k, tc)
            if params is None:
                pred[:, ix, iy] = test_obs[ANCHOR_HOUR, ix, iy]
            else:
                pred[:, ix, iy] = gompertz(fut_hours, *params)
    return clamp(pred)


def m_gbt(train_blocks, test_obs, hours, fut_hours):
    """Optional Gradient-Boosting regressor. Trains on the 4 training blocks; each block's
    replicate features come from the OTHER training blocks, mirroring how the held-out block
    uses all 4 training blocks. Held-out future is never used for training."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    n_hours, nx, ny, nb = train_blocks.shape
    obs_slice = slice(0, ANCHOR_HOUR + 1)

    def block_features(block_field, repl_mean_at, ix, iy, t):
        d0 = block_field[ANCHOR_HOUR, ix, iy]
        early = block_field[obs_slice, ix, iy]
        slope = (early[-1] - early[-7]) / 6.0 if len(early) >= 7 else 0.0
        return [t, ix, iy, d0, float(early.mean()), float(slope),
                float(repl_mean_at[ANCHOR_HOUR, ix, iy]), float(repl_mean_at[int(t), ix, iy])]

    X, y = [], []
    for b in range(nb):
        others = [o for o in range(nb) if o != b]
        repl_mean = train_blocks[:, :, :, others].mean(axis=3)      # (hours,nx,ny)
        for ix in range(nx):
            for iy in range(ny):
                for t in fut_hours:
                    X.append(block_features(train_blocks[:, :, :, b], repl_mean, ix, iy, t))
                    y.append(train_blocks[int(t), ix, iy, b])
    model = HistGradientBoostingRegressor(max_depth=4, max_iter=250, learning_rate=0.06,
                                          l2_regularization=1.0, random_state=42)
    model.fit(np.array(X), np.array(y))
    repl_mean_all = train_blocks.mean(axis=3)                       # 4-block mean
    pred = np.zeros((MAX_HORIZON, nx, ny))
    for hi, t in enumerate(fut_hours):
        feats = [block_features(test_obs, repl_mean_all, ix, iy, t)
                 for ix in range(nx) for iy in range(ny)]
        out = model.predict(np.array(feats)).reshape(nx, ny)
        pred[hi] = out
    return clamp(pred)


def metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    true = np.asarray(true, float); pred = np.asarray(pred, float)
    err = pred - true
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return dict(
        mae=float(np.mean(np.abs(err))),
        rmse=float(np.sqrt(np.mean(err ** 2))),
        r2=(1.0 - float(np.sum(err ** 2)) / ss_tot) if ss_tot > 1e-9 else np.nan,
        bias=float(np.mean(pred) - np.mean(true)),
        n=int(true.size),
    )


def trustworthy(group: pd.DataFrame) -> int:
    held = 0
    for _, r in group.sort_values("horizon").iterrows():
        if pd.isna(r["r2"]) or r["r2"] < 0.8:
            break
        held = int(r["horizon"])
    return held


def aggregate(preds: pd.DataFrame):
    def block(df):
        return metrics(df["true_depth"].to_numpy(), df["predicted_depth"].to_numpy())

    fold_rows = []
    for keys, g in preds.groupby(["method", "meat", "temperature", "fold", "horizon"]):
        fold_rows.append(dict(zip(["method", "meat", "temperature", "fold", "horizon"], keys), **block(g)))
    fold_metrics = pd.DataFrame(fold_rows)

    def summarize(by):
        rows = []
        for keys, g in preds.groupby(by):
            keys = keys if isinstance(keys, tuple) else (keys,)
            rows.append(dict(zip(by, keys), **block(g)))
        return pd.DataFrame(rows)

    by_method = summarize(["method"])
    by_method_temp = summarize(["method", "temperature"])
    by_meat_temp = summarize(["method", "meat", "temperature"])
    by_horizon = summarize(["method", "temperature", "horizon"])

    # trustworthy horizon per (method, temperature) and (method, meat, temperature)
    th_rows = []
    for (method, temp), g in by_horizon.groupby(["method", "temperature"]):
        th_rows.append(dict(method=method, meat="ALL", temperature=temp, trustworthy_horizon=trustworthy(g)))
    per_meat_h = summarize(["method", "meat", "temperature", "horizon"])
    for (method, meat, temp), g in per_meat_h.groupby(["method", "meat", "temperature"]):
        th_rows.append(dict(method=method, meat=meat, temperature=temp, trustworthy_horizon=trustworthy(g)))
    trustworthy_df = pd.DataFrame(th_rows)
    return fold_metrics, by_method, by_method_temp, by_meat_temp, by_horizon, trustworthy_df
