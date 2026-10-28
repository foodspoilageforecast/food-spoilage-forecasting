#!/usr/bin/env python3
"""RCA-Gompertz with shared GRU encoding and replicate cross-attention.

Queries and keys cover hours 0-24. Values contain reference trajectories over
hours 0-36. Each fold excludes one replicate index across all food-temperature
conditions. Training uses three reference blocks; evaluation uses four.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

D_MAX = 3.0
GRID = 5
N_CELLS = GRID * GRID
ANCHOR = 24
MAX_HORIZON = 12
N_BLOCKS = 5
MEATS = ["Beef", "Chicken", "Duck", "Fish", "Pork", "Shrimp"]
TEMPS = ["25C", "4C"]
RANDOM_STATE = 42


# --------------------------------------------------------------------------------------
class RCAGompertz(nn.Module):
    def __init__(self, d_model: int = 48, nhead: int = 4, k_scale: float = 1.0,
                 residual_scale: float = 0.15):
        super().__init__()
        self.d = d_model
        self.k_scale = k_scale
        self.residual_scale = residual_scale
        self.encoder = nn.GRU(input_size=1, hidden_size=d_model, batch_first=True)
        self.meat_emb = nn.Embedding(len(MEATS), d_model)
        self.temp_emb = nn.Embedding(len(TEMPS), d_model)
        self.cell_emb = nn.Embedding(N_CELLS, d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.head_A = nn.Sequential(nn.Linear(2 * d_model + 1, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.head_k = nn.Sequential(nn.Linear(2 * d_model + 1, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.head_res = nn.Sequential(nn.Linear(2 * d_model + 2, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def _encode(self, series):                       # series (N, T, 1) -> (N, d)
        out, _ = self.encoder(series)
        return out[:, -1]

    def forward(self, query_series, key_series, value_series, ctx_valid, meat_idx, temp_idx, cell_idx):
        """query_series (N, Tq, 1)   held-out block, 0-24 h  -> Q
        key_series   (N, C, Tk, 1)   training blocks, 0-24 h -> K  (early, for matching)
        value_series (N, C, Tv, 1)   training blocks, 0-36 h -> V  (full, for retrieval)
        ctx_valid (N, C) bool. Returns predicted normalised depth (N, MAX_HORIZON).

        Attention matches the held-out early curve (Q) against the training blocks' EARLY
        curves (K); the resulting weights then read out the training blocks' FULL kinetics (V).
        So similarity is an early-vs-early comparison and only the retrieved value carries the
        25-36 h dynamics (from training blocks, never the held-out block)."""
        n, c, tk, _ = key_series.shape
        tv = value_series.shape[2]
        cond = self.meat_emb(meat_idx) + self.temp_emb(temp_idx) + self.cell_emb(cell_idx)  # (N,d)
        q = self._encode(query_series) + cond                                               # (N,d)
        K = self._encode(key_series.reshape(n * c, tk, 1)).reshape(n, c, self.d) + cond[:, None, :]
        V = self._encode(value_series.reshape(n * c, tv, 1)).reshape(n, c, self.d) + cond[:, None, :]
        attended, _ = self.attn(q[:, None, :], K, V, key_padding_mask=~ctx_valid)           # (N,1,d)
        borrowed = self.norm(q + attended[:, 0])                                            # (N,d)

        d0 = query_series[:, -1, 0].clamp(1e-3, 1.0)                                         # anchor depth
        feat = torch.cat([q, borrowed], dim=-1)
        a_frac = torch.sigmoid(self.head_A(torch.cat([feat, d0[:, None]], -1)).squeeze(-1))
        A = (d0 + (1.0 - d0) * a_frac).clamp_min(d0 + 1e-3)
        k = F.softplus(self.head_k(torch.cat([feat, d0[:, None]], -1)).squeeze(-1)) * self.k_scale + 1e-3

        hs = torch.arange(1, MAX_HORIZON + 1, device=query_series.device, dtype=query_series.dtype)
        ek = torch.exp(-k[:, None] * hs[None, :])                                            # (N,H)
        base = A[:, None] * (d0[:, None] / A[:, None]) ** ek
        res_in = torch.cat([feat[:, None, :].expand(n, MAX_HORIZON, 2 * self.d),
                            ek[:, :, None], (hs / MAX_HORIZON)[None, :, None].expand(n, MAX_HORIZON, 1)], -1)
        residual = self.residual_scale * torch.tanh(self.head_res(res_in).squeeze(-1)) * (1.0 - ek)
        depth = torch.maximum(base + residual, d0[:, None]).clamp(max=1.0)
        return torch.cummax(depth, dim=1).values


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# --------------------------------------------------------------------------------------
# Episode construction (leakage-safe) and tensor packing
# --------------------------------------------------------------------------------------
def _series(block_field, t0, t1):
    """block_field (hours,5,5) -> (25 cells, t1-t0, 1) normalised depth."""
    seg = block_field[t0:t1] / D_MAX                       # (T,5,5)
    return np.transpose(seg, (1, 2, 0)).reshape(N_CELLS, seg.shape[0], 1)


def _pad_set(stack, max_ctx):
    """Pad the block axis (axis=1) of a (N,C,T,1) stack up to max_ctx; return padded, valid."""
    pad = max_ctx - stack.shape[1]
    valid = np.array([True] * stack.shape[1] + [False] * pad)
    if pad > 0:
        stack = np.concatenate([stack, np.zeros((N_CELLS, pad, stack.shape[2], 1), np.float32)], axis=1)
    return stack, valid


def build_training_batch(ctx_map, held_out_j, device, max_ctx=N_BLOCKS - 1):
    """Episodes for global model j: per meat-temp, query each of the 4 non-j blocks using the
    other 3 as context. Block j is never used. Keys are the context blocks' EARLY 0-24 h,
    values are their FULL 0-36 h. Returns packed tensors on `device`."""
    Q, Kk, Vv, VA, M, Tp, CE, Y = [], [], [], [], [], [], [], []
    for meat in MEATS:
        for temp in TEMPS:
            c = ctx_map.get((meat, temp))
            if c is None:
                continue
            tensor = np.asarray(c.tensor, dtype=np.float32)   # (hours,5,5,block)
            avail = [b for b in range(tensor.shape[3]) if b != held_out_j]
            for qb in avail:
                ctx_blocks = [b for b in avail if b != qb]
                query = _series(tensor[:, :, :, qb], 0, ANCHOR + 1)                 # held-out early
                keys = np.stack([_series(tensor[:, :, :, b], 0, ANCHOR + 1) for b in ctx_blocks], axis=1)  # 0-24 h
                vals = np.stack([_series(tensor[:, :, :, b], 0, tensor.shape[0]) for b in ctx_blocks], axis=1)  # 0-36 h
                keys, valid = _pad_set(keys, max_ctx)
                vals, _ = _pad_set(vals, max_ctx)
                target = np.transpose(tensor[ANCHOR + 1: ANCHOR + 1 + MAX_HORIZON, :, :, qb], (1, 2, 0)
                                      ).reshape(N_CELLS, MAX_HORIZON) / D_MAX
                Q.append(query); Kk.append(keys); Vv.append(vals); VA.append(np.tile(valid, (N_CELLS, 1)))
                M.append(np.full(N_CELLS, MEATS.index(meat))); Tp.append(np.full(N_CELLS, TEMPS.index(temp)))
                CE.append(np.arange(N_CELLS)); Y.append(target)
    pack = lambda arrs, dt: torch.tensor(np.concatenate(arrs, axis=0), dtype=dt, device=device)
    return (pack(Q, torch.float32), pack(Kk, torch.float32), pack(Vv, torch.float32), pack(VA, torch.bool),
            pack(M, torch.long), pack(Tp, torch.long), pack(CE, torch.long), pack(Y, torch.float32))


def train_global_model(ctx_map, held_out_j, device, epochs=600, lr=3e-3, history=None):
    torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)
    q, kk, vv, va, m, tp, ce, y = build_training_batch(ctx_map, held_out_j, device)
    model = RCAGompertz().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)
    model.train()
    for epoch in range(epochs):
        pred = model(q, kk, vv, va, m, tp, ce)
        w = torch.linspace(1.0, 1.6, MAX_HORIZON, device=device)[None, :]
        loss = F.l1_loss(pred * w, y * w) + 0.3 * F.mse_loss(pred * w, y * w)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if history is not None:
            history.append(dict(epoch=epoch + 1, loss=float(loss.detach()),
                                learning_rate=float(sched.get_last_lr()[0])))
    model.eval()
    return model, float(loss.detach()), count_parameters(model)


@torch.no_grad()
def predict_heldout(model, ctx_map, held_out_j, device):
    """Predict block `held_out_j`'s 25-36 h for every meat-temp, using the 4 other blocks as
    context and only the held-out block's 0-24 h as query. Returns list[dict] (schema rows)."""
    rows = []
    for meat in MEATS:
        for temp in TEMPS:
            c = ctx_map.get((meat, temp))
            if c is None or held_out_j >= c.tensor.shape[3]:
                continue
            tensor = np.asarray(c.tensor, dtype=np.float32)
            ctx_blocks = [b for b in range(tensor.shape[3]) if b != held_out_j]
            query = _series(tensor[:, :, :, held_out_j], 0, ANCHOR + 1)              # held-out early 0-24 h
            keys = np.stack([_series(tensor[:, :, :, b], 0, ANCHOR + 1) for b in ctx_blocks], axis=1)   # 0-24 h
            vals = np.stack([_series(tensor[:, :, :, b], 0, tensor.shape[0]) for b in ctx_blocks], axis=1)  # 0-36 h
            valid = np.tile(np.array([True] * keys.shape[1]), (N_CELLS, 1))
            fut_hours = np.asarray(c.hours[ANCHOR + 1: ANCHOR + 1 + MAX_HORIZON], float)
            t = lambda a, dt: torch.tensor(a, dtype=dt, device=device)
            pred = model(t(query, torch.float32), t(keys, torch.float32), t(vals, torch.float32),
                         t(valid, torch.bool),
                         t(np.full(N_CELLS, MEATS.index(meat)), torch.long),
                         t(np.full(N_CELLS, TEMPS.index(temp)), torch.long),
                         t(np.arange(N_CELLS), torch.long)).cpu().numpy() * D_MAX     # (25,H) cm
            true = tensor[ANCHOR + 1: ANCHOR + 1 + MAX_HORIZON, :, :, held_out_j]      # (H,5,5)
            train_ids = "+".join(str(b + 1) for b in ctx_blocks)
            for cell in range(N_CELLS):
                ix, iy = cell // GRID, cell % GRID
                for hi in range(MAX_HORIZON):
                    rows.append(dict(
                        meat=meat, temperature=temp, fold=held_out_j + 1, train_block_ids=train_ids,
                        test_block_id=held_out_j + 1, method="replicate_attention",
                        time=float(fut_hours[hi]), horizon=hi + 1, x=ix + 1, y=iy + 1, z=0,
                        true_depth=float(true[hi, ix, iy]),
                        predicted_depth=float(np.clip(pred[cell, hi], 0.0, D_MAX))))
    return rows


def run_all(contexts, device_str="cpu", epochs=600, folds=None, output_dir=None):
    device = torch.device(device_str)
    ctx_map = {(c.meat, c.temp_label): c for c in contexts}
    rows, info = [], []
    for j in (range(N_BLOCKS) if folds is None else folds):
        history = []
        model, loss, params = train_global_model(ctx_map, j, device, epochs=epochs, history=history)
        rows.extend(predict_heldout(model, ctx_map, j, device))
        info.append(dict(held_out_block=j + 1, final_loss=round(loss, 5), parameters=params))
        if output_dir is not None:
            checkpoint_dir = Path(output_dir) / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(dict(state_dict=model.cpu().state_dict(), held_out_block=j + 1,
                            epochs=epochs, seed=RANDOM_STATE, d_model=48, nhead=4,
                            k_scale=1.0, residual_scale=0.15, meats=MEATS, temperatures=TEMPS,
                            depth_ceiling_cm=D_MAX, anchor_hour=ANCHOR, max_horizon=MAX_HORIZON),
                       checkpoint_dir / f"fold_{j + 1}.pt")
            pd.DataFrame(history).to_csv(checkpoint_dir / f"fold_{j + 1}_training.csv", index=False)
        print(f"  RCA-Gompertz global model j={j+1}: loss={loss:.4f} params={params:,}", flush=True)
    return pd.DataFrame(rows), info
