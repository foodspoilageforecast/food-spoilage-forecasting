"""Read the five replicate trajectories from each food workbook."""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT / "data" / "raw"
HOUR_RE = re.compile(r"(?<!\d)(\d+)\s*h\b", re.IGNORECASE)
TEMP_RE = re.compile(r"(-?\d+)\s*[℃°C]")
PIECE_COLS = [f"piece{i}" for i in range(1, 6)]

@dataclass(slots=True)
class Context:
    meat: str
    temp_label: str
    temp_c: float
    hours: list[int]
    x_values: list[int]
    y_values: list[int]
    tensor: np.ndarray

def temp_label(temp_c: float) -> str:
    if float(temp_c).is_integer():
        return f"{int(temp_c)}C"
    return f"{temp_c:g}C"


def find_time_blocks(raw: pd.DataFrame) -> list[tuple[int, int, int]]:
    blocks: list[tuple[int, int, int]] = []
    for row_idx in range(raw.shape[0] - 2):
        for col_idx in range(raw.shape[1] - 6):
            cell = raw.iat[row_idx, col_idx]
            if not isinstance(cell, str):
                continue
            matched = HOUR_RE.search(cell)
            if not matched:
                continue
            x_header = str(raw.iat[row_idx + 1, col_idx]).strip().lower()
            y_header = str(raw.iat[row_idx + 1, col_idx + 1]).strip().lower()
            z_headers = [str(v).strip().lower() for v in raw.iloc[row_idx + 1, col_idx + 2 : col_idx + 7]]
            if x_header == "x" and y_header == "y" and all(h.startswith("z") for h in z_headers):
                blocks.append((row_idx, col_idx, int(matched.group(1))))
    return blocks


def parse_context_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    matched_temp = TEMP_RE.search(sheet_name)
    if not matched_temp:
        raise ValueError(f"Cannot parse temperature from sheet name: {path.name}::{sheet_name}")
    temp_c = float(matched_temp.group(1))
    rows: list[tuple[str, str, float, int, int, int, float, float, float, float, float]] = []

    for header_row, col, hour in find_time_blocks(raw):
        for data_row in range(header_row + 2, min(header_row + 27, raw.shape[0])):
            x_val = raw.iat[data_row, col]
            y_val = raw.iat[data_row, col + 1]
            if pd.isna(x_val) or pd.isna(y_val):
                continue
            pieces = raw.iloc[data_row, col + 2 : col + 7].astype(float).to_numpy()
            if len(pieces) != 5 or np.isnan(pieces).any():
                continue
            rows.append(
                (
                    path.stem,
                    temp_label(temp_c),
                    temp_c,
                    hour,
                    int(x_val),
                    int(y_val),
                    *[float(v) for v in pieces],
                )
            )

    if not rows:
        return pd.DataFrame(
            columns=["meat", "temp_label", "temp_c", "hour", "x", "y", *PIECE_COLS]
        )
    return (
        pd.DataFrame(
            rows,
            columns=["meat", "temp_label", "temp_c", "hour", "x", "y", *PIECE_COLS],
        )
        .drop_duplicates()
        .sort_values(["meat", "temp_c", "hour", "x", "y"], kind="mergesort")
        .reset_index(drop=True)
    )


def dataframe_to_context(df: pd.DataFrame) -> Context:
    hours = sorted(int(v) for v in df["hour"].unique())
    x_values = sorted(int(v) for v in df["x"].unique())
    y_values = sorted(int(v) for v in df["y"].unique())
    h2i = {h: i for i, h in enumerate(hours)}
    x2i = {x: i for i, x in enumerate(x_values)}
    y2i = {y: i for i, y in enumerate(y_values)}
    tensor = np.full((len(hours), len(x_values), len(y_values), 5), np.nan, dtype=float)
    for _, row in df.iterrows():
        tensor[h2i[int(row["hour"])], x2i[int(row["x"])], y2i[int(row["y"])], :] = row[
            PIECE_COLS
        ].to_numpy(dtype=float)

    if np.isnan(tensor).any():
        raise ValueError(f"Missing values after tensor assembly: {df.iloc[0]['meat']} {df.iloc[0]['temp_label']}")

    return Context(
        meat=str(df.iloc[0]["meat"]),
        temp_label=str(df.iloc[0]["temp_label"]),
        temp_c=float(df.iloc[0]["temp_c"]),
        hours=hours,
        x_values=x_values,
        y_values=y_values,
        tensor=tensor,
    )


def load_contexts(data_dir: Path, exclude_meats: set[str]) -> tuple[list[Context], pd.DataFrame]:
    contexts: list[Context] = []
    parsed_frames: list[pd.DataFrame] = []
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for path in sorted(data_dir.glob("*.xlsx")):
        if path.stem in exclude_meats:
            continue
        with pd.ExcelFile(path) as xls:
            for sheet_name in xls.sheet_names:
                sheet_df = parse_context_sheet(path, sheet_name)
                if sheet_df.empty:
                    warnings.warn(f"No valid time blocks parsed from {path.name}::{sheet_name}")
                    continue
                contexts.append(dataframe_to_context(sheet_df))
                parsed_frames.append(sheet_df)

    if not contexts:
        raise ValueError(f"No contexts parsed from {data_dir}")
    long_df = pd.concat(parsed_frames, ignore_index=True)
    return contexts, long_df
