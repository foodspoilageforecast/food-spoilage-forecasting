#!/usr/bin/env python3
"""Cut-away blocks showing measured or predicted spoilage penetration depth.

Depth is measured from the top surface in centimetres. The 5 by 5 measurement
grid is interpolated for display. All foods share the same tissue colours.
"""
from __future__ import annotations

import sys
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LightSource, to_rgb
from scipy.ndimage import zoom

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

D_MAX = 3.0
GRID = 5

GOOD_TISSUE_COLOR = "#b39b89"
SPOILED_TISSUE_COLOR = "#7a7751"
UNIFORM_PALETTE = dict(
    fresh=GOOD_TISSUE_COLOR,
    deep=GOOD_TISSUE_COLOR,
    marble=0.0,
    spoil=[SPOILED_TISSUE_COLOR, SPOILED_TISSUE_COLOR],
)


def palette_for(meat):
    """Return the shared two-colour material; ``meat`` is intentionally ignored."""
    p = UNIFORM_PALETTE
    cmap = LinearSegmentedColormap.from_list("spoil", p["spoil"], N=512)
    return cmap, np.array(to_rgb(p["fresh"])), np.array(to_rgb(p["deep"])), p.get("marble", 0.08)


def _marble(m: int, k: int, seed: int, strength: float) -> np.ndarray:
    """Deterministic fatty-tissue streaking as a (m,k) brightness multiplier around 1.0."""
    rng = np.random.default_rng(seed)
    base = zoom(rng.standard_normal((max(2, m // 8), max(2, k // 8))), (m / max(2, m // 8), k / max(2, k // 8)),
                order=3)[:m, :k]
    streak = zoom(rng.standard_normal((max(2, m // 20), 1)), (m / max(2, m // 20), k), order=3)[:m, :k]
    tex = 0.7 * base / (np.abs(base).max() + 1e-6) + 0.3 * streak / (np.abs(streak).max() + 1e-6)
    return np.clip(1.0 + strength * tex, 0.6, 1.35)


SPOIL_CMAP, FRESH_RGB, FRESH_DEEP, _MARBLE = palette_for(None)   # generic default
EDGE_RGB = np.array(to_rgb("#3a2a22"))
_LS = LightSource(azdeg=320, altdeg=50)


def _upsample(field: np.ndarray, factor: int) -> np.ndarray:
    return zoom(field, factor, order=3, mode="nearest")


def _shade(rgb: np.ndarray, normal_z: float, ambient: float = 0.55) -> np.ndarray:
    """Cheap Lambert term for a flat face given its upward-facing component."""
    bright = ambient + (1.0 - ambient) * np.clip(0.35 + 0.65 * normal_z, 0.0, 1.0)
    return np.clip(rgb * bright, 0.0, 1.0)


def _two_tone_face(depth_along: np.ndarray, z_grid: np.ndarray, base_normal: float,
                   spoil_cmap, fresh_rgb, fresh_deep, marble: float, seed: int,
                   soft: float = 0.0) -> np.ndarray:
    """Colour a vertical face: spoiled (top, within `depth_along` of surface) vs fresh,
    with fatty marbling on the flesh and a subtle vertical ambient-occlusion gradient.

    depth_along : (M,) penetration depth in cm along the face's horizontal axis (up-sampled)
    z_grid      : (M, K) z coordinate (cm) of each quad, 0=bottom .. D_MAX=top surface
    returns RGB (M, K, 3)
    """
    front = D_MAX - depth_along[:, None]                      # z of the spoiled/fresh boundary
    spoiled_frac = (z_grid >= front).astype(float)
    depth_from_surface = np.clip((D_MAX - z_grid) / D_MAX, 0.0, 1.0)
    spoil_rgb = spoil_cmap(0.15 + 0.8 * depth_from_surface)[..., :3]
    flesh = fresh_rgb[None, None, :] + (fresh_deep - fresh_rgb)[None, None, :] * depth_from_surface[..., None]
    tex = _marble(z_grid.shape[0], z_grid.shape[1], seed, marble)[..., None]
    flesh = flesh * tex                                       # fatty marbling on raw flesh
    spoil_rgb = spoil_rgb * (0.4 * (tex - 1.0) + 1.0)         # fainter texture on spoiled tissue
    rgb = flesh + (spoil_rgb - flesh) * spoiled_frac[..., None]
    ao = 0.80 + 0.20 * np.clip(z_grid / D_MAX, 0.0, 1.0)      # interior darker toward the bottom
    return _shade(rgb * ao[..., None], base_normal)


def render_block(ax, depth_cm: np.ndarray, meat: str | None = None, factor: int = 14,
                 cut: float = 0.5, elev: float = 24, azim: float = -54,
                 show_axes: bool = True) -> None:
    """Render one lit, cut-away spoilage block on a 3D axis. depth_cm is a 5x5 field (cm).
    ``meat`` is accepted for label compatibility but does not alter the colours."""
    spoil_cmap, fresh_rgb, fresh_deep, marble = palette_for(meat)
    depth_cm = np.clip(np.asarray(depth_cm, dtype=float), 0.0, D_MAX)
    hi = np.clip(_upsample(depth_cm, factor), 0.0, D_MAX)
    n = hi.shape[0]
    xs = np.linspace(0, GRID, n)
    ys = np.linspace(0, GRID, n)
    zs = np.linspace(0, D_MAX, max(2, int(factor * 3)))
    y_cut = int(round(cut * (n - 1)))                        # cut away the front (low-y) half

    # ----- top surface (z = D_MAX): exposed spoiled surface, shaded relief + faint texture -----
    Xt, Yt = np.meshgrid(xs, ys[y_cut:], indexing="ij")
    Zt = np.full_like(Xt, D_MAX)
    top_depth = hi[:, y_cut:]
    top_rgb = spoil_cmap(0.30 + 0.6 * np.clip(top_depth / D_MAX, 0, 1))[..., :3]
    top_rgb = top_rgb * (0.4 * (_marble(*top_rgb.shape[:2], 7, marble) - 1.0) + 1.0)[..., None]
    top_shaded = _LS.shade_rgb(top_rgb, elevation=top_depth, blend_mode="soft", vert_exag=2.0)
    ax.plot_surface(Xt, Yt, Zt, facecolors=np.clip(top_shaded, 0, 1), rstride=1, cstride=1,
                    shade=False, antialiased=True, linewidth=0)

    # ----- cut cross-section face (y = cut plane): THE penetration profile along x -----
    Zc = np.tile(zs, (n, 1))                                 # (n, K)
    depth_cut = hi[:, y_cut]
    cut_rgb = _two_tone_face(depth_cut, Zc, 0.32, spoil_cmap, fresh_rgb, fresh_deep, marble, 1)
    Xc, _ = np.meshgrid(xs, zs, indexing="ij")
    Yc = np.full_like(Xc, ys[y_cut])
    ax.plot_surface(Xc, Yc, Zc, facecolors=cut_rgb, rstride=1, cstride=1,
                    shade=False, antialiased=True, linewidth=0)

    # ----- back face (y = max) -----
    depth_back = hi[:, -1]
    back_rgb = _two_tone_face(depth_back, Zc, 0.18, spoil_cmap, fresh_rgb, fresh_deep, marble, 2)
    Yb = np.full_like(Xc, ys[-1])
    ax.plot_surface(Xc, Yb, Zc, facecolors=back_rgb, rstride=1, cstride=1,
                    shade=False, antialiased=True, linewidth=0)

    # ----- left/right faces (x = 0, x = max) over the back half -----
    Zs2 = np.tile(zs, (n - y_cut, 1))
    Yl, _ = np.meshgrid(ys[y_cut:], zs, indexing="ij")
    for si, (x_const, normal) in enumerate(((xs[0], 0.18), (xs[-1], 0.42))):
        depth_side = hi[0 if x_const == xs[0] else -1, y_cut:]
        side_rgb = _two_tone_face(depth_side, Zs2, normal, spoil_cmap, fresh_rgb, fresh_deep, marble, 3 + si)
        Xl = np.full_like(Yl, x_const)
        ax.plot_surface(Xl, Yl, Zs2, facecolors=side_rgb, rstride=1, cstride=1,
                        shade=False, antialiased=True, linewidth=0)

    # ----- bottom (fresh interior, darkest) -----
    Xbo, Ybo = np.meshgrid(xs, ys[y_cut:], indexing="ij")
    bot = _shade(fresh_deep * 0.82, 0.0)
    ax.plot_surface(Xbo, Ybo, np.zeros_like(Xbo),
                    color=tuple(bot), shade=False, antialiased=True, linewidth=0)

    ax.set_box_aspect((GRID, GRID, D_MAX * 1.25))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(0, GRID); ax.set_ylim(0, GRID); ax.set_zlim(0, D_MAX)
    if show_axes:
        ax.set_zticks([0, D_MAX]); ax.set_zticklabels(["3", "0"])  # 0 cm at the surface (top)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_zlabel("depth from surface (cm)", labelpad=-8, fontsize=6)
    else:
        ax.set_axis_off()
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_visible(False)
        pane.line.set_color((1, 1, 1, 0))
    ax.grid(False)


def figure_blocks(fields: list[np.ndarray], titles: list[str], path: Path,
                  ncols: int | None = None, suptitle: str | None = None,
                  row_labels: list[str] | None = None, meats: list[str] | None = None) -> None:
    n = len(fields)
    ncols = ncols or n
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(2.3 * ncols, 2.5 * nrows), dpi=200)
    for i, (field, title) in enumerate(zip(fields, titles)):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        render_block(ax, field, meat=(meats[i] if meats else None), show_axes=(i % ncols == 0))
        ax.set_title(title, fontsize=8, pad=-2)
    if row_labels:
        for r, lab in enumerate(row_labels):
            fig.text(0.01, 1 - (r + 0.5) / nrows, lab, rotation=90, va="center",
                     fontsize=8, fontweight="bold")
    if suptitle:
        fig.suptitle(suptitle, fontsize=10, fontweight="bold")
    fig.subplots_adjust(left=0.04, right=0.99, top=0.92, bottom=0.02, wspace=0.05, hspace=0.12)
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{suffix}"), bbox_inches="tight", pad_inches=0.04,
                    dpi=320 if suffix == "png" else None)
    plt.close(fig)


def make_timelapse(depth_by_hour: np.ndarray, hours: list[int], path: Path,
                   title: str = "", fps: int = 18, interp_per_hour: int = 6,
                   turntable: float = 26.0, meat: str | None = None) -> None:
    """Smooth MP4 of decay penetrating over time.

    depth_by_hour : (T, 5, 5) penetration depth (cm) at each sampled hour
    interp_per_hour: number of interpolated frames between consecutive sampled hours
    turntable      : total azimuth sweep (deg) across the whole clip (0 = fixed camera)
    """
    depth_by_hour = np.asarray(depth_by_hour, dtype=float)
    T = depth_by_hour.shape[0]
    # temporally interpolate the field for fluid growth
    fine_t = np.linspace(0, T - 1, (T - 1) * interp_per_hour + 1)
    base_idx = np.arange(T)
    flat = depth_by_hour.reshape(T, -1)                      # (T, 25)
    fine = np.empty((len(fine_t), flat.shape[1]))
    for c in range(flat.shape[1]):
        fine[:, c] = np.interp(fine_t, base_idx, flat[:, c])
    frames_field = fine.reshape(len(fine_t), GRID, GRID)     # (F, 5, 5)
    fine_hours = np.interp(fine_t, base_idx, np.asarray(hours, dtype=float))

    path.parent.mkdir(parents=True, exist_ok=True)
    encoder = None
    nF = len(fine_t)
    try:
        for k in range(nF):
            fig = plt.figure(figsize=(4.2, 4.4), dpi=150)
            ax = fig.add_subplot(111, projection="3d")
            azim = -54 + turntable * (k / max(nF - 1, 1))
            render_block(ax, frames_field[k], meat=meat, factor=12, elev=24, azim=azim, show_axes=True)
            ax.set_title(f"{title}   t = {fine_hours[k]:4.1f} h", fontsize=10, fontweight="bold")
            fig.subplots_adjust(left=0.0, right=1.0, top=0.95, bottom=0.0)
            fig.canvas.draw()
            buf = np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
            plt.close(fig)

            if encoder is None:
                height, width = buf.shape[:2]
                encoder = subprocess.Popen(
                    [
                        "ffmpeg", "-y", "-loglevel", "error",
                        "-f", "rawvideo", "-pix_fmt", "rgb24",
                        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
                        "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                        str(path),
                    ],
                    stdin=subprocess.PIPE,
                )
            assert encoder.stdin is not None
            encoder.stdin.write(buf.tobytes())
    finally:
        if encoder is not None:
            if encoder.stdin is not None:
                encoder.stdin.close()
            return_code = encoder.wait()
            if return_code != 0:
                raise RuntimeError(f"FFmpeg failed with exit code {return_code}: {path}")


# --------------------------------------------------------------------------------------
def _demo() -> None:
    """Render all meat types with the shared two-colour material at 25 C, hour 28."""
    from spoilage_data import DATA_DIR, load_contexts
    contexts, _ = load_contexts(DATA_DIR, set())
    cmap = {(c.meat, c.temp_label): c for c in contexts}
    meats = ["Beef", "Chicken", "Duck", "Fish", "Pork", "Shrimp"]
    fields, titles = [], []
    for meat in meats:
        c = cmap[(meat, "25C")]
        hi = list(c.hours).index(28)
        fields.append(np.asarray(c.tensor)[hi, :, :, 0])
        titles.append(meat)
    figure_blocks(fields, titles, PROJECT / "figures" / "kinetic_flow" / "meat_colour_blocks",
                  suptitle="Unified-colour spoilage blocks (25 °C, 28 h)", meats=meats)
    print("wrote meat_colour_blocks.pdf/png")


if __name__ == "__main__":
    _demo()
