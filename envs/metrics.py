# envs/metrics.py
"""
Lightweight metrics for the 3D bin-packing simulation.

Compute and save a JSON metrics report for one run with a single call:
    from envs.metrics import save_run_metrics
    save_run_metrics(bin_dims, placed_boxes, snapshot_dir=env.snapshot_dir)

Metrics included:
- Volume Fill Ratio (VFR), packed volume
- Height utilization
- Voxelized void metrics: void volume ratio, # of empty components, largest void fraction
- Layer fill ratios per z-layer (z=0..D-1)
- Support/contact metrics: # on floor / on box, contact area ratios, stack depth
- Counts summary (boxes, grid size)

No heavy dependencies; only numpy.
"""

from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np


# ----------------------------
# Core helpers
# ----------------------------

def _to_tuples(placed_boxes: List[Dict]) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    """Normalize placed_boxes into [((x,y,z), (w,h,d)), ...] tuples."""
    out = []
    for b in placed_boxes:
        pos = tuple(int(round(v)) for v in b["position"])
        siz = tuple(int(round(v)) for v in b["size"])
        out.append((pos, siz))
    return out

def _bin_volume(bin_dims: List[int]) -> int:
    W, H, D = bin_dims
    return int(W) * int(H) * int(D)

def volume_fill_ratio(bin_dims: List[int], placed_boxes: List[Dict]) -> Tuple[float, int]:
    """Return (VFR, packed_volume)."""
    tboxes = _to_tuples(placed_boxes)
    packed = sum(w*h*d for (_pos, (w, h, d)) in tboxes)
    return (packed / _bin_volume(bin_dims) if _bin_volume(bin_dims) > 0 else 0.0, packed)

def height_utilization(bin_dims: List[int], placed_boxes: List[Dict]) -> Tuple[float, int]:
    """Return (HU, max_height_used)."""
    if not placed_boxes:
        return (0.0, 0)
    tboxes = _to_tuples(placed_boxes)
    zmax = max(z + d for ((x, y, z), (w, h, d)) in tboxes)
    D = bin_dims[2]
    return (min(1.0, zmax / D) if D > 0 else 0.0, zmax)

# ----------------------------
# Voxelization & void metrics
# ----------------------------

def _voxelize(bin_dims: List[int], placed_boxes: List[Dict], voxel: int = 1) -> np.ndarray:
    """
    Voxelize the bin at resolution 'voxel' (must divide bin dims).
    Returns a boolean occ grid of shape (D', H', W') where occ[z,y,x] == True if occupied.
    """
    W, H, D = map(int, bin_dims)
    assert W % voxel == 0 and H % voxel == 0 and D % voxel == 0, "voxel must divide bin dims"
    gw, gh, gd = W // voxel, H // voxel, D // voxel

    occ = np.zeros((gd, gh, gw), dtype=bool)
    for pos, siz in _to_tuples(placed_boxes):
        x, y, z = pos
        w, h, d = siz
        xs = x // voxel
        ys = y // voxel
        zs = z // voxel
        xe = (x + w + voxel - 1) // voxel  # ceil-div for inclusive coverage
        ye = (y + h + voxel - 1) // voxel
        ze = (z + d + voxel - 1) // voxel
        xs, ys, zs = max(0, xs), max(0, ys), max(0, zs)
        xe, ye, ze = min(gw, xe), min(gh, ye), min(gd, ze)
        if xs < xe and ys < ye and zs < ze:
            occ[zs:ze, ys:ye, xs:xe] = True
    return occ

def _connected_components_3d(empty: np.ndarray) -> List[int]:
    """
    Return sizes of connected components in a 3D boolean array (True=empty),
    using 6-connectivity. No scipy needed.
    """
    Z, Y, X = empty.shape
    visited = np.zeros_like(empty, dtype=bool)
    sizes = []

    # neighbor deltas (6-connectivity)
    nbrs = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                if not empty[z, y, x] or visited[z, y, x]:
                    continue
                # BFS/DFS
                stack = [(z, y, x)]
                visited[z, y, x] = True
                size = 0
                while stack:
                    cz, cy, cx = stack.pop()
                    size += 1
                    for dz, dy, dx in nbrs:
                        nz, ny, nx = cz + dz, cy + dy, cx + dx
                        if 0 <= nz < Z and 0 <= ny < Y and 0 <= nx < X:
                            if empty[nz, ny, nx] and not visited[nz, ny, nx]:
                                visited[nz, ny, nx] = True
                                stack.append((nz, ny, nx))
                sizes.append(size)
    return sizes

def voxel_void_metrics(bin_dims: List[int], placed_boxes: List[Dict], voxel: int = 1) -> Dict:
    """
    Compute void metrics on a voxel grid:
      - void_volume_ratio
      - components_count
      - largest_void_fraction
      - grid_shape
    """
    occ = _voxelize(bin_dims, placed_boxes, voxel=voxel)
    Z, Y, X = occ.shape
    total_vox = int(Z * Y * X)
    empty = ~occ
    empty_vox = int(empty.sum())

    comp_sizes = _connected_components_3d(empty)
    components_count = len(comp_sizes)
    largest = max(comp_sizes) if comp_sizes else 0

    return {
        "voxel_size": voxel,
        "grid_shape": [int(X), int(Y), int(Z)],
        "empty_voxels": empty_vox,
        "total_voxels": total_vox,
        "void_volume_ratio": (empty_vox / total_vox) if total_vox > 0 else 0.0,
        "components_count": components_count,
        "largest_void_fraction": (largest / total_vox) if total_vox > 0 else 0.0,
    }

def layer_fill_ratios(bin_dims: List[int], placed_boxes: List[Dict], voxel: int = 1) -> List[float]:
    """
    For each z-layer in the voxel grid, compute 2D fill ratio.
    Returns list of length D' with values in [0,1].
    """
    occ = _voxelize(bin_dims, placed_boxes, voxel=voxel)
    Z, Y, X = occ.shape
    layer_ratios = []
    denom = float(X * Y) if X * Y > 0 else 1.0
    for z in range(Z):
        layer_ratios.append(float(occ[z].sum()) / denom)
    return layer_ratios

# ----------------------------
# Support/contact & stacking
# ----------------------------

def _rect_overlap_area(ax, ay, aw, ah, bx, by, bw, bh) -> int:
    """Overlap area between two axis-aligned 2D rectangles."""
    ox = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    oy = max(0, min(ay + ah, by + bh) - max(ay, by))
    return ox * oy

def _supporting_boxes_for(target_pos, placed_tuples):
    """Yield boxes that directly support target_pos (top_z == target z and XY overlap)."""
    nx, ny, nz = target_pos
    for (px, py, pz), (pw, ph, pd) in placed_tuples:
        if pz + pd == nz:
            # some XY overlap?
            if not (nx + 0 <= px or nx >= px + pw) and not (ny + 0 <= py or ny >= py + ph):
                yield (px, py, pz), (pw, ph, pd)

def contact_metrics(bin_dims: List[int], placed_boxes: List[Dict]) -> Dict:
    """
    Contact/support metrics:
      - floor_count, on_box_count
      - avg_contact_ratio (for on-box), min_contact_ratio
      - max_stack_depth, avg_stack_depth
    """
    if not placed_boxes:
        return {
            "floor_count": 0,
            "on_box_count": 0,
            "avg_contact_ratio": 0.0,
            "min_contact_ratio": 0.0,
            "max_stack_depth": 0,
            "avg_stack_depth": 0.0,
        }

    tboxes = _to_tuples(placed_boxes)

    # Compute support level (stack depth): floor = 1; on top = 1 + max(level of supports)
    levels = {}
    # Sort by z ascending ensures supports are processed before dependents
    order = sorted(range(len(tboxes)), key=lambda i: tboxes[i][0][2])

    contact_ratios = []
    floor_count = 0
    on_box_count = 0

    for idx in order:
        (x, y, z), (w, h, d) = tboxes[idx]
        footprint = w * h

        if z == 0:
            levels[idx] = 1
            floor_count += 1
            continue

        # find supports (top_z == z and XY overlap)
        supports = []
        for j in range(idx):  # only earlier (lower) boxes can support
            (px, py, pz), (pw, ph, pd) = tboxes[j]
            if pz + pd != z:
                continue
            # overlap in XY?
            if not (x + w <= px or x >= px + pw) and not (y + h <= py or y >= py + ph):
                supports.append(j)

        if not supports:
            # Shouldn't happen under your is_supported(), but handle gracefully
            levels[idx] = 1
            floor_count += 1
            # zero contact if we think it's unsupported
            continue

        # contact area with union of supports (approx: sum clipped, capped at footprint)
        contact_area = 0
        for j in supports:
            (px, py, _), (pw, ph, _) = tboxes[j]
            contact_area += _rect_overlap_area(x, y, w, h, px, py, pw, ph)
        contact_area = min(contact_area, footprint)
        ratio = (contact_area / footprint) if footprint > 0 else 0.0
        contact_ratios.append(ratio)

        on_box_count += 1
        levels[idx] = 1 + max(levels.get(j, 1) for j in supports)

    max_depth = max(levels.values()) if levels else 0
    avg_depth = (sum(levels.values()) / len(levels)) if levels else 0.0

    avg_contact = (sum(contact_ratios) / len(contact_ratios)) if contact_ratios else 0.0
    min_contact = min(contact_ratios) if contact_ratios else 0.0

    return {
        "floor_count": int(floor_count),
        "on_box_count": int(on_box_count),
        "avg_contact_ratio": float(avg_contact),
        "min_contact_ratio": float(min_contact),
        "max_stack_depth": int(max_depth),
        "avg_stack_depth": float(avg_depth),
    }

# ----------------------------
# Orchestrator: compute & save
# ----------------------------

def compute_all_metrics(
    bin_dims: List[int],
    placed_boxes: List[Dict],
    *,
    voxel: int = 1,
    include_layers: bool = True,
    extra_stats: Optional[Dict] = None,
) -> Dict:
    """
    Compute all metrics. 'extra_stats' lets you include behavioral counters
    (e.g., retries, invalid rates) if you track them elsewhere in main.py.
    """
    vfr, packed_vol = volume_fill_ratio(bin_dims, placed_boxes)
    hu, max_h = height_utilization(bin_dims, placed_boxes)
    voids = voxel_void_metrics(bin_dims, placed_boxes, voxel=voxel)
    support = contact_metrics(bin_dims, placed_boxes)
    layers = layer_fill_ratios(bin_dims, placed_boxes, voxel=voxel) if include_layers else None

    W, H, D = bin_dims
    out = {
        "bin": {"width": int(W), "height": int(H), "depth": int(D), "volume": int(_bin_volume(bin_dims))},
        "counts": {"boxes": int(len(placed_boxes))},
        "volumes": {
            "packed_volume": int(packed_vol),
            "volume_fill_ratio": float(vfr),
        },
        "height": {
            "max_height_used": int(max_h),
            "height_utilization": float(hu),
        },
        "voids": voids,
        "support": support,
    }
    if layers is not None:
        out["layers"] = {"voxel_size": int(voxel), "fill_per_layer": [float(v) for v in layers]}

    if extra_stats:
        out["extra"] = extra_stats  # user-supplied behavioral stats

    return out

def _timestamp_from_snapshot_dir(snapshot_dir: Optional[str]) -> str:
    """
    If snapshot_dir looks like 'snapshots/<timestamp>', extract <timestamp>.
    Otherwise return now().
    """
    if snapshot_dir and os.path.isdir(snapshot_dir):
        base = os.path.basename(os.path.normpath(snapshot_dir))
        if base and len(base) >= 13 and "_" in base:
            return base
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_run_metrics(
    bin_dims: List[int],
    placed_boxes: List[Dict],
    *,
    results_root: str = "results",
    snapshot_dir: Optional[str] = None,
    voxel: int = 1,
    include_layers: bool = True,
    extra_stats: Optional[Dict] = None,
) -> str:
    """
    Compute all metrics and save to results/<timestamp>/metrics.json.
    - If snapshot_dir is given, reuse its timestamp to align runs.
    - Returns the path to the saved JSON file.
    """
    ts = _timestamp_from_snapshot_dir(snapshot_dir)
    out_dir = os.path.join(results_root, ts)
    os.makedirs(out_dir, exist_ok=True)

    report = compute_all_metrics(
        bin_dims,
        placed_boxes,
        voxel=voxel,
        include_layers=include_layers,
        extra_stats=extra_stats,
    )

    out_path = os.path.join(out_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    return out_path
