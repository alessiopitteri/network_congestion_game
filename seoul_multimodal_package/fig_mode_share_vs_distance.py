#!/usr/bin/env python3
"""
Mode share vs OD distance, computed from the OD mode-share tensor built by
build_od_mode_share.py.

Input
-----
  --npz    od_mode_share_car_bus_subway_walk.npz  (uses: trips, dong_codes, mode_codes, mode_names)
  --dongs  data/nodes_dongs.csv                    (lon/lat of the dong representative points)

Usage
-----
    python fig_mode_share_vs_distance.py --npz od_mode_share_car_bus_subway_walk.npz \
        --dongs data/nodes_dongs.csv --out-dir new_figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BIN_EDGES_KM = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, np.inf]
COLORS = {4: "#8172b3", 5: "#2a9d8f", 6: "#4c72b0", 7: "#e9c46a", 8: "#c1121f", 9: "#adb5bd"}


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))


def bin_labels() -> list[str]:
    labs = ["intra-dong"]
    for a, b in zip(BIN_EDGES_KM[:-1], BIN_EDGES_KM[1:]):
        labs.append(f"{a:g}-{b:g}" if np.isfinite(b) else f">{a:g}")
    return labs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", type=Path, default=Path("od_mode_share_car_bus_subway_walk.npz"))
    ap.add_argument("--dongs", type=Path, default=Path("data/nodes_dongs.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("new_figures"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load tensor
    z = np.load(args.npz, allow_pickle=True)
    trips = z["trips"]                                   # (N, N, m)
    codes = [str(c) for c in z["dong_codes"]]
    mode_codes = [int(c) for c in z["mode_codes"]]
    mode_names = [str(n) for n in z["mode_names"]]
    N, _, m = trips.shape
    print(f"loaded {args.npz.name}: N={N}, modes={dict(zip(mode_codes, mode_names))}")

    # ---- centroid distance matrix, aligned to the tensor's dong order
    dg = pd.read_csv(args.dongs, dtype={"dong_code": str}).set_index("dong_code").reindex(codes)
    if dg[["lon", "lat"]].isna().any().any():
        raise SystemExit("Some dong_codes in the .npz are missing from nodes_dongs.csv")
    lon, lat = dg["lon"].to_numpy(), dg["lat"].to_numpy()
    D = haversine_km(lon[:, None], lat[:, None], lon[None, :], lat[None, :])   # (N, N) km

    # ---- bin matrix: 0 = intra-dong, 1.. = km ranges
    B = np.digitize(D, BIN_EDGES_KM[1:-1], right=False) + 1
    B[np.arange(N), np.arange(N)] = 0
    labels = bin_labels()
    nb = len(labels)

    tab = np.stack([np.bincount(B.ravel(), weights=trips[:, :, k].ravel(), minlength=nb)
                    for k in range(m)], axis=1)          # (nb, m) trips
    vol = tab.sum(axis=1)
    share = np.divide(tab, vol[:, None], out=np.full_like(tab, np.nan), where=vol[:, None] > 0)

    # ---- table
    df = pd.DataFrame(share, index=labels, columns=mode_names)
    df["trips"] = vol
    df["n_od_pairs"] = np.bincount(B.ravel(), minlength=nb)
    df["n_od_pairs_with_trips"] = np.bincount(B.ravel(), weights=(trips.sum(2) > 0).ravel(), minlength=nb)
    show = df.copy()
    for c in mode_names:
        show[c] = (100 * show[c]).round(1)
    show["trips"] = show["trips"].round(0)
    print("\nshare of trips per mode (%) in each distance bin (km):\n")
    print(show.to_string())
    csv = args.out_dir / "mode_share_by_distance_bin.csv"
    df.to_csv(csv)
    print(f"\nsaved {csv}")

    # ---- figure
    cols = [COLORS.get(c, "#555555") for c in mode_codes]
    x = np.arange(nb)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios": [1.2, 1.2, 0.8]})

    # Panel 0: Stacked Bars
    bottom = np.zeros(nb)
    for k, (name, col) in enumerate(zip(mode_names, cols)):
        v = np.nan_to_num(share[:, k])
        ax[0].bar(x, v, bottom=bottom, color=col, label=name, width=0.85)
        bottom += v
    ax[0].set_xticks(x, labels, rotation=45, ha="right")
    ax[0].set_ylim(0, 1)
    ax[0].set_ylabel("Share of Trips in Bin", fontsize=10)
    ax[0].set_xlabel("OD Distance (km)", fontsize=10)
    ax[0].set_title("Mode Share", fontsize=11, fontweight="bold", pad=10)
    ax[0].legend(ncol=2, fontsize=8, loc="upper right", framealpha=0.9)

    # Panel 1: Line Chart
    for k, (name, col) in enumerate(zip(mode_names, cols)):
        ax[1].plot(x, share[:, k], marker="o", color=col, label=name)
    ax[1].set_xticks(x, labels, rotation=45, ha="right")
    ax[1].set_ylim(0, 1)
    ax[1].set_ylabel("Share of Trips in Bin", fontsize=10)
    ax[1].set_xlabel("OD Distance (km)", fontsize=10)
    ax[1].set_title("Mode Share by Distance", fontsize=11, fontweight="bold", pad=10)
    ax[1].grid(alpha=0.3, linestyle="--")

    # Panel 2: Volume Histogram
    ax[2].bar(x, vol / 1e6, color="slategray", width=0.85)
    ax[2].set_xticks(x, labels, rotation=45, ha="right")
    ax[2].set_ylabel("Trips (Millions)", fontsize=10)
    ax[2].set_xlabel("OD Distance (km)", fontsize=10)
    ax[2].set_title("Total Trip Volume per Bin", fontsize=11, fontweight="bold", pad=10)

    '''# Main Figure Title (Suptitle)
    fig.suptitle(
        rf"$\mathbf{{P(\text{{Mode}} \mid \text{{Distance}})}}$ — {args.npz.name}",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )'''

    plt.tight_layout()
    path = args.out_dir / "11_mode_share_vs_distance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved {path}")


if __name__ == "__main__":
    main()