#!/usr/bin/env python3
"""
Build the OD mode-share tensor  S[i, j, k]  from flows_od_by_mode.csv.gz.

    S[i, j, k] = trips(i -> j by mode k) / sum_k' trips(i -> j by mode k')

so S[i, j, :] sums to 1 for every OD pair that has any trips.

Axes
----
  i, j : dongs, in the row order of nodes_dongs.csv (N = 427)
  k    : modes, in the order given by --modes (default: 4..9, see below)

Why not all 9 KT modes?
-----------------------
Modes 1 (air), 2 (train) and 3 (express bus) carry ~11k trips/week out of 152M
(and 1 and 3 are 100% intra-dong with zero distance). Keeping them only adds
three almost-empty columns, so the default keeps modes 4-9
(wide-area bus, local bus, subway, walk, car, other). Use --all-modes to keep all 9,
or --modes 5,6,7,8 for a custom subset (shares are re-normalised over the kept modes).

Output (.npz)
-------------
  shares       (N, N, m)  mode fractions, sums to 1 over the last axis
  trips        (N, N, m)  the underlying trip counts (same weight as --weight)
  total_trips  (N, N)     trips summed over the kept modes
  has_data     (N, N)     True where the OD pair has at least one trip
  dong_codes   (N,)       8-digit dong codes (str)
  dong_names   (N,)       English dong names
  mode_codes   (m,)       KT mode codes
  mode_names   (m,)       mode labels

Usage
-----
    python build_od_mode_share.py --data-dir data --out od_mode_share.npz

    d = np.load("od_mode_share.npz", allow_pickle=True)
    S = d["shares"]                        # (427, 427, 6)
    S[0, 1, :]                             # mode split for dong 0 -> dong 1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_mode_names(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    code_col = next((c for c in df.columns if pd.api.types.is_integer_dtype(df[c])), None)
    name_cols = [c for c in df.columns if c != code_col and not pd.api.types.is_numeric_dtype(df[c])]
    if code_col is None or not name_cols:
        return {}
    return {int(c): str(n) for c, n in zip(df[code_col], df[name_cols[0]])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("od_mode_share.npz"))
    ap.add_argument("--modes", type=str, default="4,5,6,7,8,9",
                    help="comma-separated KT mode codes to keep, in the order of the last axis")
    ap.add_argument("--all-modes", action="store_true", help="keep all modes 1-9")
    ap.add_argument("--weight", default="trips_week",
                    choices=["trips_week", "trips_weekday_mean", "trips_weekend_mean"],
                    help="which flow column to aggregate (default: whole week)")
    ap.add_argument("--fill", default="nan", choices=["nan", "zero", "citywide"],
                    help="what to put in OD pairs with no trips: NaN (default), zeros, "
                         "or the city-wide mode split (so every cell sums to 1)")
    args = ap.parse_args()

    # ---- dongs define the index order
    dongs = pd.read_csv(args.data_dir / "nodes_dongs.csv", dtype={"dong_code": str})
    codes = dongs["dong_code"].tolist()
    names = dongs["name"].tolist()
    idx = {c: i for i, c in enumerate(codes)}
    N = len(codes)

    # ---- modes
    mode_names_all = load_mode_names(args.data_dir / "mode_codes.csv")
    modes = list(range(1, 10)) if args.all_modes else [int(x) for x in args.modes.split(",")]
    kidx = {m: k for k, m in enumerate(modes)}
    m = len(modes)

    # ---- flows -> trips tensor
    flows = pd.read_csv(args.data_dir / "flows_od_by_mode.csv.gz",
                        dtype={"origin": str, "destination": str})
    total_all = flows[args.weight].sum()
    flows = flows[flows["mode"].isin(modes)]
    kept = flows[args.weight].sum()

    o = flows["origin"].map(idx)
    d = flows["destination"].map(idx)
    bad = o.isna() | d.isna()
    if bad.any():
        print(f"WARNING: dropping {bad.sum()} rows whose dong code is not in nodes_dongs.csv")
        flows, o, d = flows[~bad], o[~bad], d[~bad]

    trips = np.zeros((N, N, m))
    np.add.at(trips, (o.astype(int).to_numpy(), d.astype(int).to_numpy(),
                      flows["mode"].map(kidx).to_numpy()), flows[args.weight].to_numpy())

    # ---- shares
    total = trips.sum(axis=2)                       # (N, N)
    has_data = total > 0
    shares = np.full_like(trips, np.nan)
    np.divide(trips, total[..., None], out=shares, where=has_data[..., None])

    citywide = trips.sum(axis=(0, 1)) / trips.sum()
    if args.fill == "zero":
        shares[~has_data] = 0.0
    elif args.fill == "citywide":
        shares[~has_data] = citywide

    # ---- report
    mnames = [mode_names_all.get(c, f"mode {c}") for c in modes]
    print(f"tensor shape            : {shares.shape}  (origin, destination, mode)")
    print(f"modes (last axis)       : {dict(zip(modes, mnames))}")
    print(f"weight column           : {args.weight}")
    print(f"trips kept              : {100 * kept / total_all:.3f}% of all trips")
    print(f"OD pairs with trips     : {has_data.sum():,} of {N * N:,} ({100 * has_data.mean():.1f}%)")
    print(f"empty OD pairs filled by: {args.fill}")
    s = np.nansum(shares[has_data], axis=1)
    print(f"max |sum(shares) - 1|   : {np.abs(s - 1).max():.2e}  (over pairs with trips)")
    print("\ncity-wide mode split:")
    for n, v in zip(mnames, citywide):
        print(f"  {n:<14s}{100 * v:6.2f}%")
    diag = trips[np.arange(N), np.arange(N)].sum(axis=0)
    print("\nintra-dong (diagonal) split:")
    for n, v in zip(mnames, diag / diag.sum()):
        print(f"  {n:<14s}{100 * v:6.2f}%")

    i, j = np.unravel_index(np.argmax(np.where(np.eye(N, dtype=bool), 0, total)), total.shape)
    print(f"\nbusiest inter-dong pair: {names[i]} -> {names[j]}  ({total[i, j]:,.0f} trips)")
    for n, v in zip(mnames, shares[i, j]):
        print(f"  {n:<14s}{100 * v:6.2f}%")

    # ---- save
    np.savez_compressed(
        args.out,
        shares=shares, trips=trips, total_trips=total, has_data=has_data,
        dong_codes=np.array(codes), dong_names=np.array(names),
        mode_codes=np.array(modes), mode_names=np.array(mnames),
    )
    print(f"\nsaved {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
