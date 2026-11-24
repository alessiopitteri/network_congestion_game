#!/usr/bin/env python3
"""
Explore and visualise the Seoul multimodal package described in README.md.

Usage
-----
    python explore_seoul_multimodal.py --data-dir /path/to/package --out-dir figures

Requires: pandas, numpy, matplotlib, geopandas (pyogrio or fiona backend).
Optional: --skip-walk to skip the heavy pedestrian layer (274k edges).

What it does
------------
  1. Prints a summary of every file (shape, columns, key statistics).
  2. Saves figures (PNG) to --out-dir:
       01_mode_share.png            weekly trips by mode, weekday vs weekend
       02_distance_time_by_mode.png trip-weighted distance / time distributions
       03_flow_choropleths.png      outflow, inflow and net flow per dong
       04_top_od_pairs.png          largest inter-dong flows
       05_network_overview.png      dongs + rail + bus + major roads
       06_rail_bus_service.png      trains/day per rail line, bus frequency map
       07_topis_speeds.png          road speeds by hour of day
       08_routed_times.png          routed travel-time maps from a chosen dong
       09_kt_vs_routed_time.png     KT observed time vs R5 routed time
       10_walk_network.png          pedestrian network by highway type
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import geopandas as gpd
except ImportError:  # pragma: no cover
    gpd = None


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def find_col(df: pd.DataFrame, *keywords: str, required: bool = False):
    """Return first column whose lower-cased name contains all keywords."""
    for c in df.columns:
        lc = c.lower()
        if all(k in lc for k in keywords):
            return c
    if required:
        raise KeyError(f"No column containing {keywords} in {list(df.columns)}")
    return None


def banner(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def describe_df(name: str, df: pd.DataFrame, n_cols: int = 40) -> None:
    print(f"\n[{name}] {len(df):,} rows x {df.shape[1]} cols")
    print("  columns:", ", ".join(map(str, df.columns[:n_cols])),
          "..." if df.shape[1] > n_cols else "")


def save(fig, out: Path, name: str) -> None:
    path = out / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def read_csv_safe(path: Path, **kw) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  (missing: {path.name})")
        return None
    return pd.read_csv(path, **kw)


def read_layer(gpkg: Path, layer: str):
    if gpd is None or not gpkg.exists():
        return None
    try:
        return gpd.read_file(gpkg, layer=layer)
    except Exception as e:  # layer missing / driver problem
        print(f"  (cannot read layer '{layer}': {e})")
        return None


def mode_lookup(modes: pd.DataFrame | None) -> dict[int, str]:
    """code -> label from mode_codes.csv (column names are auto-detected)."""
    fallback = {i: f"mode {i}" for i in range(1, 10)}
    if modes is None or modes.empty:
        return fallback
    code_col = next((c for c in modes.columns if pd.api.types.is_integer_dtype(modes[c])), None)
    label_cols = [c for c in modes.columns
                  if c != code_col and not pd.api.types.is_numeric_dtype(modes[c])]
    if code_col is None or not label_cols:
        return fallback
    # prefer an English label column
    label_col = next((c for c in label_cols if any(k in c.lower() for k in ("en", "name", "label", "mode"))),
                     label_cols[0])
    return {int(r[code_col]): str(r[label_col]) for _, r in modes.iterrows()}


def code_for(labels: dict[int, str], *keywords: str):
    for k, v in labels.items():
        if all(w in v.lower() for w in keywords):
            return k
    return None


def choropleth(ax, gdf, col, title, cmap="viridis", **kw):
    gdf.plot(column=col, ax=ax, cmap=cmap, legend=True, linewidth=0.1,
             edgecolor="white", legend_kwds={"shrink": 0.6}, **kw)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("."), help="folder with the package files")
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    ap.add_argument("--skip-walk", action="store_true", help="skip the pedestrian layer")
    ap.add_argument("--focus-dong", type=str, default=None,
                    help="8-digit dong code for the routed-time maps (default: busiest destination)")
    args = ap.parse_args()

    d, out = args.data_dir, args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    gpkg = d / "seoul_multimodal_network.gpkg"
    plt.rcParams.update({"font.size": 9, "axes.grid": False})

    # ------------------------------------------------------------------ nodes
    banner("NODES")
    dongs_tbl = read_csv_safe(d / "nodes_dongs.csv", dtype=str)
    rail_stops = read_csv_safe(d / "nodes_rail_stops.csv", dtype={"dong_code": str})
    bus_stops = read_csv_safe(d / "nodes_bus_stops.csv", dtype={"dong_code": str})
    for nm, df in [("nodes_dongs", dongs_tbl), ("nodes_rail_stops", rail_stops), ("nodes_bus_stops", bus_stops)]:
        if df is not None:
            describe_df(nm, df)

    dong_code_col = name_col = gu_col = None
    if dongs_tbl is not None:
        dong_code_col = find_col(dongs_tbl, "dong", "code") or dongs_tbl.columns[0]
        name_col = next((c for c in dongs_tbl.columns if c.lower() in ("dong_name", "name", "dong_name_en")),
                        find_col(dongs_tbl, "name"))
        gu_col = find_col(dongs_tbl, "gu", "name")
        print(f"\n  dongs: {dongs_tbl[dong_code_col].nunique()} | "
              f"gus: {dongs_tbl[gu_col].nunique() if gu_col else 'n/a'}")
        ns = find_col(dongs_tbl, "name_source")
        if ns:
            print("  name sources:", dongs_tbl[ns].value_counts().to_dict())
    if rail_stops is not None:
        g = find_col(rail_stops, "station_group")
        print(f"  rail platforms: {len(rail_stops)}"
              + (f" | distinct stations: {rail_stops[g].nunique()}" if g else ""))
    if bus_stops is not None:
        print(f"  bus stops: {len(bus_stops):,}")
        if "dong_code" in bus_stops:
            top = bus_stops["dong_code"].value_counts().head(5)
            print("  dongs with most bus stops:", top.to_dict())

    dong_names = {}
    if dongs_tbl is not None and name_col:
        dong_names = dict(zip(dongs_tbl[dong_code_col].astype(str), dongs_tbl[name_col]))

    dongs_gdf = read_layer(gpkg, "dongs")
    if dongs_gdf is not None:
        dc = find_col(dongs_gdf, "dong", "code") or dong_code_col
        dongs_gdf[dc] = dongs_gdf[dc].astype(str)
        dongs_gdf = dongs_gdf.rename(columns={dc: "dong_code"})
        print(f"\n  dongs layer CRS: {dongs_gdf.crs}")

    # ------------------------------------------------------------------ flows
    banner("FLOWS (KT OD x mode)")
    flows = read_csv_safe(d / "flows_od_by_mode.csv.gz", dtype={"origin": str, "destination": str})
    modes_tbl = read_csv_safe(d / "mode_codes.csv")
    labels = mode_lookup(modes_tbl)
    if modes_tbl is not None:
        print("\nmode_codes.csv:\n", modes_tbl.to_string(index=False))

    if flows is not None:
        describe_df("flows", flows)
        flows["mode_label"] = flows["mode"].map(labels).fillna(flows["mode"].astype(str))
        total = flows["trips_week"].sum()
        print(f"\n  total weekly trips: {total/1e6:,.1f} M | OD rows: {len(flows):,}")

        by_mode = flows.groupby("mode_label").agg(
            trips_week=("trips_week", "sum"),
            weekday_mean=("trips_weekday_mean", "sum"),
            weekend_mean=("trips_weekend_mean", "sum"),
        )
        by_mode["share_%"] = 100 * by_mode["trips_week"] / total
        by_mode["weekend/weekday"] = by_mode["weekend_mean"] / by_mode["weekday_mean"]
        # trip-weighted dist/time per mode
        w = flows["trips_week"]
        for c in ("mean_dist_m", "mean_time_min"):
            by_mode[c] = (flows[c] * w).groupby(flows["mode_label"]).sum() / by_mode["trips_week"]
        # NB: KT time includes access, waiting and detection lag, so this is NOT a vehicle speed
        by_mode["dist/time_kmh(not real speed)"] = (
            (by_mode["mean_dist_m"] / 1000) / (by_mode["mean_time_min"] / 60)
        ).replace([np.inf, -np.inf], np.nan)
        by_mode = by_mode.sort_values("trips_week", ascending=False)
        print("\n", by_mode.round(2).to_string())

        intra = flows.groupby("intra_dong")["trips_week"].sum()
        print("\n  intra-dong share of trips: "
              f"{100 * intra.get(True, 0) / total:.1f}%")
        print("  intra-dong share by mode (%):")
        print((100 * flows[flows.intra_dong].groupby("mode_label")["trips_week"].sum()
               / flows.groupby("mode_label")["trips_week"].sum()).round(1).to_string())

        # --- Fig 1: mode share + weekday/weekend
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        by_mode["trips_week"].div(1e6).plot.barh(ax=ax[0], color="steelblue")
        ax[0].invert_yaxis()
        ax[0].set_xlabel("trips per week (millions)")
        ax[0].set_title("Weekly trips by mode")
        for i, (v, s) in enumerate(zip(by_mode["trips_week"] / 1e6, by_mode["share_%"])):
            ax[0].text(v, i, f" {s:.1f}%", va="center")
        by_mode[["weekday_mean", "weekend_mean"]].div(1e6).plot.bar(ax=ax[1], color=["#4c72b0", "#dd8452"])
        ax[1].set_ylabel("mean trips per day (millions)")
        ax[1].set_title("Average weekday vs weekend day")
        ax[1].tick_params(axis="x", rotation=30)
        save(fig, out, "01_mode_share.png")

        # --- Fig 2: distance and time distribution per mode
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for m, grp in flows[~flows.intra_dong].groupby("mode_label"):
            if grp["trips_week"].sum() < 0.005 * total:
                continue
            for a, col, xl in ((ax[0], "mean_dist_m", "km"), (ax[1], "mean_time_min", "min")):
                x = grp[col] / (1000 if col == "mean_dist_m" else 1)
                h, edges = np.histogram(x.clip(upper=x.quantile(0.99)), bins=50,
                                        weights=grp["trips_week"])
                a.plot(0.5 * (edges[1:] + edges[:-1]), h / h.sum(), label=m)
                a.set_xlabel(f"trip-weighted mean {col.split('_')[1]} per OD pair ({xl})")
        ax[0].set_title("Distance distribution (inter-dong OD pairs)")
        ax[1].set_title("KT travel-time distribution")
        ax[0].set_ylabel("share of trips")
        ax[0].legend()
        save(fig, out, "02_distance_time_by_mode.png")

        # --- per-dong in/out flows
        inter = flows[~flows.intra_dong]
        outflow = inter.groupby("origin")["trips_week"].sum().rename("outflow")
        inflow = inter.groupby("destination")["trips_week"].sum().rename("inflow")
        per_dong = pd.concat([outflow, inflow], axis=1).fillna(0)
        per_dong["net"] = per_dong["inflow"] - per_dong["outflow"]
        per_dong["name"] = per_dong.index.map(dong_names)
        print("\n  top 10 destination dongs (weekly inflow, excl. intra-dong):")
        print((per_dong.sort_values("inflow", ascending=False).head(10)[["name", "inflow"]]
               .assign(inflow=lambda x: (x.inflow / 1e3).round(0))
               .rename(columns={"inflow": "inflow (thousand)"})).to_string())
        print("\n  net flow = inflow - outflow (thousand trips/week); 5 most negative (net emitters), "
              "then 5 most positive (net attractors):")
        print((per_dong.sort_values("net")["net"] / 1e3).round(0).iloc[[*range(5), *range(-5, 0)]]
              .rename(lambda k: f"{k} {dong_names.get(k, '')}").to_string())

        if dongs_gdf is not None:
            g = dongs_gdf.merge(per_dong.reset_index(names="dong_code"), on="dong_code", how="left")
            fig, ax = plt.subplots(1, 3, figsize=(16, 5))
            choropleth(ax[0], g, "outflow", "Outflow (trips/week)", "YlOrRd")
            choropleth(ax[1], g, "inflow", "Inflow (trips/week)", "YlGnBu")
            lim = np.nanpercentile(np.abs(g["net"]), 95)
            choropleth(ax[2], g, "net", "Net flow (inflow - outflow)", "RdBu",
                       vmin=-lim, vmax=lim)
            save(fig, out, "03_flow_choropleths.png")

        # --- top OD pairs
        top = (inter.groupby(["origin", "destination"])["trips_week"].sum()
               .sort_values(ascending=False).head(15))
        labels_od = [f"{dong_names.get(o, o)} → {dong_names.get(t, t)}" for o, t in top.index]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(labels_od[::-1], (top.values / 1e3)[::-1], color="slateblue")
        ax.set_xlabel("trips per week (thousands)")
        ax.set_title("Top 15 inter-dong OD pairs (all modes)")
        save(fig, out, "04_top_od_pairs.png")

    # ------------------------------------------------------------------ edges
    banner("EDGES: rail, bus, roads")
    rail_e = read_csv_safe(d / "edges_rail.csv")
    bus_e = read_csv_safe(d / "edges_bus.csv.gz")
    roads_attr = read_csv_safe(d / "edges_roads_attributes.csv.gz")
    topis = read_csv_safe(d / "topis_road_speeds_weekday_hourly.csv")

    if rail_e is not None:
        describe_df("edges_rail", rail_e)
        line_c = find_col(rail_e, "line")
        trains_c = find_col(rail_e, "train")
        len_c = find_col(rail_e, "length") or find_col(rail_e, "len")
        if "express_chord" in rail_e:
            print("  express_chord segments:", int(rail_e["express_chord"].astype(bool).sum()))
        topo = rail_e[~rail_e["express_chord"].astype(bool)] if "express_chord" in rail_e else rail_e
        if line_c and len_c:
            print("\n  network length per line (km, topological segments):")
            scale = 1 if "km" in len_c.lower() else 1 / 1000
            print((topo.groupby(line_c)[len_c].sum() * scale).round(1).sort_values(ascending=False)
                  .head(20).to_string())
    if bus_e is not None:
        describe_df("edges_bus", bus_e)
        print(bus_e.describe().round(1).T.to_string())
    if roads_attr is not None:
        describe_df("edges_roads_attributes", roads_attr)
        if "topis_matched" in roads_attr:
            print("  roads matched to TOPIS:",
                  int(roads_attr["topis_matched"].astype(bool).sum()), "of", len(roads_attr))
        hw = find_col(roads_attr, "highway")
        if hw:
            print(roads_attr[hw].value_counts().to_string())

    # --- TOPIS speeds
    speed_src = topis if topis is not None else roads_attr
    if speed_src is not None:
        vcols = [c for c in speed_src.columns if c.startswith("v_h")]
        if vcols:
            sp = speed_src[vcols].apply(pd.to_numeric, errors="coerce")
            hours = [int(c[3:5]) for c in vcols]
            med, q1, q3 = sp.median(), sp.quantile(.25), sp.quantile(.75)
            print(f"\n  TOPIS roads with speeds: {sp.notna().any(axis=1).sum()}")
            print(f"  network-median speed: slowest {med.min():.1f} km/h at {hours[int(np.argmin(med.values))]:02d}h; "
                  f"fastest {med.max():.1f} km/h at {hours[int(np.argmax(med.values))]:02d}h")
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].plot(hours, med.values, color="k", label="median")
            ax[0].fill_between(hours, q1.values, q3.values, alpha=.25, label="IQR")
            ax[0].set_xlabel("hour of day (weekday)"); ax[0].set_ylabel("km/h")
            ax[0].set_title("Road speed by hour (TOPIS)"); ax[0].legend()
            sub = sp.dropna(how="all")
            if len(sub) > 300:
                sub = sub.sample(300, random_state=0)
            order = sub.mean(axis=1).sort_values().index
            im = ax[1].imshow(sub.loc[order].values, aspect="auto", cmap="RdYlGn",
                              extent=[hours[0] - .5, hours[-1] + .5, len(sub), 0])
            ax[1].set_xlabel("hour"); ax[1].set_ylabel("roads (sorted by mean speed)")
            ax[1].set_title("Speed per road and hour"); plt.colorbar(im, ax=ax[1], label="km/h")
            save(fig, out, "07_topis_speeds.png")

    # ------------------------------------------------------------------ network overview map
    if gpd is not None and gpkg.exists():
        banner("GEOPACKAGE LAYERS")
        r_edges = read_layer(gpkg, "rail_edges")
        b_edges = read_layer(gpkg, "bus_edges")
        roads = read_layer(gpkg, "roads")
        r_nodes = read_layer(gpkg, "rail_stops")
        b_nodes = read_layer(gpkg, "bus_stops")
        for nm, g in [("rail_edges", r_edges), ("bus_edges", b_edges), ("roads", roads),
                      ("rail_stops", r_nodes), ("bus_stops", b_nodes)]:
            if g is not None:
                print(f"  {nm}: {len(g):,} features")

        fig, ax = plt.subplots(figsize=(10, 9))
        if dongs_gdf is not None:
            dongs_gdf.plot(ax=ax, color="#f2f2f2", edgecolor="white", linewidth=0.3)
        if b_edges is not None:
            tc = find_col(b_edges, "trip") or find_col(b_edges, "per_day")
            lw = 0.15 if tc is None else 0.1 + 0.8 * (b_edges[tc] / b_edges[tc].quantile(.99)).clip(0, 1)
            b_edges.plot(ax=ax, color="#2a9d8f", linewidth=lw, alpha=.35, label="bus")
        if roads is not None:
            hwc = find_col(roads, "highway")
            major = roads[roads[hwc].isin(["motorway", "trunk", "primary"])] if hwc else roads
            major.plot(ax=ax, color="#e9a100", linewidth=0.7, alpha=.8, label="major roads")
        if r_edges is not None:
            rt = r_edges[~r_edges["express_chord"].astype(bool)] if "express_chord" in r_edges else r_edges
            rt.plot(ax=ax, color="#c1121f", linewidth=1.2, label="rail")
        if r_nodes is not None:
            r_nodes.plot(ax=ax, color="black", markersize=4, zorder=5)
        ax.set_title("Seoul multimodal network: dongs, bus, major roads, rail")
        ax.legend(loc="lower left"); ax.set_axis_off()
        save(fig, out, "05_network_overview.png")

        # --- rail service + bus stop density
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        if r_edges is not None:
            tc = find_col(r_edges, "train")
            lc = find_col(r_edges, "line")
            if tc and lc:
                s = (r_edges[~r_edges["express_chord"].astype(bool)]
                     if "express_chord" in r_edges else r_edges).groupby(lc)[tc].median()
                s.sort_values().plot.barh(ax=ax[0], color="#c1121f")
                ax[0].set_xlabel("median trains per day per segment")
                ax[0].set_title("Rail service level by line")
        if dongs_gdf is not None and bus_stops is not None and "dong_code" in bus_stops:
            cnt = bus_stops["dong_code"].value_counts().rename("n_bus_stops").rename_axis("dong_code")
            g = dongs_gdf.merge(cnt.reset_index(), on="dong_code", how="left").fillna({"n_bus_stops": 0})
            choropleth(ax[1], g, "n_bus_stops", "Bus stops per dong", "Greens")
        save(fig, out, "06_rail_bus_service.png")

    # ------------------------------------------------------------------ routed travel times
    banner("ROUTED TRAVEL TIMES (R5)")
    od = read_csv_safe(d / "od_routed_travel_times.csv.gz", dtype={"origin": str, "destination": str})
    if od is not None:
        if "origin" not in od.columns:
            od = od.rename(columns={od.columns[0]: "origin", od.columns[1]: "destination"})
            od["origin"] = od["origin"].astype(str); od["destination"] = od["destination"].astype(str)
        describe_df("od_routed", od)
        # columns are e.g. transit_p50_min, car_topis_h08_min -> strip the unit suffix
        od = od.rename(columns=lambda c: c[:-4] if c.endswith("_min") else c)
        tcols = [c for c in od.columns if c not in ("origin", "destination")]
        off = od[od.origin != od.destination]
        print("\n  median routed time over off-diagonal pairs (min):")
        print(off[tcols].median().round(1).to_string())
        print("\n  share of OD pairs unreachable (NaN):")
        print((100 * off[tcols].isna().mean()).round(2).to_string())
        if {"transit_p50", "car_topis_h08"} <= set(od.columns):
            ratio = (off["transit_p50"] / off["car_topis_h08"]).replace([np.inf, -np.inf], np.nan)
            print(f"\n  transit / car (08h TOPIS) time ratio: median {ratio.median():.2f}, "
                  f"transit faster in {100 * (ratio < 1).mean():.1f}% of pairs")
        if {"car_freeflow", "car_topis_h08", "car_topis_h17", "car_topis_h03"} <= set(od.columns):
            print("  car slowdown vs free-flow (median ratio):",
                  {c: round(float((off[c] / off["car_freeflow"]).median()), 2)
                   for c in ("car_topis_h03", "car_topis_h08", "car_topis_h17")})

        # focus dong
        focus = args.focus_dong
        if focus is None and flows is not None:
            focus = per_dong["inflow"].idxmax()
        if focus is None:
            focus = od["destination"].iloc[0]
        print(f"\n  focus dong for maps: {focus} ({dong_names.get(focus, '?')})")

        if dongs_gdf is not None:
            to_f = od[od.destination == focus].set_index("origin")
            cols_to_plot = [c for c in ("walk", "transit_p50", "car_topis_h08", "car_topis_h17") if c in to_f]
            if not cols_to_plot:
                print("  (no routed-time columns found; skipping figure 08)")
                cols_to_plot = []
        if dongs_gdf is not None and cols_to_plot:
            fig, axs = plt.subplots(1, len(cols_to_plot), figsize=(4.5 * len(cols_to_plot), 5))
            axs = np.atleast_1d(axs)
            vmax = np.nanpercentile(to_f[cols_to_plot].values, 95)
            for a, c in zip(axs, cols_to_plot):
                g = dongs_gdf.merge(to_f[[c]].reset_index(names="dong_code"), on="dong_code", how="left")
                choropleth(a, g, c, f"{c} → {dong_names.get(focus, focus)} (min)", "magma_r",
                           vmin=0, vmax=vmax)
            save(fig, out, "08_routed_times.png")

        # KT observed vs routed
        if flows is not None:
            fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
            pairs = [("subway", "transit_p50"), ("car", "car_topis_h08")]
            for a, (kw, rc) in zip(axs, pairs):
                code = code_for(labels, kw)
                if code is None or rc not in od:
                    a.set_visible(False); continue
                f = flows[(flows["mode"] == code) & (~flows.intra_dong)]
                m = f.merge(od[["origin", "destination", rc]], on=["origin", "destination"])
                m = m.dropna(subset=[rc, "mean_time_min"])
                m = m[m.trips_week >= m.trips_week.quantile(.5)]
                if m.empty:
                    continue
                a.hexbin(m[rc], m["mean_time_min"], gridsize=50, bins="log", cmap="viridis",
                         extent=(0, 120, 0, 120), mincnt=1)
                a.plot([0, 120], [0, 120], "r--", lw=1)
                r = np.corrcoef(m[rc], m["mean_time_min"])[0, 1]
                a.set_title(f"KT '{labels[code]}' vs routed '{rc}' (r={r:.2f})")
                a.set_xlabel("R5 routed time (min)"); a.set_ylabel("KT mean time (min)")
            save(fig, out, "09_kt_vs_routed_time.png")

    # ------------------------------------------------------------------ interlayer
    banner("INTERLAYER COUPLINGS")
    rb = read_csv_safe(d / "interlayer_rail_bus_within300m.csv")
    if rb is not None:
        describe_df("rail–bus 300 m", rb)
        dist_c = find_col(rb, "dist")
        if dist_c:
            print(f"  median rail–bus distance: {rb[dist_c].median():.0f} m")
        rid = find_col(rb, "rail") or rb.columns[0]
        print(f"  bus stops per rail stop: median {rb.groupby(rid).size().median():.0f}, "
              f"max {rb.groupby(rid).size().max()}")
    snap = read_csv_safe(d / "interlayer_to_walk_nodes.csv")
    if snap is not None:
        describe_df("snap to walk nodes", snap)
        dc = find_col(snap, "dist")
        kc = find_col(snap, "type") or find_col(snap, "layer") or find_col(snap, "kind")
        if dc and kc:
            print((snap.groupby(kc)[dc].describe()[["count", "50%", "max"]]).round(1).to_string())

    # ------------------------------------------------------------------ walk
    if not args.skip_walk:
        banner("PEDESTRIAN LAYER")
        we = read_csv_safe(d / "edges_walk.csv.gz")
        wn = read_csv_safe(d / "nodes_walk.csv.gz", dtype={"dong_code": str})
        if we is not None:
            describe_df("edges_walk", we)
            hw, ln = find_col(we, "highway"), find_col(we, "length")
            if hw and ln:
                by_hw = (we.groupby(hw)[ln].sum() / 1000).sort_values(ascending=False)
                print(f"\n  total length: {by_hw.sum():,.0f} km")
                print(by_hw.round(0).head(12).to_string())
            gc = find_col(we, "giant")
            if gc and ln:
                print(f"  giant component share of length: "
                      f"{100 * we.loc[we[gc].astype(bool), ln].sum() / we[ln].sum():.2f}%")
        if wn is not None:
            describe_df("nodes_walk", wn)
            dg = find_col(wn, "degree")
            if dg:
                print("  node degree distribution (%):")
                print((100 * wn[dg].clip(upper=6).value_counts(normalize=True).sort_index()).round(1).to_string())

        wg = read_layer(gpkg, "walk_edges")
        if wg is not None and dongs_gdf is not None:
            hw = find_col(wg, "highway")
            fig, ax = plt.subplots(figsize=(10, 9))
            dongs_gdf.plot(ax=ax, color="#fafafa", edgecolor="#dddddd", linewidth=0.2)
            palette = {"residential": "#6c757d", "footway": "#e76f51", "path": "#2a9d8f",
                       "cycleway": "#264653", "service": "#adb5bd", "primary": "#e9c46a"}
            for k, c in palette.items():
                sub = wg[wg[hw] == k]
                if len(sub):
                    sub.plot(ax=ax, color=c, linewidth=0.15, label=f"{k} ({len(sub):,})")
            ax.legend(loc="lower left", markerscale=3)
            ax.set_title("Pedestrian network by highway type"); ax.set_axis_off()
            save(fig, out, "10_walk_network.png")

    banner("DONE")
    print(f"Figures written to: {out.resolve()}")


if __name__ == "__main__":
    main()
