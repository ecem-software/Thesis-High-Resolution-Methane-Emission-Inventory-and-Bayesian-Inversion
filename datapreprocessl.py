import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# footprints
try:
    import xarray as xr
except Exception:
    xr = None

# obs
import pyarrow.dataset as ds
import pandas as pd



# CONFIG + UTILS


def load_cfg(p="configne.json") -> dict:
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(f"Missing config: {p.resolve()}")
    return json.loads(p.read_text(encoding="utf-8"))

def ensure_dir(p) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def extract_yyyymmdd_from_fp_path(fp: Path) -> str | None:
    # expects MUC_YYYYMMDD_column_ERA5.nc (or parent footprint_YYYYMMDD)
    name = fp.name
    digits = "".join(ch if ch.isdigit() else " " for ch in name).split()
    for d in digits:
        if len(d) == 8:
            return d
    parent = fp.parent.name
    if parent.startswith("footprint_") and parent[10:18].isdigit():
        return parent[10:18]
    return None

def in_range(d: str, start: str, end: str) -> bool:
    return (d >= start) and (d <= end)

def yyyymmddhhmm(dt: pd.Timestamp) -> str:
    return dt.strftime("%Y%m%d%H%M")


# LIST FOOTPRINT FILES

def list_footprints(cfg: dict) -> List[Path]:
    root = Path(cfg["root_footprints"])
    layout = cfg["footprint_layout"]
    years = cfg["years"]

    years_with_sub = set(layout.get("years_with_subfolders", []))
    years_flat = set(layout.get("years_flat", []))
    year_dir_glob = layout["year_dir_glob"]
    subdir_glob = layout.get("subdir_glob", "footprint_*")
    nc_glob = layout["nc_glob"]

    all_files: List[Path] = []
    for y in years:
        y = int(y)
        year_dir = root / year_dir_glob.format(year=y)
        if not year_dir.exists():
            print(f"[WARN] Missing year dir: {year_dir}")
            continue

        if y in years_with_sub:
            files_y = []
            for sub in sorted(year_dir.glob(subdir_glob)):
                if sub.is_dir():
                    files_y.extend(sorted(sub.glob(nc_glob)))
        elif y in years_flat:
            files_y = sorted(year_dir.glob(nc_glob))
        else:
            files_y = sorted(year_dir.rglob(nc_glob))

        print(f"{y}: {len(files_y)} footprint file")
        all_files.extend(files_y)

    print(f"TOTAL footprint files: {len(all_files)}")
    return sorted(all_files)


# ============================================================
# OBS STREAMING: 15-min bin + NETWORK LOWEST background
# ============================================================

def _stream_station_bins(
    parquet_path: Path,
    time_col: str,
    xcol: str,
    qcol: str | None,
    keep_q: int | None,
    campaign_start_yyyymmdd: str,
    campaign_end_yyyymmdd: str,
    bin_minutes: int,
    batch_size: int = 200_000,
) -> pd.Series:
    """
    Returns station binned XCH4 mean per 15-min bin (UTC).
    Uses streaming batches to avoid huge memory.
    """
    dataset = ds.dataset(str(parquet_path), format="parquet")
    cols = [time_col, xcol]
    if qcol:
        cols.append(qcol)

    # accumulate sum/count per bin
    sum_by_bin = {}
    cnt_by_bin = {}

    scanner = dataset.scanner(columns=cols, batch_size=batch_size)

    for batch in scanner.to_batches():
        df = batch.to_pandas()

        # time -> pandas datetime UTC
        t = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        df["_t"] = t
        df = df.dropna(subset=["_t", xcol])

        # campaign filter (day-based; robust)
        day = df["_t"].dt.strftime("%Y%m%d")
        df = df[(day >= campaign_start_yyyymmdd) & (day <= campaign_end_yyyymmdd)]
        if df.empty:
            continue

        # quality filter
        if qcol and (keep_q is not None) and (qcol in df.columns):
            df = df[df[qcol] == keep_q]
            if df.empty:
                continue

        # bin
        b = df["_t"].dt.floor(f"{bin_minutes}min")
        x = df[xcol].astype(float)

        # group quickly
        gsum = x.groupby(b).sum()
        gcnt = x.groupby(b).count()

        for k, v in gsum.items():
            sum_by_bin[k] = sum_by_bin.get(k, 0.0) + float(v)
        for k, v in gcnt.items():
            cnt_by_bin[k] = cnt_by_bin.get(k, 0) + int(v)

    if not cnt_by_bin:
        return pd.Series(dtype=float)

    idx = sorted(cnt_by_bin.keys())
    vals = []
    for k in idx:
        vals.append(sum_by_bin[k] / max(cnt_by_bin[k], 1))

    return pd.Series(vals, index=pd.DatetimeIndex(idx, tz="UTC")).sort_index()


def load_obs_15min_network_lowest(cfg: dict) -> Dict[str, Dict[str, float]]:
    """
    Josef "Lowest" mantığına benzer:
    - Her 15 dk için network içindeki minimum XCH4 = background
    - Enhancement(station) = XCH4_station - background
    - Sonuç: enh_15m[station][YYYYMMDDHHMM] = float
    """
    ocfg = cfg["observation"]
    root_obs = Path(ocfg["root_obs"])
    if not root_obs.exists():
        raise FileNotFoundError(f"observation.root_obs not found: {root_obs.resolve()}")

    stations = cfg["stations"]
    time_col = ocfg["time_col"]
    xcol = ocfg["xch4_col"]
    qcol = ocfg.get("qflag_col", None)
    keep_q = ocfg.get("qflag_keep_value", None)
    parquet_glob = ocfg.get("parquet_glob", "*.parquet")
    bin_minutes = int(ocfg.get("bin_minutes", 15))

    start = cfg.get("campaign_start", "19000101")
    end = cfg.get("campaign_end", "21000101")

    all_files = sorted(root_obs.glob(parquet_glob))
    if not all_files:
        raise FileNotFoundError(f"No parquet found in {root_obs} with glob={parquet_glob}")

    # station -> file
    station_file = {}
    for st in stations:
        hits = [f for f in all_files if f.name.lower().startswith(st.lower() + "_")]
        if not hits:
            print(f"[OBS:WARN] No parquet for station {st} in {root_obs}")
            continue
        # usually one big parquet per station
        station_file[st] = hits[0]

    station_bins = {}

    for st, f in station_file.items():
        try:
            s = _stream_station_bins(
                parquet_path=f,
                time_col=time_col,
                xcol=xcol,
                qcol=qcol,
                keep_q=keep_q,
                campaign_start_yyyymmdd=start,
                campaign_end_yyyymmdd=end,
                bin_minutes=bin_minutes,
            )
            if s.empty:
                print(f"[OBS:WARN] Station {st}: empty after filters.")
                continue
            station_bins[st] = s
            print(f"[OBS] {st}: bins={len(s)} span={s.index.min()} → {s.index.max()}")
        except Exception as e:
            print(f"[OBS:WARN] read fail {f.name}: {repr(e)}")

    if not station_bins:
        raise RuntimeError("No station data loaded. Check stations naming, parquet files, and column names in config.")

    # network dataframe
    df_net = pd.concat(station_bins, axis=1)
    bkg_min = df_net.min(axis=1, skipna=True).rename("bkg_min")

    enh_15m = {st: {} for st in stations}
    for st, s in station_bins.items():
        common = s.index.intersection(bkg_min.index)
        enh = (s.loc[common] - bkg_min.loc[common]).astype(float)
        enh = enh.replace([np.inf, -np.inf], np.nan).dropna()
        enh_15m[st] = {yyyymmddhhmm(t): float(v) for t, v in enh.items()}
        print(f"[ENH] {st}: n={len(enh_15m[st])}")

    return enh_15m


# ============================================================
# BUILD 15-min H and y by aligning footprints to enh bins
# ============================================================

def build_Hy_15min(cfg: dict, fp_files: List[Path], enh_15m: Dict[str, Dict[str, float]], outdir: Path):
    if xr is None:
        raise RuntimeError("Need xarray installed: pip install xarray netCDF4")

    start = cfg.get("campaign_start", "19000101")
    end = cfg.get("campaign_end", "21000101")
    stations = cfg["stations"]
    bin_minutes = int(cfg["observation"].get("bin_minutes", 15))

    kept_rows = []
    kept_times = []
    kept_station = []

    ny = nx = None
    skipped = 0
    used = 0

    skip_log = outdir / "skipped_fp_15min.txt"
    skip_log.write_text("", encoding="utf-8")

    for fp in fp_files:
        d = extract_yyyymmdd_from_fp_path(fp)
        if d is None or (not in_range(d, start, end)):
            continue

        try:
            dsx = xr.open_dataset(fp, engine="netcdf4")
        except Exception as e:
            skipped += 1
            skip_log.write_text(skip_log.read_text(encoding="utf-8") + f"OPEN_FAIL\t{fp}\t{repr(e)}\n", encoding="utf-8")
            continue

        try:
            if "recep_time" not in dsx.coords and "recep_time" not in dsx.variables:
                skipped += 1
                skip_log.write_text(skip_log.read_text(encoding="utf-8") + f"NO_RECEP_TIME\t{fp}\n", encoding="utf-8")
                continue

            try:
                rt = xr.decode_cf(dsx[["recep_time"]])["recep_time"].values
            except Exception:
                rt = dsx["recep_time"].values

            rt = pd.to_datetime(rt, utc=True, errors="coerce")
            if rt.isna().all():
                skipped += 1
                skip_log.write_text(skip_log.read_text(encoding="utf-8") + f"BAD_TIME\t{fp}\n", encoding="utf-8")
                continue

            if ny is None or nx is None:
                for st in stations:
                    vname = f"{st} foot"
                    if vname in dsx.data_vars:
                        arr = np.asarray(dsx[vname].values)
                        arr = np.squeeze(arr)
                        if arr.ndim == 3:
                            _, ny0, nx0 = arr.shape
                            ny, nx = int(ny0), int(nx0)
                            break

            if ny is None or nx is None:
                skipped += 1
                skip_log.write_text(skip_log.read_text(encoding="utf-8") + f"NO_STATION_FOOT_VARS\t{fp}\n", encoding="utf-8")
                continue

            for st in stations:
                vname = f"{st} foot"
                if vname not in dsx.data_vars:
                    continue
                if st not in enh_15m or not enh_15m[st]:
                    continue

                foot = np.asarray(dsx[vname].values)
                foot = np.squeeze(foot)
                if foot.ndim != 3:
                    continue

                for it, tstamp in enumerate(rt):
                    if pd.isna(tstamp):
                        continue

                    # IMPORTANT: floor footprint time to 15-min bin
                    tbin = pd.Timestamp(tstamp).floor(f"{bin_minutes}min")
                    key = yyyymmddhhmm(tbin)

                    if key[:8] < start or key[:8] > end:
                        continue

                    yval = enh_15m[st].get(key, None)
                    if yval is None:
                        continue

                    fp2d = foot[it, :, :]
                    fp2d = np.nan_to_num(fp2d, nan=0.0, posinf=0.0, neginf=0.0)

                    kept_rows.append(fp2d.reshape(ny * nx).astype(np.float32, copy=False))
                    kept_times.append(key)
                    kept_station.append(st)
                    used += 1

        except Exception as e:
            skipped += 1
            skip_log.write_text(skip_log.read_text(encoding="utf-8") + f"READ_FAIL\t{fp}\t{repr(e)}\n", encoding="utf-8")
        finally:
            dsx.close()

        if used > 0 and used % 5000 == 0:
            print(f"Built rows: {used}")

    if not kept_rows:
        raise RuntimeError("No aligned 15-min rows built. Check enh bins, station names, and footprint vars.")

    H = np.vstack(kept_rows)
    y = np.array([enh_15m[st][t] for st, t in zip(kept_station, kept_times)], dtype=np.float32)

    meta = {
        "grid_shape": [int(ny), int(nx)],
        "n_rows": int(H.shape[0]),
        "n_grid": int(H.shape[1]),
        "stations": stations,
        "campaign_start": start,
        "campaign_end": end,
        "obs_bin_minutes": int(cfg["observation"].get("bin_minutes", 15)),
        "background_method": "network_lowest (min across stations per 15-min bin)",
        "note": "Enhancement = XCH4_station_bin - min_network_XCH4_bin; footprint recep_time floored to bin."
    }

    print("\n15-min preprocessing diagnostics:")
    print("  H shape:", H.shape, "(n_obs x n_grid)")
    print("  y shape:", y.shape)
    print("  time span:", min(kept_times), "→", max(kept_times))
    print("  kept rows:", len(kept_times), " skipped files:", skipped)
    print("  skip log:", skip_log.resolve())

    return H, y, np.array(kept_times, dtype="U12"), meta


def main():
    cfg = load_cfg("configne.json")
    outdir = ensure_dir(cfg["output_dir"])

    fp_files = list_footprints(cfg)
    enh_15m = load_obs_15min_network_lowest(cfg)

    H, y, times, meta = build_Hy_15min(cfg, fp_files, enh_15m, outdir)

    npz_name = cfg.get("io", {}).get("preprocessed_npz_name", "preprocessed_15min_H_y_times.npz")
    np.savez_compressed(outdir / npz_name, H=H, y=y, times=times, meta=json.dumps(meta))
    (outdir / "preprocess_15min_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nSaved NPZ : {(outdir / npz_name).resolve()}")
    print(f"Saved META: {(outdir / 'preprocess_15min_meta.json').resolve()}")
    print("DONE.")


if __name__ == "__main__":
    main()
