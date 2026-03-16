# make_monthly_UBA_from_EDGAR_GNFR_NO_TOTAL_CORR_WITH_PLOTS.py
# FINAL:
# - Preserve sector seasonality 100% (EDGAR sector shapes, day-weighted normalized)
# - NO total correction (no closure / no scaling)
# - Outputs:
#   * Sector monthly NetCDFs
#   * TOTAL from sum(sectors) NetCDF (diagnostic)
#   * Plots:
#       - per-sector per-year: domain-mean monthly series
#       - per-year multipanel: all sectors domain-mean
#       - total diagnostic: sum(sectors) vs UBA total × EDGAR TOTAL shape


import os
import glob
import calendar
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd
import joblib
import matplotlib.pyplot as plt



# USER SETTINGS


UBA_PKL = r"D:\DogruKodlarTez\Anthropogenic\UBA_Inv_regridded.pkl"
EDGAR_DIR = r"D:\DogruKodlarTez\Anthropogenic\EdgarData"

OUTDIR = Path(r"D:\DogruKodlarTez\Anthropogenic\Results\Data\UBA_MONTHLY_FROM_EDGAR_SHAPE_NO_TOTAL_CORR")
OUTDIR.mkdir(parents=True, exist_ok=True)

EDGAR_VAR = "fluxes"
GAS_KEY = "CH4"

# Munich bbox
MINX, MAXX = 11.10, 12.10
MINY, MAXY = 47.80, 48.40

# UBA grid
DX = DY = 0.01
NY, NX = 60, 100

WRITE_NETCDFS = True
MAKE_PLOTS = True



# OUTPUT PLOT FOLDERS


PLOT_DIR = OUTDIR / "PLOTS"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_DIR_BY_SECTOR = PLOT_DIR / "by_sector"
PLOT_DIR_BY_SECTOR.mkdir(parents=True, exist_ok=True)

PLOT_DIR_BY_YEAR = PLOT_DIR / "by_year"
PLOT_DIR_BY_YEAR.mkdir(parents=True, exist_ok=True)

PLOT_DIR_TOTAL = PLOT_DIR / "total_checks"
PLOT_DIR_TOTAL.mkdir(parents=True, exist_ok=True)


# GNFR → EDGAR mapping


UBA_TO_EDGAR = {
    "A": ["POWER_INDUSTRY"],
    "B": ["IND_COMBUSTION", "IND_PROCESSES"],
    "C": ["BUILDINGS"],
    "D": ["FUEL_EXPLOITATION"],
    "E": ["IND_PROCESSES"],          # solvents proxy via processes
    "F": ["TRANSPORT"],
    "G": ["TRANSPORT"],
    "H": ["TRANSPORT"],
    "I": ["TRANSPORT"],
    "J": ["WASTE"],
    "K": ["AGRICULTURE"],
    "L": ["AGRICULTURE"],
}

GNFR_NAMES = {
    "A": "PublicPower",
    "B": "Industry",
    "C": "OtherStationaryComb",
    "D": "Fugitives",
    "E": "Solvents",
    "F": "RoadTransport",
    "G": "Shipping",
    "H": "Aviation",
    "I": "OffRoad",
    "J": "Waste",
    "K": "AgriLivestock",
    "L": "AgriOther",
}

EDGAR_TOTAL_COMPONENTS = [
    "POWER_INDUSTRY",
    "IND_COMBUSTION",
    "IND_PROCESSES",
    "BUILDINGS",
    "TRANSPORT",
    "AGRICULTURE",
    "FUEL_EXPLOITATION",
    "WASTE",
]



def month_days(year: int) -> np.ndarray:
    return np.array([calendar.monthrange(year, m)[1] for m in range(1, 13)], dtype=float)

def mid_month_times(year: int) -> pd.DatetimeIndex:
    return pd.to_datetime([f"{year}-{m:02d}-15" for m in range(1, 13)])

def build_coords():
    lons = MINX + (np.arange(NX) + 0.5) * DX
    lats = MINY + (np.arange(NY) + 0.5) * DY
    return lats, lons

def safe_close(ds):
    try:
        ds.close()
    except Exception:
        pass

def find_edgar(year: int, sector: str) -> str | None:
    # year → string çevir
    year_str = str(year)
    # EDGAR_DIR sonuna year klasörünü ekle
    search_dir = os.path.join(EDGAR_DIR, year_str)
    # pattern
    pat = f"*CH4_{year_str}*_{sector}_flx.nc"
    # dosya arama
    files = glob.glob(os.path.join(search_dir, pat))
    return files[0] if files else None

def munich_domain_mean(ds: xr.Dataset) -> xr.DataArray:
    lat = ds["lat"]
    lat_slice = slice(MAXY, MINY) if float(lat[0]) > float(lat[-1]) else slice(MINY, MAXY)
    sub = ds.sel(lon=slice(MINX, MAXX), lat=lat_slice)
    return sub[EDGAR_VAR].mean(dim=("lat", "lon"), skipna=True)

def normalize_day_weighted(ts12: np.ndarray, year: int) -> np.ndarray | None:
    d = month_days(year)
    denom = np.nansum(ts12 * d) / np.nansum(d)
    if not np.isfinite(denom) or denom == 0:
        return None
    return ts12 / denom


# PLOTTING


def plot_sector_domainmean(year, sec, t_index, dm, out_png):
    plt.figure(figsize=(10, 6))
    plt.plot(t_index, dm, marker="o")
    plt.title(f"Sector monthly CH4 (domain-mean) – {year} | GNFR {sec}: {GNFR_NAMES.get(sec,'')}")
    plt.xlabel("Time")
    plt.ylabel("CH4 flux (same units as UBA)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_year_multipanel(year, series_dict, out_png, title_suffix=""):
    secs = list("ABCDEFGHIJKL")
    ncols = 3
    nrows = int(np.ceil(len(secs) / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 12), sharex=True)
    axes = np.array(axes).reshape(-1)

    for i, sec in enumerate(secs):
        ax = axes[i]
        if sec in series_dict:
            t, y = series_dict[sec]
            ax.plot(t, y, marker="o")
            ax.set_title(f"{sec} {GNFR_NAMES.get(sec,'')}", fontsize=10)
            ax.grid(True, alpha=0.3)
        else:
            ax.axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"UBA monthly domain-mean by GNFR sector {title_suffix} – {year}", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def plot_total_compare(year, t_index, dm_sum, dm_ref, out_png):
    plt.figure(figsize=(10, 6))
    plt.plot(t_index, dm_sum, marker="o", label="TOTAL = sum(sectors) (no correction)")
    plt.plot(t_index, dm_ref, marker="x", linestyle="--",
             label="UBA total annual × EDGAR TOTAL shape (reference)")
    plt.title(f"TOTAL monthly CH4 (domain-mean) – {year} (diagnostic only)")
    plt.xlabel("Time")
    plt.ylabel("CH4 flux (same units as UBA)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# MAIN

def main():
    uba_all = joblib.load(UBA_PKL)
    if GAS_KEY not in uba_all:
        raise KeyError(f"{GAS_KEY} not found in UBA pkl keys: {list(uba_all.keys())}")
    uba = uba_all[GAS_KEY]  # dict: year(str) -> sector -> 2D

    lats, lons = build_coords()
    years = sorted(map(int, uba.keys()))
    print(f"[INFO] UBA years: {years}")

    for year in years:
        print(f"\n=== Year {year} ===")
        t_index = mid_month_times(year)

        total_sum_3d = np.zeros((12, NY, NX), dtype=float)
        series_for_multipanel = {}
        sectors_written = 0

        # build sector monthlies
        for sec, edgar_sectors in UBA_TO_EDGAR.items():
            if str(year) not in uba or sec not in uba[str(year)]:
                print(f"  {sec}: missing in UBA → skipped")
                continue

            annual = np.array(uba[str(year)][sec], dtype=float)
            if annual.shape != (NY, NX):
                print(f"  {sec}: shape mismatch {annual.shape} != {(NY, NX)} → skipped")
                continue

            monthly_ts = None
            used_files = []

            for eds in edgar_sectors:
                f = find_edgar(year, eds)
                if f is None:
                    continue
                used_files.append(os.path.basename(f))

                ds = xr.open_dataset(f)
                if EDGAR_VAR not in ds:
                    safe_close(ds)
                    continue

                ts = munich_domain_mean(ds).values.astype(float)
                safe_close(ds)

                monthly_ts = ts if monthly_ts is None else (monthly_ts + ts)

            if monthly_ts is None:
                print(f"  {sec}: no EDGAR files found → skipped")
                continue

            shape = normalize_day_weighted(monthly_ts, year)
            if shape is None or np.any(~np.isfinite(shape)):
                print(f"  {sec}: invalid shape → skipped")
                continue

            monthly_3d = shape[:, None, None] * annual[None, :, :]
            total_sum_3d += monthly_3d
            sectors_written += 1

            # domain-mean series for plots
            dm = monthly_3d.mean(axis=(1, 2))
            series_for_multipanel[sec] = (t_index, dm)

            # write sector netcdf
            if WRITE_NETCDFS:
                da = xr.DataArray(
                    monthly_3d,
                    dims=("time", "lat", "lon"),
                    coords={"time": t_index, "lat": lats, "lon": lons},
                    name="CH4"
                )
                da.attrs["description"] = f"UBA GNFR-{sec} monthly CH4, annual mean preserved, EDGAR sector shape (NO total correction)"
                da.attrs["GNFR_sector"] = sec
                da.attrs["GNFR_name"] = GNFR_NAMES.get(sec, "")
                da.attrs["EDGAR_mapping"] = "+".join(edgar_sectors)
                da.attrs["EDGAR_files_used"] = ";".join(used_files)

                out_nc = OUTDIR / f"UBA_CH4_{year}_GNFR-{sec}_monthly.nc"
                da.to_dataset().to_netcdf(out_nc)

            # per-sector plot (year+sector)
            if MAKE_PLOTS:
                out_png = PLOT_DIR_BY_SECTOR / f"sector_domainmean_{year}_GNFR-{sec}.png"
                plot_sector_domainmean(year, sec, t_index, dm, out_png)

            print(f"  {sec}: written")

        # write TOTAL from sum (diagnostic)
        if WRITE_NETCDFS:
            da_sum = xr.DataArray(
                total_sum_3d,
                dims=("time", "lat", "lon"),
                coords={"time": t_index, "lat": lats, "lon": lons},
                name="CH4"
            )
            da_sum.attrs["description"] = f"UBA TOTAL monthly from sum of sector monthlies (NO correction) – {year}"
            da_sum.attrs["sectors_written"] = str(sectors_written)
            out_sum = OUTDIR / f"UBA_CH4_{year}_TOTAL_fromSum_monthly.nc"
            da_sum.to_dataset().to_netcdf(out_sum)

        # multipanel plot (year)
        if MAKE_PLOTS:
            out_png_year = PLOT_DIR_BY_YEAR / f"multipanel_domainmean_{year}.png"
            plot_year_multipanel(year, series_for_multipanel, out_png_year, title_suffix="(no total correction)")

        # total diagnostic plot vs EDGAR total-shape reference (NO correction)
        if MAKE_PLOTS and str(year) in uba and "total" in uba[str(year)]:
            total_annual_map = np.array(uba[str(year)]["total"], dtype=float)
            if total_annual_map.shape == (NY, NX):
                edgar_total_ts = None
                for eds in EDGAR_TOTAL_COMPONENTS:
                    f = find_edgar(year, eds)
                    if f is None:
                        continue
                    ds = xr.open_dataset(f)
                    if EDGAR_VAR not in ds:
                        safe_close(ds)
                        continue
                    ts = munich_domain_mean(ds).values.astype(float)
                    safe_close(ds)
                    edgar_total_ts = ts if edgar_total_ts is None else (edgar_total_ts + ts)

                if edgar_total_ts is not None:
                    s_total = normalize_day_weighted(edgar_total_ts, year)
                    if s_total is not None and np.all(np.isfinite(s_total)):
                        total_ref_3d = s_total[:, None, None] * total_annual_map[None, :, :]
                        dm_sum = total_sum_3d.mean(axis=(1, 2))
                        dm_ref = total_ref_3d.mean(axis=(1, 2))

                        out_png_total = PLOT_DIR_TOTAL / f"TOTAL_domainmean_sum_vs_ref_{year}.png"
                        plot_total_compare(year, t_index, dm_sum, dm_ref, out_png_total)

        print(f"[DONE] Year {year} sectors_written={sectors_written}")

    print("\n[DONE] Outputs written to:")
    print(f"  {OUTDIR}")


if __name__ == "__main__":
    main()
