# inversion_monthly_15min_fallback_year.py
# - Monthly inversion using 15-min preprocessed (H,y,times)
# - Prior is read from UBA monthly NetCDF sector files (sum GNFR A..L)
# - NO biogenic
# - FALLBACK YEAR: if requested year not available in prior folder, use nearest available year

import json
from pathlib import Path
import calendar
import re

import numpy as np
import matplotlib.pyplot as plt

try:
    import xarray as xr
except Exception:
    xr = None

from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import spsolve


# -------------------------
# helpers
# -------------------------
def month_key_from_yyyymmddhhmm(s: str) -> str:
    # YYYYMMDDHHMM -> YYYY-MM
    return f"{s[:4]}-{s[4:6]}"

def month_sort_key(m: str):
    y, mm = m.split("-")
    return (int(y), int(mm))

def seconds_in_month(year: int, month: int) -> int:
    ndays = calendar.monthrange(year, month)[1]
    return ndays * 24 * 3600



# Bayes

def bayesian_inversion(y, K, x_a, S_a, S_eps):
    y = np.asarray(y, dtype=float).ravel()
    K = np.asarray(K, dtype=float)
    x_a = np.asarray(x_a, dtype=float).ravel()

    K_sparse = csc_matrix(K)
    S_a_sparse = csc_matrix(S_a)
    S_eps_diag = diags(np.diag(S_eps))

    # Gain matrix
    G = (spsolve(K_sparse @ S_a_sparse @ K_sparse.T + S_eps_diag,
                 K_sparse @ S_a_sparse)).T

    x_hat = x_a + G @ (y - K_sparse @ x_a)
    return x_hat, G


# prior from monthly sector NC (sum A..L)

def parse_year_sector(fn: str):
    m = re.search(r"UBA_CH4_(\d{4})_GNFR-([A-L])_monthly\.nc$", fn)
    if not m:
        return None
    return m.group(1), m.group(2)

def pick_flux_var(ds: "xr.Dataset") -> str:
    candidates = list(ds.data_vars)
    if not candidates:
        raise KeyError("No data_vars in dataset.")
    for name in ["CH4", "ch4", "flux", "fluxes", "emis", "emissions"]:
        if name in candidates:
            return name
    return candidates[0]


# ---------- OPTION A: FALLBACK YEAR HELPERS ----------
def available_years_in_prior(folder: Path) -> list[int]:
    """
    Scan folder for UBA_CH4_YYYY_GNFR-*_monthly.nc
    and return sorted unique years.
    """
    years = set()
    for p in folder.glob("UBA_CH4_*_GNFR-*_monthly.nc"):
        ps = parse_year_sector(p.name)
        if ps:
            ystr, _ = ps
            years.add(int(ystr))
    return sorted(years)

def pick_fallback_year(request_year: int, years_avail: list[int]) -> int:
    """
    Choose nearest available year.
    """
    return min(years_avail, key=lambda y: abs(y - request_year))


def read_monthly_total_map(
    folder: Path,
    year: int,
    month: int,
    target_shape: tuple[int, int],
    *,
    years_avail_cache: list[int] | None = None
) -> tuple[np.ndarray, int]:
    """
    Returns:
      total_map_2d (ny,nx), used_year
    If year not present in folder, uses nearest available year (fallback).
    """
    if xr is None:
        raise RuntimeError("Need xarray installed: pip install xarray netCDF4")

    # determine available years
    years_avail = years_avail_cache if years_avail_cache is not None else available_years_in_prior(folder)
    if not years_avail:
        raise FileNotFoundError(f"No GNFR monthly files found in {folder}")

    use_year = year if year in years_avail else pick_fallback_year(year, years_avail)
    if use_year != year:
        print(f"[PRIOR:FALLBACK] year {year} not found -> using {use_year}")

    ystr = f"{use_year:04d}"

    # collect sector files for chosen year
    files = []
    for p in folder.glob(f"UBA_CH4_{ystr}_GNFR-*_monthly.nc"):
        if parse_year_sector(p.name):
            files.append(p)

    if not files:
        # This shouldn't happen if years_avail worked, but keep robust:
        raise FileNotFoundError(f"No GNFR monthly files for year {ystr} in {folder}")

    total = None

    # month is 1..12, netcdf indexing 0..11
    midx = month - 1
    if not (0 <= midx <= 11):
        raise ValueError(f"month must be 1..12, got {month}")

    for f in sorted(files):
        dsx = xr.open_dataset(f, engine="netcdf4")
        try:
            v = pick_flux_var(dsx)
            arr = np.asarray(dsx[v].values)
            arr = np.squeeze(arr)

            if arr.ndim == 3:
                # expect (time=12, y, x)
                if arr.shape[0] < 12:
                    raise ValueError(f"{f.name}: first dim expected >=12 months, got {arr.shape}")
                arr_m = arr[midx, :, :]
            elif arr.ndim == 2:
                # already a monthly slice (rare)
                arr_m = arr
            else:
                raise ValueError(f"Unexpected var shape in {f.name}: {arr.shape}")

            arr_m = np.nan_to_num(arr_m, nan=0.0, posinf=0.0, neginf=0.0)

            if arr_m.shape != target_shape:
                raise ValueError(f"Shape mismatch in {f.name}: {arr_m.shape} vs {target_shape}")

            total = arr_m if total is None else (total + arr_m)
        finally:
            dsx.close()

    if total is None:
        raise RuntimeError(f"Could not build total map for {use_year}-{month:02d}")

    return total, use_year


# main

def main():
    cfg = json.loads(Path("configne.json").read_text(encoding="utf-8"))
    outdir = Path(cfg["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    npz_name = cfg.get("io", {}).get("preprocessed_npz_name", "preprocessed_15min_H_y_times.npz")
    npz_path = outdir / npz_name
    if not npz_path.exists():
        raise FileNotFoundError(f"Preprocessed npz not found: {npz_path.resolve()}")

    z = np.load(npz_path, allow_pickle=True)
    H = z["H"].astype(float)
    y = z["y"].astype(float).ravel()
    times = [str(s) for s in z["times"]]
    meta = json.loads(str(z["meta"]))

    ny, nx = meta["grid_shape"]
    n_grid = ny * nx

    # group rows by month
    rows_by_month: dict[str, list[int]] = {}
    for i, t in enumerate(times):
        mk = month_key_from_yyyymmddhhmm(t)
        rows_by_month.setdefault(mk, []).append(i)

    months = sorted(rows_by_month.keys(), key=month_sort_key)

    min_obs = int(cfg.get("monthly", {}).get("min_obs_per_month", 200))
    print("Total aligned rows:", len(times))
    print("Unique months:", len(months))
    print("min_obs_per_month:", min_obs)
    print("First 12 months counts:")
    for mk in months[:12]:
        print(" ", mk, len(rows_by_month[mk]))

    # constants for kton/month
    # NOTE: if you have per-lat cell areas, replace with area vector;
    # here we use constant cell area (your earlier approach).
    dlat, dlon = 0.01, 0.01
    lat_m = 111_320.0
    lon_m = 111_320.0 * np.cos(np.deg2rad(48.1))
    cell_area = (lat_m * lon_m * dlat * dlon)
    UMOL_TO_KT = 1.604e-14

    prior_folder = Path(cfg["prior_monthly_nc_folder"])
    if not prior_folder.exists():
        raise FileNotFoundError(f"prior_monthly_nc_folder not found: {prior_folder}")

    # cache available years once (faster + consistent)
    years_avail = available_years_in_prior(prior_folder)
    print("[PRIOR] available years:", years_avail)

    results = {}
    months_kept = []
    prior_list = []
    post_list = []

    for mk in months:
        idx = rows_by_month[mk]
        if len(idx) < min_obs:
            continue

        yr = int(mk.split("-")[0])
        mo = int(mk.split("-")[1])

        # prior (NO biogenic): sum A..L for that month, with fallback year
        prior_map, used_year = read_monthly_total_map(
            prior_folder, yr, mo, (ny, nx),
            years_avail_cache=years_avail
        )
        x_a = prior_map.reshape(n_grid)

        K = H[idx, :]
        ym = y[idx]

        # column scaling for numerical stability
        col_scale = np.maximum(np.linalg.norm(K, axis=0), 1e-12)
        K_s = K / col_scale
        xas = x_a * col_scale

        # Sa relative
        f_sigma_a = 1.0
        Sa_diag = (f_sigma_a * np.maximum(xas, 1e-12)) ** 2
        S_a = np.diag(Sa_diag)

        resid0 = ym - (K_s @ xas)
        sigma_eps = max(float(np.nanstd(resid0)), 1e-6)
        S_eps = np.eye(ym.size) * (sigma_eps ** 2)

        x_hat_s, G = bayesian_inversion(ym, K_s, xas, S_a, S_eps)
        x_hat = x_hat_s / col_scale

        sec = seconds_in_month(yr, mo)
        prior_kton = float(np.nansum(x_a   * cell_area * sec * UMOL_TO_KT))
        post_kton  = float(np.nansum(x_hat * cell_area * sec * UMOL_TO_KT))

        results[mk] = {
            "n_obs": int(len(idx)),
            "prior_kton": prior_kton,
            "posterior_kton": post_kton,
            "sigma_eps": sigma_eps,
            "AvK_trace": float(np.trace(G @ K_s)),
            "prior_year_used": int(used_year)
        }

        months_kept.append(mk)
        prior_list.append(prior_kton)
        post_list.append(post_kton)

        tag = " [15-min high-res]"
        if used_year != yr:
            tag += f" (prior_year={used_year})"
        print(f"{mk}: n={len(idx)} prior={prior_kton:.3f} post={post_kton:.3f}{tag}")

    if not months_kept:
        raise RuntimeError("No months passed min_obs_per_month. Lower threshold or check coverage.")

    import joblib
    out_pkl = outdir / "monthly_inversion_15min_NO_BIO_fromNC_FALLBACKYEAR.pkl"
    joblib.dump(results, out_pkl, compress=3)
    print(f"\nSaved: {out_pkl.resolve()}")

    # plot
    fig, ax = plt.subplots(figsize=(18, 6))
    x = np.arange(len(months_kept))
    w = 0.38
    ax.bar(x - w/2, prior_list, width=w,
           label="Prior (UBA monthly NC, sum A..L, NO bio; fallback year)",
           color="#87CEEB")
    ax.bar(x + w/2, post_list,  width=w,
           label="Posterior (15-min)",
           color="#FFA500")

    ax.set_xticks(x)
    ax.set_xticklabels(months_kept, rotation=45, ha="right")
    ax.set_ylabel("Total CH₄ emissions of Munich [kton/month]")
    ax.set_title("Monthly prior and posterior CH$_4$ emissions over the Munich domain from Bayesian inversion")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    plt.tight_layout()

    fig_path = outdir / "monthly_prior_posterior_15min_NO_BIO_fromNC_FALLBACKYEAR.png"
    plt.savefig(fig_path, dpi=200)
    plt.show()
    print(f"Saved figure: {fig_path.resolve()}")


if __name__ == "__main__":
    main()
