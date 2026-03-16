# uba_monthly_to_hourly_profiles_DEU_FIXED_WITH_PLOTS_EXCEL.py
# ------------------------------------------------------------
# What this script does:
# 1) Reads UBA monthly sector NetCDFs (mean flux maps): UBA_CH4_YYYY_GNFR-X_monthly.nc
# 2) Disaggregates MONTHLY MEAN -> HOURLY MEAN using weekly + hourly profiles (DEU)
#    - daily_shape is mean-normalized over days (mean=1)
#    - hour_shape is mean-normalized over 24 hours (mean=1) via (fractions*24)
#    => Monthly mean is preserved.
# 3) Writes:
#    - hourly NetCDF per sector: UBA_CH4_YYYY_GNFR-X_hourly.nc
#    - hourly domain-mean plot per sector
#    - yearly multipanel plot with all sectors (domain-mean)
#    - yearly Excel workbook with one sheet per sector (hourly domain-mean time series)
# ------------------------------------------------------------

import re
import calendar
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt





UBA_MONTHLY_DIR = Path(r"D:\DogruKodlarTez\Anthropogenic\Results\Data\UBA_MONTHLY_FROM_EDGAR_SHAPE_NO_TOTAL_CORR")

# Outputs
OUTDIR = Path(r"D:\DogruKodlarTez\Anthropogenic\Results\Data\UBA_HOURLY_FROM_PROFILES")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Aux tables folder (the CSVs you uploaded)
AUX_DIR = Path(r"D:\DogruKodlarTez\Anthropogenic\AuxiliaryTablesHourly")

COUNTRY_A3 = "DEU"
MONTHLY_VAR = "CH4"     # variable name in monthly NetCDFs
HOURLY_VAR = "CH4"

# Enable outputs
WRITE_NETCDF = True
MAKE_PLOTS = True
WRITE_EXCEL = True

# Timezone shifts (matching original logic)
APPLY_UTC_SHIFT = True
APPLY_DST_SHIFT = True

# GNFR -> activity_code(s) in profile tables
# You can edit ONLY this mapping if you want different profile choices.
GNFR_TO_ACTIVITY = {
    "A": ["ENE", "TRF", "REF"],
    "B": ["IND", "CHE", "IRO", "NFE", "NMM", "PAP", "FOO", "PRU"],   # combined by key-average
    "C": ["RCO"],
    "D": ["PRO", "FFF"],
    "E": ["SOL"],
    "F": ["TRO"],
    "G": ["TNR"],
    "H": ["TNR"],
    "I": ["TNR"],
    "J": ["SWD", "WWT"],   # combined by key-average
    "K": ["ENF", "MNM"],
    "L": ["AGS", "AWB"],
}

GNFR_NAMES = {
    "A": "PublicPower",
    "B": "Industry+Processes",
    "C": "Buildings",
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

H_COLS = [f"h{i}" for i in range(1, 25)]



# OUTPUT FOLDERS

PLOT_DIR = OUTDIR / "PLOTS"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_DIR_BY_SECTOR = PLOT_DIR / "by_sector"
PLOT_DIR_BY_SECTOR.mkdir(parents=True, exist_ok=True)

PLOT_DIR_BY_YEAR = PLOT_DIR / "by_year"
PLOT_DIR_BY_YEAR.mkdir(parents=True, exist_ok=True)

EXCEL_DIR = OUTDIR / "EXCEL"
EXCEL_DIR.mkdir(parents=True, exist_ok=True)



# HELPERS

def parse_year_sector(p: Path):
    # UBA_CH4_2020_GNFR-A_monthly.nc
    m = re.search(r"UBA_CH4_(\d{4})_GNFR-([A-L])_monthly\.nc$", p.name)
    if not m:
        return None, None
    return int(m.group(1)), m.group(2)

def build_month_calendar_df(year: int, month: int) -> pd.DataFrame:
    """Rows: day_in_month + Weekday_id (Mon=1..Sun=7)."""
    cal = calendar.Calendar()
    rows = []
    for day, wd in cal.itermonthdays2(year, month):
        if day <= 0:
            continue
        rows.append({"day_in_month": day, "Weekday_id": wd + 1})
    return pd.DataFrame(rows)

def ensure_hourly_fractions_sum_to_one(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure h1..h24 sum to 1 for each row."""
    s = df[H_COLS].sum(axis=1).replace(0, np.nan)
    df[H_COLS] = df[H_COLS].div(s, axis=0)
    return df

def apply_utc_shift_row(row: pd.Series) -> pd.Series:
    """Shift hourly fractions left by UTC_reference."""
    row.loc[H_COLS] = np.roll(row.loc[H_COLS].values, -int(row["UTC_reference"]))
    return row

def apply_dst_adjust_row(row: pd.Series, day: int, month: int, year: int) -> pd.Series:
    """If date is within DST interval, shift left by 1 hour."""
    date_format = "%d/%m/%Y"
    current_date = datetime.strptime(f"{day}/{month}/{year}", date_format)
    in_dst = 1 if (row["start_dst_date"] < current_date < row["end_dst_date"]) else 0
    row["UTC_reference"] = int(in_dst)
    row.loc[H_COLS] = np.roll(row.loc[H_COLS].values, -int(row["UTC_reference"]))
    return row

def mean_normalize(x: np.ndarray) -> np.ndarray:
    """Normalize so mean(x)=1."""
    m = np.nanmean(x)
    if not np.isfinite(m) or m == 0:
        raise ValueError("Cannot mean-normalize: invalid mean.")
    return x / m

def combine_profiles_by_key(
    df: pd.DataFrame,
    activities: list[str],
    key_cols: list[str],
    numeric_cols: list[str],
    country_code: str
) -> pd.DataFrame:
    """
    Combine multiple activity_code profiles by averaging AFTER aligning by keys.
    - Filter country & activities
    - Group by (activity_code + key_cols): mean numeric (safety)
    - Group by key_cols: mean across activities
    """
    sub = df[(df["Country_code_A3"] == country_code) & (df["activity_code"].isin(activities))].copy()
    if sub.empty:
        raise ValueError(f"No rows for activities={activities}, country={country_code}")

    g1_cols = ["activity_code"] + key_cols
    sub = sub.groupby(g1_cols, as_index=False)[numeric_cols].mean()
    out = sub.groupby(key_cols, as_index=False)[numeric_cols].mean()

    out["Country_code_A3"] = country_code
    out["activity_code"] = "+".join(activities)
    return out

def plot_sector_hourly_domainmean(year: int, sec: str, t: pd.DatetimeIndex, y: np.ndarray, out_png: Path):
    plt.figure(figsize=(12, 5))
    plt.plot(t, y, linewidth=1.0)
    plt.title(f"Hourly CH4 domain-mean – {year} | GNFR-{sec} ({GNFR_NAMES.get(sec,'')})")
    plt.xlabel("Time")
    plt.ylabel("Hourly CH$_4$ flux (µmol m-2 s-1)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_year_multipanel(year: int, series_dict: dict, out_png: Path):
    secs = list("ABCDEFGHIJKL")
    ncols = 3
    nrows = int(np.ceil(len(secs) / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 12), sharex=True)
    axes = np.array(axes).reshape(-1)

    for i, sec in enumerate(secs):
        ax = axes[i]
        if sec in series_dict:
            t, y = series_dict[sec]
            ax.plot(t, y, linewidth=0.8)
            ax.set_title(f"{sec} {GNFR_NAMES.get(sec,'')}", fontsize=10)
            ax.grid(True, alpha=0.3)
        else:
            ax.axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Hourly domain-mean CH4 by GNFR sector – {year}", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ===============================
# LOAD AUX TABLES
# ===============================
weekly_profiles = pd.read_csv(AUX_DIR / "weekly_profiles.csv", sep=";", header=0)
weekend_days = pd.read_csv(AUX_DIR / "weekenddays.csv", sep=";", header=0)
week_days = pd.read_csv(AUX_DIR / "weekdays.csv", sep=";", header=0)
week_days_per_country = weekend_days.merge(week_days, how="left", on=["Weekend_type_id"])

hourly_profiles = pd.read_csv(AUX_DIR / "hourly_profiles.csv", sep=";", header=0)

timezones = pd.read_csv(
    AUX_DIR / "timezones_definition.csv",
    sep=",",
    header=0,
    dtype={"UTC_reference": np.int8, "TZ_id": np.uint16},
)

dst_times = pd.read_csv(
    AUX_DIR / "daylite_saving_times.csv",
    sep=";",
    header=0,
    dayfirst=True,
    parse_dates=["start_dst_date", "end_dst_date"],
    dtype={"TZ_id": np.uint16, "Year": np.int16},
)

tz_row = timezones.loc[timezones["Country_code_A3"] == COUNTRY_A3]
if tz_row.empty:
    raise ValueError(f"{COUNTRY_A3} not found in timezones_definition.csv")
TZ_ID = int(tz_row.iloc[0]["TZ_id"])
UTC_REF_BASE = int(tz_row.iloc[0]["UTC_reference"])


# ===============================
# MAIN
# ===============================
def main():
    monthly_files = sorted(UBA_MONTHLY_DIR.glob("UBA_CH4_*_GNFR-*_monthly.nc"))
    if not monthly_files:
        raise FileNotFoundError(f"No UBA monthly sector files found in: {UBA_MONTHLY_DIR}")

    # group files by year to create 1 Excel per year + multipanel plot per year
    by_year = {}
    for f in monthly_files:
        year, sec = parse_year_sector(f)
        if year is None:
            continue
        by_year.setdefault(year, []).append((sec, f))

    for year in sorted(by_year.keys()):
        print(f"\n==================== YEAR {year} ====================")
        # Excel writer for this year
        excel_path = EXCEL_DIR / f"UBA_CH4_{year}_hourly_domainmean.xlsx"
        writer = pd.ExcelWriter(excel_path, engine="openpyxl") if WRITE_EXCEL else None

        series_for_multipanel = {}

        for sec, f in sorted(by_year[year], key=lambda x: x[0]):
            activities = GNFR_TO_ACTIVITY.get(sec, None)
            if activities is None:
                print(f"[SKIP] GNFR-{sec}: no activity mapping")
                continue

            print(f"\n--- {year} | GNFR-{sec} | activities={activities} ---")

            ds_m = xr.open_dataset(f)
            if MONTHLY_VAR not in ds_m:
                ds_m.close()
                raise KeyError(f"{f.name} does not contain variable '{MONTHLY_VAR}'")
            da_m = ds_m[MONTHLY_VAR]  # (time=12, lat, lon)

            month_times = pd.to_datetime(da_m["time"].values)
            lat = da_m["lat"].values
            lon = da_m["lon"].values

            # WEEKLY (daily factors) combined by Weekday_id
            wp = combine_profiles_by_key(
                df=weekly_profiles,
                activities=activities,
                key_cols=["Weekday_id"],
                numeric_cols=["daily_factor"],
                country_code=COUNTRY_A3
            )

            # HOURLY combined by (month_id, Daytype_id)
            hp = combine_profiles_by_key(
                df=hourly_profiles,
                activities=activities,
                key_cols=["month_id", "Daytype_id"],
                numeric_cols=H_COLS,
                country_code=COUNTRY_A3
            )

            # normalize hourly fractions to sum=1
            hp = ensure_hourly_fractions_sum_to_one(hp)

            # attach timezone info
            hp["TZ_id"] = TZ_ID
            hp["UTC_reference"] = UTC_REF_BASE

            if APPLY_UTC_SHIFT:
                hp = hp.apply(apply_utc_shift_row, axis=1)

            dst_year = dst_times[dst_times["Year"] == year].copy()

            # Build hourly arrays for the whole year (sector)
            hourly_time = []
            hourly_vals = []

            for t in month_times:
                month = int(t.month)
                month_id = month

                monthly_grid = da_m.sel(time=t).values.astype(np.float64)  # monthly MEAN

                # month calendar
                cal_df = build_month_calendar_df(year, month)
                cal_df["Country_code_A3"] = COUNTRY_A3

                # weekday -> daytype mapping (DEU weekend definition)
                cal_df = cal_df.merge(week_days_per_country, how="left", on=["Country_code_A3", "Weekday_id"])
                if cal_df["Daytype_id"].isna().any():
                    miss = cal_df[cal_df["Daytype_id"].isna()]
                    raise ValueError(f"Missing Daytype_id mapping for {year}-{month:02d}:\n{miss}")

                # merge daily factors
                cal_df = cal_df.merge(wp[["Weekday_id", "daily_factor"]], how="left", on=["Weekday_id"])
                if cal_df["daily_factor"].isna().any():
                    miss = cal_df[cal_df["daily_factor"].isna()]
                    raise ValueError(f"Missing daily_factor for {year}-{month:02d}:\n{miss}")

                # MEAN-FLUX logic: daily shape mean over days = 1 (preserves monthly mean)
                daily_shape = mean_normalize(cal_df["daily_factor"].to_numpy(dtype=float))

                # hourly profiles for this month
                hp_month = hp[hp["month_id"] == month_id].copy()
                if hp_month.empty:
                    raise ValueError(f"No hourly profiles for month_id={month_id}, activities={activities}")

                hp_month = hp_month.merge(dst_year, how="left", on=["TZ_id"])

                for i, drow in cal_df.iterrows():
                    day = int(drow["day_in_month"])
                    daytype = int(drow["Daytype_id"])
                    dshape = float(daily_shape[i])

                    hp_row = hp_month[hp_month["Daytype_id"] == daytype].copy()
                    if hp_row.empty:
                        raise ValueError(f"No hourly profile for Daytype_id={daytype} in {year}-{month:02d}")

                    if APPLY_DST_SHIFT:
                        hp_row = hp_row.apply(lambda r: apply_dst_adjust_row(r, day, month, year), axis=1)

                    hour_fracs = hp_row.iloc[0][H_COLS].to_numpy(dtype=float)
                    # fractions sum=1 => mean=1/24 -> multiply by 24 so mean=1 (preserves daily mean)
                    hour_shape = hour_fracs * 24.0

                    for h in range(24):
                        ts = pd.Timestamp(year=year, month=month, day=day, hour=h)
                        hourly_time.append(ts)
                        hourly_vals.append(monthly_grid * dshape * hour_shape[h])

            hourly_vals = np.stack(hourly_vals, axis=0)  # (time, lat, lon)
            hourly_time = pd.to_datetime(hourly_time)

            # Write NetCDF (per sector)
            if WRITE_NETCDF:
                out_nc = OUTDIR / f"UBA_CH4_{year}_GNFR-{sec}_hourly.nc"
                da_out = xr.DataArray(
                    hourly_vals,
                    dims=("time", "lat", "lon"),
                    coords={"time": hourly_time, "lat": lat, "lon": lon},
                    name=HOURLY_VAR
                )
                da_out.attrs["description"] = (
                    "Hourly disaggregation from UBA MONTHLY MEAN sector grids using weekly + hourly profiles "
                    "(DEU/Munich). Monthly mean preserved. NO total correction."
                )
                da_out.attrs["source_monthly_file"] = f.name
                da_out.attrs["GNFR_sector"] = sec
                da_out.attrs["GNFR_name"] = GNFR_NAMES.get(sec, "")
                da_out.attrs["activity_code_used"] = "+".join(activities)
                da_out.attrs["country"] = COUNTRY_A3
                da_out.attrs["TZ_id"] = str(TZ_ID)
                da_out.attrs["UTC_reference_base"] = str(UTC_REF_BASE)
                da_out.attrs["apply_utc_shift"] = str(bool(APPLY_UTC_SHIFT))
                da_out.attrs["apply_dst_shift"] = str(bool(APPLY_DST_SHIFT))
                da_out.to_dataset().to_netcdf(out_nc)
                print(f"[WRITE] {out_nc.name}  hours={len(hourly_time)}")

            # Domain-mean series (for plots + excel)
            dm = hourly_vals.mean(axis=(1, 2))  # (time,)
            series_for_multipanel[sec] = (hourly_time, dm)

            # Plot per sector
            if MAKE_PLOTS:
                out_png = PLOT_DIR_BY_SECTOR / f"hourly_domainmean_{year}_GNFR-{sec}.png"
                plot_sector_hourly_domainmean(year, sec, hourly_time, dm, out_png)

            # Excel per sector sheet (domain-mean only; Excel cannot handle full grid)
            if WRITE_EXCEL and writer is not None:
                df_out = pd.DataFrame({
                    "time": hourly_time,
                    "CH4_domain_mean": dm
                })
                # Excel sheet names max 31 chars
                sheet = f"GNFR-{sec}"
                df_out.to_excel(writer, sheet_name=sheet, index=False)

            ds_m.close()

        # multipanel plot for the year
        if MAKE_PLOTS and series_for_multipanel:
            out_png_year = PLOT_DIR_BY_YEAR / f"multipanel_hourly_domainmean_{year}.png"
            plot_year_multipanel(year, series_for_multipanel, out_png_year)

        # finalize Excel
        if WRITE_EXCEL and writer is not None:
            writer.close()
            print(f"[WRITE] Excel: {excel_path.name}")

    print("\n[DONE] All outputs written to:")
    print(f"  {OUTDIR}")
    if MAKE_PLOTS:
        print(f"  Plots: {PLOT_DIR}")
    if WRITE_EXCEL:
        print(f"  Excel: {EXCEL_DIR}")


if __name__ == "__main__":
    main()
