import os, glob, calendar
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# ===============================
# USER SETTINGS
# ===============================

# EDGAR monthly NetCDF folder (same as your annual->monthly script)
EDGAR_DIR = Path(r"D:\DogruKodlarTez\Anthropogenic\EdgarData")
EDGAR_VAR = "fluxes"   # in EDGAR files

# AUX tables folder (same as your monthly->hourly script)
AUX_DIR = Path(r"D:\DogruKodlarTez\Anthropogenic\AuxiliaryTablesHourly")

# Outputs (you can change)
OUTDIR = Path(r"D:\DogruKodlarTez\Anthropogenic\Results\Data\TEMPORAL_PROFILES_PURE_ANNUAL_TO_ALL")
OUTDIR.mkdir(parents=True, exist_ok=True)

COUNTRY_A3 = "DEU"
YEARS = list(range(2019, 2026))  # 2019–2025

# Munich bbox (same as your annual->monthly script)
MINX, MAXX = 11.10, 12.10
MINY, MAXY = 47.80, 48.40

# Apply timezone adjustments (same logic family as your hourly script)
APPLY_UTC_SHIFT = True
APPLY_DST_SHIFT = True

# Weekly aggregation rule (display)
WEEKLY_RULE = "W-MON"  # keep as you used before


# ===============================
# SECTOR SET (requested)
# GNFR -> EDGAR mapping for MONTHLY seasonality
# ===============================
GNFR_TO_EDGAR = {
    "A": ("PublicPower", ["POWER_INDUSTRY"]),
    "C": ("Buildings",   ["BUILDINGS"]),
    "J": ("Waste",       ["WASTE"]),
    "K": ("Livestock",   ["AGRICULTURE"]),
}

# GNFR -> activity_code(s) for weekly/hourly profile tables
GNFR_TO_ACTIVITY = {
    "A": ["ENE"],
    "C": ["RCO"],
    "J": ["SWD", "WWT"],
    "K": ["AWB"],
}

H_COLS = [f"h{i}" for i in range(1, 25)]


# ===============================
# OUTPUT FOLDERS
# ===============================
PLOT_DIR = OUTDIR / "PLOTS"
EXCEL_DIR = OUTDIR / "EXCEL"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
EXCEL_DIR.mkdir(parents=True, exist_ok=True)

for freq in ["hourly", "daily", "weekly", "monthly"]:
    (PLOT_DIR / freq / "by_sector").mkdir(parents=True, exist_ok=True)
    (PLOT_DIR / freq / "combined").mkdir(parents=True, exist_ok=True)


# ===============================
# HELPERS
# ===============================
def month_days(year: int) -> np.ndarray:
    return np.array([calendar.monthrange(year, m)[1] for m in range(1, 13)], dtype=float)

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
    s = df[H_COLS].sum(axis=1).replace(0, np.nan)
    df[H_COLS] = df[H_COLS].div(s, axis=0)
    return df

def mean_normalize(x: np.ndarray) -> np.ndarray:
    m = np.nanmean(x)
    if not np.isfinite(m) or m == 0:
        raise ValueError("Cannot mean-normalize: invalid mean.")
    return x / m

def normalize_day_weighted(ts12: np.ndarray, year: int) -> np.ndarray:
    """Day-weighted annual mean becomes 1 (same as your annual->monthly script)."""
    d = month_days(year)
    denom = np.nansum(ts12 * d) / np.nansum(d)
    if not np.isfinite(denom) or denom == 0:
        raise ValueError("Invalid denom in day-weighted normalization.")
    return ts12 / denom

def combine_profiles_by_key(df, activities, key_cols, numeric_cols, country_code):
    sub = df[(df["Country_code_A3"] == country_code) & (df["activity_code"].isin(activities))].copy()
    if sub.empty:
        raise ValueError(f"No rows for activities={activities}, country={country_code}")

    g1_cols = ["activity_code"] + key_cols
    sub = sub.groupby(g1_cols, as_index=False)[numeric_cols].mean()
    out = sub.groupby(key_cols, as_index=False)[numeric_cols].mean()
    out["Country_code_A3"] = country_code
    out["activity_code"] = "+".join(activities)
    return out

def apply_utc_shift_row(row: pd.Series) -> pd.Series:
    row.loc[H_COLS] = np.roll(row.loc[H_COLS].values, -int(row["UTC_reference"]))
    return row

def apply_dst_adjust_row(row: pd.Series, day: int, month: int, year: int) -> pd.Series:
    # If DST info missing, do nothing
    if pd.isna(row.get("start_dst_date", pd.NaT)) or pd.isna(row.get("end_dst_date", pd.NaT)):
        return row

    date_format = "%d/%m/%Y"
    current_date = datetime.strptime(f"{day}/{month}/{year}", date_format)
    in_dst = 1 if (row["start_dst_date"] < current_date < row["end_dst_date"]) else 0
    row["UTC_reference"] = int(in_dst)
    row.loc[H_COLS] = np.roll(row.loc[H_COLS].values, -int(row["UTC_reference"]))
    return row

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

def read_edgar_monthly_ts(year: int, edgar_sectors: list[str]) -> np.ndarray | None:
    ts_sum = None
    used = 0
    for s in edgar_sectors:
        f = find_edgar(year, s)
        if f is None:
            continue
        ds = xr.open_dataset(f)
        if EDGAR_VAR not in ds:
            ds.close()
            continue
        ts = munich_domain_mean(ds).values.astype(float)  # 12 values
        ds.close()
        ts_sum = ts if ts_sum is None else (ts_sum + ts)
        used += 1
    return ts_sum if used > 0 else None


# ===============================
# PLOTTING
# ===============================
def plot_series(t: pd.DatetimeIndex, y: np.ndarray, title: str, out_png: Path):
    plt.figure(figsize=(12, 5))
    plt.plot(t, y, linewidth=1.0)
    plt.axhline(1.0, linewidth=1.0)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Scaling factor (pure)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_combined(series_dict: dict, title: str, out_png: Path):
    plt.figure(figsize=(12, 5))
    for label, (t, y) in series_dict.items():
        plt.plot(t, y, linewidth=1.0, label=label)
    plt.axhline(1.0, linewidth=1.0)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Scaling factor (pure)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# ===============================
# LOAD AUX TABLES (weekly/hourly)
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
# CORE: Build annual->hourly pure scaling factor series
# ===============================
def build_annual_to_hourly_factor(year: int, gnfr: str) -> pd.Series | None:
    """
    Pure scaling factor for annual->hourly:
      (EDGAR monthly seasonality shape) * (within-month daily shape) * (within-day hourly shape)
    No UBA multiplication.
    """
    # 1) EDGAR monthly seasonality (12 values)
    _, edgar_sectors = GNFR_TO_EDGAR[gnfr]
    ts12 = read_edgar_monthly_ts(year, edgar_sectors)
    if ts12 is None:
        return None
    month_shape = normalize_day_weighted(ts12, year)  # day-weighted annual mean = 1

    # 2) Weekly + hourly shapes for DEU
    activities = GNFR_TO_ACTIVITY[gnfr]

    wp = combine_profiles_by_key(
        df=weekly_profiles,
        activities=activities,
        key_cols=["Weekday_id"],
        numeric_cols=["daily_factor"],
        country_code=COUNTRY_A3
    )

    hp = combine_profiles_by_key(
        df=hourly_profiles,
        activities=activities,
        key_cols=["month_id", "Daytype_id"],
        numeric_cols=H_COLS,
        country_code=COUNTRY_A3
    )

    hp = ensure_hourly_fractions_sum_to_one(hp)
    hp["TZ_id"] = TZ_ID
    hp["UTC_reference"] = UTC_REF_BASE

    if APPLY_UTC_SHIFT:
        hp = hp.apply(apply_utc_shift_row, axis=1)

    dst_year = dst_times[dst_times["Year"] == year].copy()
    hp = hp.merge(dst_year, how="left", on=["TZ_id"])

    times = []
    factors = []

    # Build hour-by-hour
    for month in range(1, 13):
        month_id = month
        mfac = float(month_shape[month - 1])  # monthly factor from EDGAR

        cal_df = build_month_calendar_df(year, month)
        cal_df["Country_code_A3"] = COUNTRY_A3

        # weekday -> daytype mapping
        cal_df = cal_df.merge(week_days_per_country, how="left", on=["Country_code_A3", "Weekday_id"])
        if cal_df["Daytype_id"].isna().any():
            raise ValueError(f"Missing Daytype_id mapping for {year}-{month:02d}")

        # merge daily factors
        cal_df = cal_df.merge(wp[["Weekday_id", "daily_factor"]], how="left", on=["Weekday_id"])
        if cal_df["daily_factor"].isna().any():
            raise ValueError(f"Missing daily_factor for {year}-{month:02d}")

        # within-month mean normalized (preserves monthly mean)
        daily_shape = mean_normalize(cal_df["daily_factor"].to_numpy(dtype=float))

        hp_month = hp[hp["month_id"] == month_id].copy()
        if hp_month.empty:
            raise ValueError(f"No hourly profiles for month_id={month_id}, gnfr={gnfr}")

        for i, drow in cal_df.iterrows():
            day = int(drow["day_in_month"])
            daytype = int(drow["Daytype_id"])
            dfac = float(daily_shape[i])

            hp_row = hp_month[hp_month["Daytype_id"] == daytype].copy()
            if hp_row.empty:
                raise ValueError(f"No hourly profile for Daytype_id={daytype} in {year}-{month:02d}")

            if APPLY_DST_SHIFT:
                hp_row = hp_row.apply(lambda r: apply_dst_adjust_row(r, day, month, year), axis=1)

            hour_fracs = hp_row.iloc[0][H_COLS].to_numpy(dtype=float)  # sum=1
            hour_shape = hour_fracs * 24.0  # mean=1

            for h in range(24):
                ts = pd.Timestamp(year=year, month=month, day=day, hour=h)
                times.append(ts)
                factors.append(mfac * dfac * hour_shape[h])

    s = pd.Series(np.array(factors, dtype=float), index=pd.to_datetime(times)).sort_index()
    s.name = "scaling_factor_hourly"
    return s


# ===============================
# MAIN
# ===============================
def main():
    for year in YEARS:
        print(f"\n==================== YEAR {year} ====================")

        # Önce tüm sektör serilerini üretelim (varsa)
        results = {}  # gnfr -> dict(hour/day/week/month series)
        for gnfr, (name, _) in GNFR_TO_EDGAR.items():
            print(f"  -> GNFR-{gnfr} {name}")

            s_hour = build_annual_to_hourly_factor(year, gnfr)
            if s_hour is None:
                print(f"     [SKIP] EDGAR monthly files not found for {year} GNFR-{gnfr}")
                continue

            # Aggregations
            s_day = s_hour.resample("D").mean()

            wsum = s_hour.resample(WEEKLY_RULE).sum()
            wcnt = s_hour.resample(WEEKLY_RULE).count()
            s_week = wsum / wcnt  # edge weeks included, no dropping

            s_mon = s_hour.resample("MS").mean()

            results[gnfr] = {
                "name": name,
                "hourly": s_hour,
                "daily": s_day,
                "weekly": s_week,
                "monthly": s_mon,
            }

        # Eğer o yıl hiçbir sektör yoksa → hiçbir şey yazma, patlama yok
        if not results:
            print(f"  [SKIP] No sectors found for {year} → no plots/excels.")
            continue

        # ========== EXCEL WRITERS (artık güvenli) ==========
        xw_hour = pd.ExcelWriter(EXCEL_DIR / f"factors_hourly_{year}.xlsx", engine="openpyxl")
        xw_day  = pd.ExcelWriter(EXCEL_DIR / f"factors_daily_{year}.xlsx", engine="openpyxl")
        xw_week = pd.ExcelWriter(EXCEL_DIR / f"factors_weekly_{year}.xlsx", engine="openpyxl")
        xw_mon  = pd.ExcelWriter(EXCEL_DIR / f"factors_monthly_{year}.xlsx", engine="openpyxl")

        combined_hour = {}
        combined_day  = {}
        combined_week = {}
        combined_mon  = {}

        for gnfr, pack in results.items():
            name = pack["name"]
            label = f"GNFR-{gnfr} {name}"
            sheet = f"GNFR-{gnfr}"

            s_hour = pack["hourly"]
            s_day  = pack["daily"]
            s_week = pack["weekly"]
            s_mon  = pack["monthly"]

            # Excel
            pd.DataFrame({"time": s_hour.index, "factor": s_hour.values}).to_excel(xw_hour, sheet_name=sheet, index=False)
            pd.DataFrame({"time": s_day.index,  "factor": s_day.values}).to_excel(xw_day,  sheet_name=sheet, index=False)
            pd.DataFrame({"time": s_week.index, "factor": s_week.values}).to_excel(xw_week, sheet_name=sheet, index=False)
            pd.DataFrame({"time": s_mon.index,  "factor": s_mon.values}).to_excel(xw_mon,  sheet_name=sheet, index=False)

            # Combined plots dict
            combined_hour[label] = (s_hour.index, s_hour.values)
            combined_day[label]  = (s_day.index,  s_day.values)
            combined_week[label] = (s_week.index, s_week.values)
            combined_mon[label]  = (s_mon.index,  s_mon.values)

            # Per-sector plots
            plot_series(
                s_hour.index, s_hour.values,
                f"Annual→Hourly pure scaling – {year} | {label}",
                PLOT_DIR / "hourly" / "by_sector" / f"hourly_factor_{year}_GNFR-{gnfr}_{name}.png"
            )
            plot_series(
                s_day.index, s_day.values,
                f"Annual→Daily pure scaling – {year} | {label}",
                PLOT_DIR / "daily" / "by_sector" / f"daily_factor_{year}_GNFR-{gnfr}_{name}.png"
            )
            plot_series(
                s_week.index, s_week.values,
                f"Annual→Weekly pure scaling – {year} | {label}",
                PLOT_DIR / "weekly" / "by_sector" / f"weekly_factor_{year}_GNFR-{gnfr}_{name}.png"
            )
            plot_series(
                s_mon.index, s_mon.values,
                f"Annual→Monthly pure scaling – {year} | {label}",
                PLOT_DIR / "monthly" / "by_sector" / f"monthly_factor_{year}_GNFR-{gnfr}_{name}.png"
            )

        # Combined plots
        plot_combined(
            combined_hour,
            f"Annual→Hourly pure scaling – {year} | A,C,J,K",
            PLOT_DIR / "hourly" / "combined" / f"combined_hourly_factor_{year}_A_C_J_K.png"
        )
        plot_combined(
            combined_day,
            f"Annual→Daily pure scaling – {year} | A,C,J,K",
            PLOT_DIR / "daily" / "combined" / f"combined_daily_factor_{year}_A_C_J_K.png"
        )
        plot_combined(
            combined_week,
            f"Annual→Weekly pure scaling – {year} | A,C,J,K",
            PLOT_DIR / "weekly" / "combined" / f"combined_weekly_factor_{year}_A_C_J_K.png"
        )
        plot_combined(
            combined_mon,
            f"Annual→Monthly pure scaling – {year} | A,C,J,K",
            PLOT_DIR / "monthly" / "combined" / f"combined_monthly_factor_{year}_A_C_J_K.png"
        )

        # Close Excel writers
        xw_hour.close()
        xw_day.close()
        xw_week.close()
        xw_mon.close()

        print(f"  [WRITE] Excel + plots written for {year}")

    print("\n[DONE]")
    print(f"Outputs: {OUTDIR}")


if __name__ == "__main__":
    main()
