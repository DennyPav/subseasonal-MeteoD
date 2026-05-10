"""
ERA5 Ensemble Members – Weekly Aggregation Pipeline
=====================================================
Variabili d'ambiente richieste:
  CDSAPI_URL  →  https://cds.climate.copernicus.eu/api
  CDSAPI_KEY  →  il tuo UUID dal profilo Copernicus CDS
  ERA5_YEAR   →  anno da elaborare (es. "2005")
                 se non impostato, elabora tutti gli anni 2005-2025
"""

import os
import cdsapi
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Credenziali
# ---------------------------------------------------------------------------
CDSAPI_URL = os.environ.get("CDSAPI_URL")
CDSAPI_KEY = os.environ.get("CDSAPI_KEY")

if not CDSAPI_URL or not CDSAPI_KEY:
    raise EnvironmentError(
        "Variabili d'ambiente mancanti: CDSAPI_URL e CDSAPI_KEY\n"
        "Impostale prima di eseguire lo script:\n"
        "  export CDSAPI_URL='https://cds.climate.copernicus.eu/api'\n"
        "  export CDSAPI_KEY='il-tuo-uuid-key'"
    )

# ---------------------------------------------------------------------------
# Anno target: da env ERA5_YEAR oppure tutti
# ---------------------------------------------------------------------------
_env_year = os.environ.get("ERA5_YEAR")
if _env_year:
    YEARS = [int(_env_year)]
    log.info("Modalità singolo anno: %d", YEARS[0])
else:
    YEARS = list(range(2005, 2026))
    log.info("Modalità tutti gli anni: %d – %d", YEARS[0], YEARS[-1])

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------
MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS   = [f"{d:02d}" for d in range(1, 32)]
TIMES  = [f"{h:02d}:00" for h in range(0, 24, 3)]
AREA   = [51, -12, 30, 42]   # N, W, S, E

DATASET   = "reanalysis-era5-single-levels"
VARIABLES = ["2m_temperature", "total_precipitation"]

RAW_DIR    = Path("era5_raw")
WEEKLY_DIR = Path("era5_weekly")
RAW_DIR.mkdir(exist_ok=True)
WEEKLY_DIR.mkdir(exist_ok=True)

client = cdsapi.Client(url=CDSAPI_URL, key=CDSAPI_KEY)

# ---------------------------------------------------------------------------
# Utilità
# ---------------------------------------------------------------------------

def week_of_year(dt: pd.Timestamp) -> int:
    """Settimana fissa: week k = DOY (k-1)*7+1 .. k*7, max 52."""
    return min((dt.day_of_year - 1) // 7 + 1, 52)

def grib_path(year: int, month: str) -> Path:
    return RAW_DIR / f"ens_{year}_{month}.grib"

def nc_path(year: int) -> Path:
    return WEEKLY_DIR / f"ERA5_ensemble_weekly_{year}.nc"

# ---------------------------------------------------------------------------
# Step 1 – Download mensile
# ---------------------------------------------------------------------------

def download_month(year: int, month: str) -> Path:
    out = grib_path(year, month)
    if out.exists():
        log.info("[SKIP] Già presente: %s", out.name)
        return out

    request = {
        "product_type": ["ensemble_members"],
        "variable": VARIABLES,
        "year":  [str(year)],
        "month": [month],
        "day":   DAYS,
        "time":  TIMES,
        "data_format":     "grib",
        "download_format": "unarchived",
        "area": AREA,
    }
    log.info("[DL]   %d/%s …", year, month)
    client.retrieve(DATASET, request).download(str(out))
    log.info("[OK]   Salvato: %s  (%.1f MB)", out.name, out.stat().st_size / 1e6)
    return out

# ---------------------------------------------------------------------------
# Step 2 – Apertura GRIB
# ---------------------------------------------------------------------------

def open_variable(grib_file: Path, short_name: str) -> xr.DataArray:
    ds = xr.open_dataset(
        str(grib_file),
        engine="cfgrib",
        filter_by_keys={"shortName": short_name},
        backend_kwargs={"errors": "ignore"},
    )
    var = list(ds.data_vars)[0]
    da = ds[var].load()
    ds.close()
    return da

# ---------------------------------------------------------------------------
# Step 3 – Aggregazione settimanale
# ---------------------------------------------------------------------------

def aggregate_year(year: int) -> None:
    t2m_chunks, tp_chunks = [], []

    for month in MONTHS:
        grib = grib_path(year, month)
        if not grib.exists():
            log.warning("[WARN] File mancante: %s – mese saltato", grib.name)
            continue
        log.info("[PROC] Apro %s …", grib.name)
        t2m_chunks.append(open_variable(grib, "2t"))
        tp_chunks.append(open_variable(grib, "tp"))

    if not t2m_chunks:
        raise RuntimeError(f"Nessun dato disponibile per l'anno {year}")

    t2m_full = xr.concat(t2m_chunks, dim="time")
    tp_full  = xr.concat(tp_chunks,  dim="time")

    times_pd = pd.DatetimeIndex(t2m_full["time"].values)
    weeks    = np.array([week_of_year(t) for t in times_pd], dtype=np.int8)

    t2m_full = t2m_full.assign_coords(week=("time", weeks))
    tp_full  = tp_full.assign_coords(week=("time", weeks))

    log.info("[AGG]  %d: aggrego per settimana …", year)
    t2m_weekly = t2m_full.groupby("week").mean("time")
    tp_weekly  = tp_full.groupby("week").sum("time")

    all_weeks  = np.arange(1, 53)
    t2m_weekly = t2m_weekly.reindex(week=all_weeks)
    tp_weekly  = tp_weekly.reindex(week=all_weeks)

    ds_out = xr.Dataset({
        "t2m_mean": t2m_weekly.rename("t2m_mean"),
        "tp_sum":   tp_weekly.rename("tp_sum"),
    })
    ds_out.attrs = {
        "description": f"ERA5 ensemble members – weekly aggregation – {year}",
        "variables":   "2m_temperature (mean), total_precipitation (sum)",
        "weeks":       "fixed: week k = DOY (k-1)*7+1 to k*7, max 52",
        "area":        "N51 W12 S30 E42",
        "members":     "10 ERA5 ensemble members (number 0–9)",
        "created_by":  "era5_ensemble_weekly.py",
    }
    encoding = {
        "t2m_mean": {"dtype": "float32", "zlib": True, "complevel": 4},
        "tp_sum":   {"dtype": "float32", "zlib": True, "complevel": 4},
    }
    out = nc_path(year)
    ds_out.to_netcdf(str(out), encoding=encoding)
    log.info("[SAVE] %s  (%.1f MB)", out.name, out.stat().st_size / 1e6)

# ---------------------------------------------------------------------------
# Step 4 – Pulizia GRIB
# ---------------------------------------------------------------------------

def delete_year_gribs(year: int):
    deleted = 0
    for month in MONTHS:
        f = grib_path(year, month)
        if f.exists():
            f.unlink()
            deleted += 1
    log.info("[CLEAN] Cancellati %d GRIB per il %d", deleted, year)

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run():
    log.info("=== ERA5 Ensemble Weekly Pipeline ===")

    for year in YEARS:
        if nc_path(year).exists():
            log.info("[SKIP] Anno %d già elaborato.", year)
            continue

        log.info("━━━ Anno %d: download ━━━", year)
        for month in MONTHS:
            try:
                download_month(year, month)
            except Exception as e:
                log.error("[ERR]  Download %d/%s: %s", year, month, e)

        available = [m for m in MONTHS if grib_path(year, m).exists()]
        if not available:
            log.error("[ERR]  Nessun GRIB per %d, anno saltato.", year)
            continue

        log.info("━━━ Anno %d: aggregazione ━━━", year)
        try:
            aggregate_year(year)
            delete_year_gribs(year)
        except Exception as e:
            log.error("[ERR]  Aggregazione %d: %s", year, e)

    log.info("=== Pipeline completata ===")

if __name__ == "__main__":
    run()
