"""
ERA5 Ensemble Members – Weekly Aggregation Pipeline
=====================================================
Scarica i dati ERA5 ensemble members (10 membri, 3-orari) mese per mese.
Dopo ogni anno completo, calcola le medie settimanali e cancella i GRIB.

Variabili:
  - 2m_temperature      → media settimanale per griglia × membro
  - total_precipitation → cumulata settimanale per griglia × membro

Definizione settimane (fisse, indipendenti dall'anno):
  Settimana k = giorni dell'anno da (k-1)*7+1 a k*7  (k = 1..52)
  I giorni 358-365/366 confluiscono nella settimana 52.

Output: un file NetCDF per anno in era5_weekly/
  era5_weekly/ERA5_ensemble_weekly_{YEAR}.nc
  Dimensioni: (week:52, number:10, latitude, longitude)
  Variabili:  t2m_mean [K], tp_sum [m]

Requisiti:
  pip install cdsapi cfgrib xarray numpy
"""

import cdsapi
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import os

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
# Configurazione
# ---------------------------------------------------------------------------
YEARS  = list(range(2005, 2026))          # 2005 – 2025
MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS   = [f"{d:02d}" for d in range(1, 32)]
TIMES  = [f"{h:02d}:00" for h in range(0, 24, 3)]   # 3-orario: 00,03,...,21
AREA   = [51, -12, 30, 42]                            # N, W, S, E

DATASET   = "reanalysis-era5-single-levels"
VARIABLES = ["2m_temperature", "total_precipitation"]

RAW_DIR    = Path("era5_raw")      # GRIB temporanei (cancellati dopo ogni anno)
WEEKLY_DIR = Path("era5_weekly")   # NetCDF finali, uno per anno
RAW_DIR.mkdir(exist_ok=True)
WEEKLY_DIR.mkdir(exist_ok=True)

client = cdsapi.Client()

# ---------------------------------------------------------------------------
# Utilità
# ---------------------------------------------------------------------------

def week_of_year(dt: pd.Timestamp) -> int:
    """
    Settimana fissa basata sul day-of-year:
      settimana k = giorni [(k-1)*7+1 .. k*7], max 52.
    """
    return min((dt.day_of_year - 1) // 7 + 1, 52)


def grib_path(year: int, month: str) -> Path:
    return RAW_DIR / f"ens_{year}_{month}.grib"


def nc_path(year: int) -> Path:
    return WEEKLY_DIR / f"ERA5_ensemble_weekly_{year}.nc"


# ---------------------------------------------------------------------------
# Step 1 – Download
# ---------------------------------------------------------------------------

def download_month(year: int, month: str) -> Path:
    """
    Scarica un mese di dati ensemble ERA5.
    Se il file esiste già, lo salta (resume-friendly).
    """
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
# Step 2 – Apertura GRIB con cfgrib
# ---------------------------------------------------------------------------

def open_variable(grib_file: Path, short_name: str) -> xr.DataArray:
    """
    Apre un file GRIB e restituisce la DataArray per la variabile richiesta.
    Dimensioni attese da cfgrib: (number, time, latitude, longitude).
    """
    ds = xr.open_dataset(
        str(grib_file),
        engine="cfgrib",
        filter_by_keys={"shortName": short_name},
        backend_kwargs={"errors": "ignore"},
    )
    # cfgrib rinomina le variabili: 2t → t2m, tp → tp
    var = list(ds.data_vars)[0]
    da = ds[var].load()   # carica in memoria, poi chiude il file
    ds.close()
    return da


# ---------------------------------------------------------------------------
# Step 3 – Aggregazione settimanale per un anno
# ---------------------------------------------------------------------------

def aggregate_year(year: int) -> xr.Dataset:
    """
    Carica tutti i 12 GRIB mensili dell'anno, concatena lungo 'time',
    assegna il numero di settimana fissa e aggrega:
      - t2m → media settimanale  (number, week, lat, lon)
      - tp  → somma  settimanale (number, week, lat, lon)

    Restituisce un Dataset con dimensioni (week, number, latitude, longitude).
    """
    t2m_chunks = []
    tp_chunks  = []

    for month in MONTHS:
        grib = grib_path(year, month)
        if not grib.exists():
            log.warning("[WARN] File mancante: %s – il mese verrà saltato", grib.name)
            continue

        log.info("[PROC] Apro %s …", grib.name)
        t2m_chunks.append(open_variable(grib, "2t"))
        tp_chunks.append(open_variable(grib, "tp"))

    if not t2m_chunks:
        raise RuntimeError(f"Nessun dato disponibile per l'anno {year}")

    # Concatena lungo time
    t2m_full = xr.concat(t2m_chunks, dim="time")
    tp_full  = xr.concat(tp_chunks,  dim="time")

    # Calcola il numero di settimana per ogni timestamp
    times_pd = pd.DatetimeIndex(t2m_full["time"].values)
    weeks    = np.array([week_of_year(t) for t in times_pd], dtype=np.int8)

    # Aggiungi coordinata 'week' al dim time
    t2m_full = t2m_full.assign_coords(week=("time", weeks))
    tp_full  = tp_full.assign_coords(week=("time", weeks))

    log.info("[AGG]  %d: aggrego per settimana …", year)

    # Aggrega: mean per t2m, sum per tp  →  (number, week, lat, lon)
    t2m_weekly = t2m_full.groupby("time.week").mean("time")  # usa coord 'week'
    # Nota: groupby("time.week") non esiste nativamente; usiamo il coord custom:
    t2m_weekly = t2m_full.groupby("week").mean("time")
    tp_weekly  = tp_full.groupby("week").sum("time")

    # Garantisce 52 settimane anche se mancano giorni (es. anni con 53 slot)
    all_weeks = np.arange(1, 53)
    t2m_weekly = t2m_weekly.reindex(week=all_weeks)
    tp_weekly  = tp_weekly.reindex(week=all_weeks)

    # Costruisce il Dataset finale
    ds_out = xr.Dataset(
        {
            "t2m_mean": t2m_weekly.rename("t2m_mean"),
            "tp_sum":   tp_weekly.rename("tp_sum"),
        }
    )
    ds_out.attrs = {
        "description": f"ERA5 ensemble members – weekly aggregation – {year}",
        "variables":   "2m_temperature (mean), total_precipitation (sum)",
        "weeks":       "fixed: week k = DOY (k-1)*7+1 to k*7, max 52",
        "area":        "N51 W12 S30 E42",
        "members":     "10 ERA5 ensemble members (number 0–9)",
        "created_by":  "era5_ensemble_weekly.py",
    }

    # Encoding per compressione
    encoding = {
        "t2m_mean": {"dtype": "float32", "zlib": True, "complevel": 4},
        "tp_sum":   {"dtype": "float32", "zlib": True, "complevel": 4},
    }
    out = nc_path(year)
    ds_out.to_netcdf(str(out), encoding=encoding)
    log.info("[SAVE] %s  (%.1f MB)", out.name, out.stat().st_size / 1e6)
    return ds_out


# ---------------------------------------------------------------------------
# Step 4 – Pulizia GRIB anno completato
# ---------------------------------------------------------------------------

def delete_year_gribs(year: int):
    """Cancella i 12 file GRIB mensili dell'anno dopo l'elaborazione."""
    deleted = 0
    for month in MONTHS:
        f = grib_path(year, month)
        if f.exists():
            f.unlink()
            deleted += 1
    log.info("[CLEAN] Cancellati %d file GRIB per l'anno %d", deleted, year)


# ---------------------------------------------------------------------------
# Pipeline principale
# ---------------------------------------------------------------------------

def run():
    log.info("=== ERA5 Ensemble Weekly Pipeline ===")
    log.info("Anni: %d – %d  |  Variabili: %s", YEARS[0], YEARS[-1], VARIABLES)
    log.info("Output: %s", WEEKLY_DIR.resolve())

    for year in YEARS:
        out_nc = nc_path(year)

        # Se il NetCDF annuale esiste già, salta completamente
        if out_nc.exists():
            log.info("[SKIP] Anno %d già elaborato: %s", year, out_nc.name)
            continue

        # --- Download 12 mesi ---
        log.info("--- Anno %d: download ---", year)
        for month in MONTHS:
            try:
                download_month(year, month)
            except Exception as e:
                log.error("[ERR]  Download %d/%s fallito: %s", year, month, e)
                # continua con i mesi successivi

        # --- Verifica che almeno 1 mese sia disponibile ---
        available = [m for m in MONTHS if grib_path(year, m).exists()]
        if not available:
            log.error("[ERR]  Nessun dato per %d, salto l'anno.", year)
            continue

        # --- Aggregazione settimanale ---
        log.info("--- Anno %d: aggregazione settimanale ---", year)
        try:
            aggregate_year(year)
        except Exception as e:
            log.error("[ERR]  Aggregazione %d fallita: %s", year, e)
            continue  # NON cancella i GRIB se l'elaborazione è fallita

        # --- Pulizia GRIB ---
        delete_year_gribs(year)

    log.info("=== Pipeline completata ===")
    log.info("File prodotti:")
    for f in sorted(WEEKLY_DIR.glob("ERA5_ensemble_weekly_*.nc")):
        log.info("  %s  (%.1f MB)", f.name, f.stat().st_size / 1e6)


if __name__ == "__main__":
    run()
