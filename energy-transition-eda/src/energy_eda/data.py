"""Getting the data: download once, keep a checksum, and refuse a file that does not look like the OWID energy dataset."""
from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"
DEFAULT_PATH = Path("data/owid-energy-data.csv")

REQUIRED_COLUMNS = [
    "country", "year", "iso_code", "population", "gdp", "electricity_generation", "carbon_intensity_elec",
    "low_carbon_share_elec", "fossil_share_elec", "renewables_share_elec", "coal_electricity", "solar_electricity",
    "coal_share_elec", "gas_share_elec", "oil_share_elec", "nuclear_share_elec", "hydro_share_elec", "wind_share_elec",
    "solar_share_elec", "other_renewables_share_elec", "per_capita_electricity", "biofuel_share_elec", "gas_electricity",
    "oil_electricity", "nuclear_electricity", "hydro_electricity", "wind_electricity", "biofuel_electricity",
    "other_renewable_exc_biofuel_electricity",
]


def sha256_of(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(path: str | Path = DEFAULT_PATH, url: str = DATA_URL, force: bool = False, timeout: int = 120) -> Path:
    """Download the dataset unless it is already there. Written atomically; its SHA-256 is recorded next to it."""
    path = Path(path)
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=timeout) as response, open(tmp, "wb") as out:      # noqa: S310 (fixed https URL)
        while chunk := response.read(1 << 20):
            out.write(chunk)
    os.replace(tmp, path)
    path.with_suffix(".sha256").write_text(f"{sha256_of(path)}  {path.name}\n")
    return path


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"this does not look like the OWID energy dataset; missing columns: {', '.join(missing)}")
    if df.duplicated(["country", "year"]).any():
        raise ValueError("duplicate (country, year) rows found")


def load(path: str | Path = DEFAULT_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_schema(df)
    return df
