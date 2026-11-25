"""Cleaning: separate real countries from aggregates, and build a complete, non-overlapping electricity mix."""
from __future__ import annotations

import pandas as pd

# OWID's `other_renewables_share_elec` already INCLUDES biofuel, so `biofuel_share_elec` must not be added as well:
# doing so double-counts and makes the shares add up to more than 100% for about a quarter of countries.
MIX_SOURCES = ["coal", "gas", "oil", "nuclear", "hydro", "wind", "solar", "other_renewables"]
FOSSIL = ["coal", "gas", "oil"]
LOW_CARBON = ["nuclear", "hydro", "wind", "solar", "other_renewables"]
MIX_COLUMNS = [f"{s}_share_elec" for s in MIX_SOURCES]
LABELS = {"coal": "Coal", "gas": "Gas", "oil": "Oil", "nuclear": "Nuclear", "hydro": "Hydro", "wind": "Wind", "solar": "Solar",
          "other_renewables": "Bio & other renewables"}
SHARE_TOLERANCE = 0.5          # percentage points allowed when checking that a mix adds up to 100


def split_entities(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Countries have an ISO code. Regions ('Europe (Ember)'), income groups, 'World' and historical entities do not.

    Summing or ranking the two together would double-count, so analyses use the countries only; the 'World' row is kept
    separately for global totals.
    """
    has_iso = df["iso_code"].notna()
    return df[has_iso].copy(), df[~has_iso].copy()


def coverage_by_year(countries: pd.DataFrame, columns: list[str], years: range | list[int]) -> pd.DataFrame:
    """How many countries have a value, per indicator and year. Used to choose analysis years honestly."""
    sub = countries[countries["year"].isin(list(years))]
    return sub.groupby("year")[columns].count().astype(int)


def _with_totals(countries: pd.DataFrame, min_twh: float) -> pd.DataFrame:
    rows = countries.dropna(subset=["electricity_generation", *MIX_COLUMNS])
    rows = rows[rows["electricity_generation"] >= min_twh].copy()
    rows["share_total"] = rows[MIX_COLUMNS].sum(axis=1)
    return rows


def inconsistent_share_rows(countries: pd.DataFrame, min_twh: float = 5.0) -> pd.DataFrame:
    """Country-years whose source shares do not add up to 100% (within 0.5 points). In the real data they are all pre-2000."""
    rows = _with_totals(countries, min_twh)
    return rows[(rows["share_total"] - 100).abs() > SHARE_TOLERANCE][["country", "year", "share_total"]].reset_index(drop=True)


def electricity_mix(countries: pd.DataFrame, year: int, min_twh: float = 5.0, on_mismatch: str = "drop") -> pd.DataFrame:
    """One row per country: shares (%) of each source in that year, plus total generation. Tiny grids are dropped.

    Every later step assumes a mix adds up to 100%. Rows that do not are dropped and listed in `result.attrs["dropped"]`,
    or raise a ValueError when on_mismatch="raise".
    """
    if on_mismatch not in ("drop", "raise"):
        raise ValueError("on_mismatch must be 'drop' or 'raise'")
    rows = _with_totals(countries[countries["year"] == year], min_twh)
    off = (rows["share_total"] - 100).abs() > SHARE_TOLERANCE
    if off.any() and on_mismatch == "raise":
        raise ValueError(f"electricity shares do not add up to 100% for {rows.loc[off, 'country'].iloc[0]} in {year}")
    dropped = sorted(rows.loc[off, "country"])
    rows = rows[~off]
    mix = rows.set_index("country")[MIX_COLUMNS].copy()
    mix.columns = MIX_SOURCES
    mix["generation_twh"] = rows.set_index("country")["electricity_generation"]
    mix = mix.sort_index()
    mix.attrs["dropped"] = dropped
    return mix


def double_count_check(countries: pd.DataFrame, year: int, min_twh: float = 5.0) -> tuple[int, int, float]:
    """The trap in the raw data: adding the separate biofuel share to the eight sources double-counts biofuel.

    Returns (countries checked, countries whose shares then exceed 100%, the largest total reached).
    """
    rows = countries[(countries["year"] == year) & (countries["electricity_generation"] >= min_twh)].dropna(subset=[*MIX_COLUMNS, "biofuel_share_elec"])
    naive = rows[MIX_COLUMNS].sum(axis=1) + rows["biofuel_share_elec"]
    return len(rows), int((naive > 100 + SHARE_TOLERANCE).sum()), float(naive.max())


def gdp_per_capita(df: pd.DataFrame) -> pd.Series:
    """GDP per person (international-$, 2011 prices). NaN where either input is missing or population is zero."""
    pop = df["population"].where(df["population"] > 0)
    return df["gdp"] / pop


def wind_solar_share(df: pd.DataFrame) -> pd.Series:
    return df["wind_share_elec"] + df["solar_share_elec"]
