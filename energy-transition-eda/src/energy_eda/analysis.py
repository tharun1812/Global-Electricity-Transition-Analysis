"""The analyses. Each function takes plain DataFrames and returns plain DataFrames or small dataclasses, so each is testable."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

from .cleaning import LABELS, LOW_CARBON, MIX_SOURCES, electricity_mix, gdp_per_capita, wind_solar_share


# ---- 1. who decarbonized? ----------------------------------------------------------------------------------------
def decarbonization_change(countries: pd.DataFrame, y0: int, y1: int, min_twh: float = 5.0) -> pd.DataFrame:
    """Change in the low-carbon share of electricity between two years, with what replaced what (percentage points).

    Only countries that generate at least `min_twh` in BOTH years are compared, so tiny grids do not dominate the ranking.
    The per-source changes add up to zero, because shares always sum to 100.
    """
    a, b = electricity_mix(countries, y0, min_twh), electricity_mix(countries, y1, min_twh)
    both = a.index.intersection(b.index)
    a, b = a.loc[both], b.loc[both]
    out = pd.DataFrame(index=both)
    out["low_carbon_start"] = a[LOW_CARBON].sum(axis=1)
    out["low_carbon_end"] = b[LOW_CARBON].sum(axis=1)
    out["change_pp"] = out["low_carbon_end"] - out["low_carbon_start"]
    d = b[MIX_SOURCES] - a[MIX_SOURCES]
    for source in MIX_SOURCES:
        if source not in ("wind", "solar"):
            out[f"d_{source}"] = d[source]
    out["d_wind_solar"] = d["wind"] + d["solar"]
    out["generation_end_twh"] = b["generation_twh"]
    out.index.name = "country"
    return out.sort_values("change_pp", ascending=False)


def carbon_intensity_change(countries: pd.DataFrame, y0: int, y1: int, changes: pd.DataFrame) -> pd.DataFrame:
    """Add the change in carbon intensity of electricity (gCO2 per kWh) to a decarbonization table."""
    ci = countries.pivot(index="country", columns="year", values="carbon_intensity_elec")
    joined = changes.join(ci[[y0, y1]].rename(columns={y0: "ci_start", y1: "ci_end"}), how="left").dropna(subset=["ci_start", "ci_end"])
    joined["ci_change"] = joined["ci_end"] - joined["ci_start"]
    return joined


# ---- 2. how fast does a new source take off? --------------------------------------------------------------------
def adoption_speed(countries: pd.DataFrame, lo: float = 1.0, hi: float = 10.0, first_year: int = 1985, min_twh: float = 5.0) -> pd.DataFrame:
    """For each country: the year wind+solar first reached `lo`% of electricity, the year it first reached `hi`%, and the gap.

    Countries that reached `lo` but not yet `hi` are 'censored': we only know they have taken at least
    (last observed year - takeoff year) years so far. Countries already above `lo` in the first observed year are excluded,
    and so are grids smaller than `min_twh` (their share can jump from 1% to 10% in a single year with one new solar farm).
    """
    rows = []
    sub = countries[countries["year"] >= first_year].dropna(subset=["wind_share_elec", "solar_share_elec"]).copy()
    sub["ws"] = wind_solar_share(sub)
    for country, g in sub.groupby("country"):
        g = g.sort_values("year")
        if g["ws"].iloc[0] >= lo:
            continue
        above_lo = g[g["ws"] >= lo]
        if above_lo.empty:
            continue
        takeoff = int(above_lo["year"].iloc[0])
        above_hi = g[(g["ws"] >= hi) & (g["year"] >= takeoff)]
        reached = not above_hi.empty
        end_year = int(above_hi["year"].iloc[0]) if reached else int(g["year"].iloc[-1])
        rows.append({"country": country, "takeoff_year": takeoff, "years": end_year - takeoff, "reached": reached,
                     "latest_share": float(g["ws"].iloc[-1]), "generation_twh": float(g["electricity_generation"].iloc[-1])})
    columns = ["takeoff_year", "years", "reached", "latest_share", "generation_twh"]
    if not rows:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(rows).set_index("country")
    out = out[out["generation_twh"] >= min_twh]
    return out.sort_values(["reached", "years"], ascending=[False, True])[columns]


# ---- 3. wealth and the mix ---------------------------------------------------------------------------------------
def spearman_ci(x, y, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float, int]:
    """Spearman rank correlation with a bootstrap confidence interval. Returns (rho, low, high, n)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    n = len(x)
    if n < 5:
        raise ValueError("need at least 5 complete pairs")
    rho = float(stats.spearmanr(x, y)[0])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(x[idx])) > 1 and len(np.unique(y[idx])) > 1:
            boots.append(stats.spearmanr(x[idx], y[idx])[0])
    low, high = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return rho, float(low), float(high), n


def wealth_relationships(countries: pd.DataFrame, year: int, min_twh: float = 1.0, seed: int = 0) -> pd.DataFrame:
    """Rank correlation between GDP per person and several features of a country's electricity, with 95% bootstrap CIs."""
    sub = countries[(countries["year"] == year) & (countries["electricity_generation"] >= min_twh)].copy()
    sub["log_gdp_pc"] = np.log(gdp_per_capita(sub))
    outcomes = {"low_carbon_share_elec": "Low-carbon share of electricity (%)", "renewables_share_elec": "Renewables share (%)",
                "fossil_share_elec": "Fossil share (%)", "carbon_intensity_elec": "Carbon intensity (gCO2/kWh)",
                "per_capita_electricity": "Electricity per person (kWh)"}
    rows = []
    for col, label in outcomes.items():
        rho, lo, hi, n = spearman_ci(sub["log_gdp_pc"], sub[col], seed=seed)
        rows.append({"outcome": label, "rho": rho, "ci_low": lo, "ci_high": hi, "n": n})
    return pd.DataFrame(rows)


# ---- 4. archetypes: clustering electricity mixes -----------------------------------------------------------------
def transform_mix(mix_pct: pd.DataFrame) -> np.ndarray:
    """Square-root of the proportions. Compared to raw shares it stops one huge share (like 90% coal) from dominating the
    distance, so a country with 5% solar is no longer 'identical' to one with 0%."""
    return np.sqrt(mix_pct[MIX_SOURCES].to_numpy() / 100.0)


@dataclass
class ClusterResult:
    k: int
    labels: pd.Series                 # cluster id per country (0 = largest cluster)
    names: dict[int, str]
    centers: pd.DataFrame             # mean mix (%) per cluster
    sizes: pd.Series
    silhouette: float
    stability: float
    scores: pd.DataFrame              # k, silhouette, stability for every k tried
    pca: pd.DataFrame                 # 2-D coordinates for plotting
    model: KMeans
    order: dict[int, int]             # raw KMeans label -> ordered cluster id

    def predict(self, mix_pct: pd.DataFrame) -> pd.Series:
        raw = self.model.predict(transform_mix(mix_pct))
        return pd.Series([self.order[r] for r in raw], index=mix_pct.index, name="cluster")


def _stability(X: np.ndarray, k: int, full_labels: np.ndarray, seed: int, n_boot: int) -> float:
    """Mean adjusted Rand index between the full clustering and clusterings of random 80% subsamples (1 = perfectly stable)."""
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_boot):
        idx = rng.choice(len(X), size=int(0.8 * len(X)), replace=False)
        sub = KMeans(n_clusters=k, n_init=10, random_state=int(rng.integers(0, 2**31 - 1))).fit(X[idx])
        scores.append(adjusted_rand_score(full_labels[idx], sub.labels_))
    return float(np.mean(scores))


def name_cluster(center: pd.Series) -> str:
    """Name a cluster after its dominant source(s), straight from the numbers."""
    ranked = center[MIX_SOURCES].sort_values(ascending=False)
    top, second = ranked.index[0], ranked.index[1]
    if ranked.iloc[0] >= 50:
        return f"{LABELS[top]}-dominated"
    return f"{LABELS[top]} + {LABELS[second]}"


def fit_clusters(mix: pd.DataFrame, k: int | None = None, k_range: range = range(4, 9), seed: int = 0, n_boot: int = 25,
                 min_stability: float = 0.7) -> ClusterResult:
    """Cluster countries by electricity mix.

    If k is not given it is chosen from k_range: among the values whose stability is at least `min_stability`, take the one
    with the best silhouette (if none is stable enough, take the most stable). The range starts at 4 because splits into
    2 or 3 groups only separate 'fossil' from 'hydro' and are too coarse to be informative; every k is reported.
    """
    if len(mix) < max(k_range) + 2 and k is None:
        raise ValueError("too few countries for the requested range of k")
    X = transform_mix(mix)
    rows, fits = [], {}
    for kk in ([k] if k else list(k_range)):
        model = KMeans(n_clusters=kk, n_init=20, random_state=seed).fit(X)
        sil = float(silhouette_score(X, model.labels_)) if kk > 1 else float("nan")
        rows.append({"k": kk, "silhouette": sil, "stability": _stability(X, kk, model.labels_, seed, n_boot)})
        fits[kk] = model
    scores = pd.DataFrame(rows)
    if k is None:
        stable = scores[scores["stability"] >= min_stability]
        pool, key = (stable, "silhouette") if len(stable) else (scores, "stability")
        chosen = int(pool.sort_values(key, ascending=False).iloc[0]["k"])
    else:
        chosen = k
    model = fits[chosen]

    sizes_raw = pd.Series(model.labels_).value_counts()
    order = {int(raw): i for i, raw in enumerate(sizes_raw.index)}                  # cluster 0 is the biggest
    labels = pd.Series([order[int(r)] for r in model.labels_], index=mix.index, name="cluster")
    centers = mix[MIX_SOURCES].groupby(labels).mean()
    names, seen = {}, {}
    for cid, center in centers.iterrows():
        base = name_cluster(center)
        seen[base] = seen.get(base, 0) + 1
        names[int(cid)] = base if seen[base] == 1 else f"{base} ({seen[base]})"
    coords = PCA(n_components=2, random_state=seed).fit_transform(X)
    row = scores[scores["k"] == chosen].iloc[0]
    return ClusterResult(chosen, labels, names, centers, labels.value_counts().sort_index(), float(row["silhouette"]), float(row["stability"]),
                         scores, pd.DataFrame(coords, index=mix.index, columns=["pc1", "pc2"]), model, order)


def cluster_transitions(result: ClusterResult, mix_before: pd.DataFrame, mix_after_labels: pd.Series) -> pd.DataFrame:
    """Where did each country's earlier mix sit, relative to today's clusters? Rows: earlier cluster; columns: today's cluster."""
    before = result.predict(mix_before)
    both = before.index.intersection(mix_after_labels.index)
    table = pd.crosstab(before.loc[both].map(result.names), mix_after_labels.loc[both].map(result.names))
    table.index.name, table.columns.name = "before", "now"
    return table


# ---- 5. coal -----------------------------------------------------------------------------------------------------
def coal_peaks(countries: pd.DataFrame, first_year: int = 1985, last_year: int | None = None, drop: float = 0.25,
               min_peak_twh: float = 1.0) -> pd.DataFrame:
    """Peak year of coal power per country, and whether generation has since fallen by at least `drop` (default 25%)."""
    sub = countries[(countries["year"] >= first_year)].dropna(subset=["coal_electricity"])
    last_year = last_year or int(sub["year"].max())
    sub = sub[sub["year"] <= last_year]
    rows = []
    for country, g in sub.groupby("country"):
        if g["year"].max() < last_year - 1:
            continue                                                       # no recent data: cannot say whether it has fallen
        peak = g.loc[g["coal_electricity"].idxmax()]
        if peak["coal_electricity"] < min_peak_twh:
            continue
        latest = g.sort_values("year").iloc[-1]["coal_electricity"]
        decline = 1 - latest / peak["coal_electricity"]
        rows.append({"country": country, "peak_year": int(peak["year"]), "peak_twh": float(peak["coal_electricity"]),
                     "latest_twh": float(latest), "decline": float(decline), "past_peak": bool(decline >= drop and peak["year"] < last_year)})
    return pd.DataFrame(rows).set_index("country").sort_values("decline", ascending=False)


def concentration(countries: pd.DataFrame, column: str, top_n: int = 5, years: list[int] | None = None) -> pd.DataFrame:
    """Share of the world total of `column` produced by the `top_n` largest countries, per year."""
    pivot = countries.pivot(index="year", columns="country", values=column)
    if years is not None:
        pivot = pivot.loc[[y for y in years if y in pivot.index]]
    total = pivot.sum(axis=1)
    top = pivot.apply(lambda row: row.nlargest(top_n).sum(), axis=1)
    names = pivot.apply(lambda row: ", ".join(row.nlargest(top_n).index), axis=1)
    return pd.DataFrame({"top_share": top / total, "top_countries": names, "world_total": total}).dropna()


# ---- 6. the world as a whole -------------------------------------------------------------------------------------
def world_series(df: pd.DataFrame, first_year: int = 1985) -> pd.DataFrame:
    """Global generation (TWh) by source, plus the headline ratios, from OWID's own 'World' row."""
    w = df[df["country"] == "World"].set_index("year").sort_index().loc[first_year:]
    out = pd.DataFrame({s: w[f"{s}_electricity"] for s in ("coal", "gas", "oil", "nuclear", "hydro", "wind", "solar")})
    out["other_renewables"] = w["biofuel_electricity"] + w["other_renewable_exc_biofuel_electricity"]
    out["total"] = w["electricity_generation"]
    for col in ("low_carbon_share_elec", "fossil_share_elec", "coal_share_elec", "carbon_intensity_elec", "wind_share_elec", "solar_share_elec"):
        out[col] = w[col]
    return out.dropna(subset=["total"])
