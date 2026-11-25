"""Figures. One function per chart; each saves a PNG and returns its path. A headless backend, so it runs anywhere."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

from .cleaning import LABELS, MIX_SOURCES, coverage_by_year, gdp_per_capita, wind_solar_share   # noqa: E402

COLORS = {"coal": "#3f3f46", "gas": "#e0883f", "oil": "#8b6b4a", "nuclear": "#8e5ea2", "hydro": "#3a7ca5", "wind": "#4fb3b3",
          "solar": "#f2c14e", "other_renewables": "#7fb069", "wind_solar": "#f2a65a"}
PALETTE = ["#0f766e", "#d97706", "#1d4ed8", "#be123c", "#7c3aed", "#475569", "#0891b2", "#65a30d"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "savefig.bbox": "tight", "figure.dpi": 100})


def _save(fig, path: str | Path) -> Path:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return Path(path)


def plot_coverage(countries: pd.DataFrame, path, years=range(1985, 2026)) -> Path:
    names = {"electricity_generation": "electricity generation", "carbon_intensity_elec": "carbon intensity", "solar_share_elec": "solar share", "gdp": "GDP"}
    cov = coverage_by_year(countries, list(names), years)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for (col, label), colour in zip(names.items(), PALETTE):
        ax.plot(cov.index, cov[col], lw=2, label=label, color=colour)
    ax.set_xlabel("year")
    ax.set_ylabel("countries with a value")
    ax.set_title("How many countries have data? Coverage drops at the recent edge")
    ax.legend(frameon=False, loc="lower right")
    return _save(fig, path)


def plot_world_mix(world: pd.DataFrame, path) -> Path:
    order = ["coal", "oil", "gas", "nuclear", "hydro", "wind", "solar", "other_renewables"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    a.stackplot(world.index, [world[s] for s in order], colors=[COLORS[s] for s in order], labels=[LABELS[s] for s in order], alpha=0.95)
    a.set_title("World electricity generation by source")
    a.set_ylabel("TWh per year")
    a.legend(loc="upper left", frameon=False, fontsize=8, ncol=2)
    b.plot(world.index, world["low_carbon_share_elec"], color="#0f766e", lw=2.2, label="low-carbon share (%)")
    b.plot(world.index, world["fossil_share_elec"], color="#3f3f46", lw=2.2, label="fossil share (%)")
    b.set_ylabel("share of electricity (%)")
    b.set_title("The mix is shifting, slowly")
    twin = b.twinx()
    twin.plot(world.index, world["carbon_intensity_elec"], color="#d97706", lw=2, ls="--", label="carbon intensity (gCO2/kWh)")
    twin.set_ylabel("gCO2 per kWh")
    twin.grid(False)
    twin.spines["right"].set_visible(True)
    l1, l2 = b.get_legend_handles_labels(), twin.get_legend_handles_labels()
    b.legend(l1[0] + l2[0], l1[1] + l2[1], frameon=False, fontsize=8, loc="center left")
    return _save(fig, path)


def plot_coal_share_vs_total(world: pd.DataFrame, path) -> Path:
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4))
    a.plot(world.index, world["coal_share_elec"], color=COLORS["coal"], lw=2.4)
    a.set_title("Coal's SHARE of world electricity")
    a.set_ylabel("%")
    b.plot(world.index, world["coal"], color=COLORS["coal"], lw=2.4)
    peak_year = int(world["coal"].idxmax())
    b.scatter([peak_year], [world["coal"].max()], color="#be123c", zorder=3)
    b.annotate(f"record: {peak_year}\n{world['coal'].max():,.0f} TWh", (peak_year, world["coal"].max()), xytext=(-120, -75),
               textcoords="offset points", arrowprops={"arrowstyle": "->", "color": "#be123c"}, color="#be123c")
    b.set_title("Coal's ABSOLUTE generation")
    b.set_ylabel("TWh")
    fig.suptitle("A falling share is not a falling total: demand grew faster than coal fell", y=1.03)
    return _save(fig, path)


def plot_leaders_laggards(changes: pd.DataFrame, path, n_top: int = 12, n_bottom: int = 8) -> Path:
    shown = pd.concat([changes.head(n_top), changes.tail(n_bottom)])
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.barh(range(len(shown)), shown["change_pp"], color=["#0f766e" if v >= 0 else "#d97706" for v in shown["change_pp"]])
    ax.set_yticks(range(len(shown)), shown.index)
    ax.invert_yaxis()
    for i, (_, r) in enumerate(shown.iterrows()):
        ax.text(r["change_pp"] + (0.8 if r["change_pp"] >= 0 else -0.8), i, f"{r['low_carbon_start']:.0f}% → {r['low_carbon_end']:.0f}%",
                va="center", ha="left" if r["change_pp"] >= 0 else "right", fontsize=8, color="#334155")
    ax.axvline(0, color="#334155", lw=0.8)
    ax.set_xlabel("change in low-carbon share of electricity (percentage points)")
    ax.set_title("Biggest gains and losses in low-carbon electricity")
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.18)
    return _save(fig, path)


def plot_replacements(changes: pd.DataFrame, path, n: int = 10) -> Path:
    top = changes.head(n)
    parts = [("d_coal", "Coal", COLORS["coal"]), ("d_gas", "Gas", COLORS["gas"]), ("d_oil", "Oil", COLORS["oil"]),
             ("d_nuclear", "Nuclear", COLORS["nuclear"]), ("d_hydro", "Hydro", COLORS["hydro"]), ("d_wind_solar", "Wind & solar", COLORS["wind_solar"]),
             ("d_other_renewables", "Bio & other renewables", COLORS["other_renewables"])]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    pos, neg = np.zeros(len(top)), np.zeros(len(top))
    for col, label, colour in parts:
        v = top[col].to_numpy()
        ax.barh(range(len(top)), np.where(v > 0, v, 0), left=pos, color=colour, label=label)
        ax.barh(range(len(top)), np.where(v < 0, v, 0), left=neg, color=colour)
        pos += np.where(v > 0, v, 0)
        neg += np.where(v < 0, v, 0)
    ax.set_yticks(range(len(top)), top.index)
    ax.invert_yaxis()
    ax.axvline(0, color="#334155", lw=0.8)
    ax.set_xlabel("change in share of electricity (percentage points)")
    ax.set_title("What replaced what: the ten biggest decarbonizers")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.3))
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def plot_adoption_curves(countries: pd.DataFrame, adoption: pd.DataFrame, path, n: int = 12, min_twh: float = 20.0, lo: float = 1.0, hi: float = 10.0) -> Path:
    picked = adoption[adoption["reached"] & (adoption["generation_twh"] >= min_twh)].sort_values("years").head(n)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (country, row) in enumerate(picked.iterrows()):
        g = countries[countries["country"] == country].sort_values("year")
        x = g["year"] - row["takeoff_year"]
        y = wind_solar_share(g)
        keep = (x >= -3) & (x <= row["years"] + 4)
        ax.plot(x[keep], y[keep], lw=1.8, color=PALETTE[i % len(PALETTE)], label=f"{country} ({int(row['takeoff_year'])})")
    ax.axhline(lo, color="#94a3b8", ls="--", lw=1)
    ax.axhline(hi, color="#94a3b8", ls="--", lw=1)
    ax.set_xlabel("years since wind + solar first reached 1% of electricity (in brackets: that year)")
    ax.set_ylabel("wind + solar share (%)")
    ax.set_title(f"Fastest takeoffs among grids of {min_twh:.0f}+ TWh: 1% to 10%")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    return _save(fig, path)


def plot_wealth(countries: pd.DataFrame, year: int, wealth: pd.DataFrame, path, min_twh: float = 1.0) -> Path:
    sub = countries[(countries["year"] == year) & (countries["electricity_generation"] >= min_twh)].copy()
    sub["gdp_pc"] = gdp_per_capita(sub)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for ax, col, title, key in [(axes[0], "low_carbon_share_elec", "Low-carbon share of electricity (%)", "Low-carbon"),
                                (axes[1], "per_capita_electricity", "Electricity per person (kWh)", "Electricity per person")]:
        d = sub.dropna(subset=["gdp_pc", col])
        ax.scatter(d["gdp_pc"], d[col], s=22, alpha=0.65, color="#0f766e", edgecolor="white", linewidth=0.4)
        bins = pd.qcut(np.log(d["gdp_pc"]), 8, duplicates="drop")
        med = d.groupby(bins, observed=True).agg(x=("gdp_pc", "median"), y=(col, "median"))
        ax.plot(med["x"], med["y"], color="#be123c", lw=2.2, marker="o", ms=4, label="median per GDP band")
        rho = wealth[wealth["outcome"].str.startswith(key)].iloc[0]
        ax.set_xscale("log")
        ax.set_xlabel(f"GDP per person ({year}, log scale)")
        ax.set_ylabel(title)
        ax.set_title(f"Spearman ρ = {rho['rho']:.2f}  (95% CI {rho['ci_low']:.2f} to {rho['ci_high']:.2f})", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
    return _save(fig, path)


def plot_clusters(result, mix: pd.DataFrame, path, label_n: int = 14) -> Path:
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.7})
    for cid in sorted(result.names):
        members = result.labels[result.labels == cid].index
        a.scatter(result.pca.loc[members, "pc1"], result.pca.loc[members, "pc2"], s=34, alpha=0.8, color=PALETTE[cid % len(PALETTE)],
                  label=f"{result.names[cid]} ({len(members)})", edgecolor="white", linewidth=0.4)
    for country in mix["generation_twh"].nlargest(label_n).index:
        a.annotate(country, result.pca.loc[country, ["pc1", "pc2"]], fontsize=7, xytext=(3, 3), textcoords="offset points")
    a.set_xlabel("principal component 1")
    a.set_ylabel("principal component 2")
    a.set_title(f"Electricity-mix archetypes (k = {result.k}, silhouette {result.silhouette:.2f}, stability {result.stability:.2f})", fontsize=10)
    a.legend(frameon=False, fontsize=8, loc="upper left")
    centers = result.centers.copy()
    left = np.zeros(len(centers))
    for source in MIX_SOURCES:
        b.barh(range(len(centers)), centers[source], left=left, color=COLORS[source], label=LABELS[source])
        left += centers[source].to_numpy()
    b.set_yticks(range(len(centers)), [result.names[i] for i in centers.index])
    b.invert_yaxis()
    b.set_xlabel("average share of electricity (%)")
    b.set_title("What each archetype generates")
    b.legend(frameon=False, fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    b.grid(axis="y", visible=False)
    return _save(fig, path)


def plot_transitions(table: pd.DataFrame, path, year0: int, year1: int) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    im = ax.imshow(table.to_numpy(), cmap="Blues")
    ax.set_xticks(range(table.shape[1]), table.columns, rotation=35, ha="right")
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.grid(False)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.iat[i, j]
            if v:
                ax.text(j, i, int(v), ha="center", va="center", color="white" if v > table.to_numpy().max() * 0.55 else "#0f172a", fontsize=9)
    ax.set_xlabel(f"archetype in {year1}")
    ax.set_ylabel(f"archetype their {year0} mix resembled")
    ax.set_title("Which countries changed archetype? Off-diagonal cells moved")
    fig.colorbar(im, ax=ax, shrink=0.75, label="countries")
    return _save(fig, path)


def plot_intensity(ci: pd.DataFrame, slope: float, intercept: float, rho: float, path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.scatter(ci["change_pp"], ci["ci_change"], s=26, alpha=0.7, color="#0f766e", edgecolor="white", linewidth=0.4)
    xs = np.array([ci["change_pp"].min(), ci["change_pp"].max()])
    ax.plot(xs, slope * xs + intercept, color="#be123c", lw=2, label=f"{slope:.1f} gCO2/kWh per point (ρ = {rho:.2f})")
    drops, rises = list(ci["ci_change"].nsmallest(4).index), list(ci["ci_change"].nlargest(3).index)
    offsets = {c: o for c, o in zip(drops, [(-78, 6), (-70, -16), (8, -14), (-50, -26)])}        # hand-spread so neighbouring labels do not collide
    offsets.update({c: (6, 4) for c in rises})
    for country, offset in offsets.items():
        ax.annotate(country, (ci.loc[country, "change_pp"], ci.loc[country, "ci_change"]), fontsize=7, xytext=offset, textcoords="offset points",
                    arrowprops={"arrowstyle": "-", "color": "#94a3b8", "lw": 0.6})
    ax.axhline(0, color="#94a3b8", lw=0.8)
    ax.axvline(0, color="#94a3b8", lw=0.8)
    ax.set_xlabel("change in low-carbon share (percentage points)")
    ax.set_ylabel("change in carbon intensity (gCO2 per kWh)")
    ax.set_title("Cleaner mix, cleaner kilowatt-hour: 2014 to 2024")
    ax.legend(frameon=False)
    return _save(fig, path)


def plot_coal_peaks(peaks: pd.DataFrame, path, n: int = 12) -> Path:
    d = peaks[peaks["past_peak"]].assign(drop_twh=lambda x: x["peak_twh"] - x["latest_twh"]).sort_values("drop_twh", ascending=False).head(n)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    y = np.arange(len(d))
    ax.barh(y - 0.2, d["peak_twh"], height=0.38, color="#3f3f46", label="at its peak")
    ax.barh(y + 0.2, d["latest_twh"], height=0.38, color="#e0883f", label="latest year")
    ax.set_yticks(y, [f"{c} ({int(r['peak_year'])})" for c, r in d.iterrows()])
    ax.invert_yaxis()
    ax.set_xlabel("coal electricity (TWh)")
    ax.set_title("Countries past peak coal: biggest absolute declines (peak year in brackets)")
    ax.legend(frameon=False)
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def plot_concentration(solar: pd.DataFrame, wind: pd.DataFrame, path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    years = list(solar.index)
    x = np.arange(len(years))
    ax.bar(x - 0.2, solar["top_share"] * 100, width=0.38, color=COLORS["solar"], label="solar")
    ax.bar(x + 0.2, wind.loc[years, "top_share"] * 100, width=0.38, color=COLORS["wind"], label="wind")
    ax.set_xticks(x, years)
    ax.set_ylim(0, 100)
    ax.set_ylabel("share of world generation (%)")
    ax.set_title("Five countries produce about two-thirds of the world's wind and solar")
    ax.legend(frameon=False, loc="upper right", ncol=2)
    return _save(fig, path)
