"""Run the whole analysis, collect the headline numbers, and write the figures and the findings document.

Every number in reports/insights.md is computed here from the data; nothing is typed in by hand.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis as an
from . import cleaning as cl
from . import plots


@dataclass
class Analysis:
    df: pd.DataFrame
    countries: pd.DataFrame
    aggregates: pd.DataFrame
    world: pd.DataFrame
    y0: int
    y1: int
    gdp_year: int
    min_twh: float
    mix0: pd.DataFrame
    mix1: pd.DataFrame
    changes: pd.DataFrame
    intensity: pd.DataFrame
    adoption: pd.DataFrame
    wealth: pd.DataFrame
    clusters: an.ClusterResult
    transitions: pd.DataFrame
    coal: pd.DataFrame
    solar_conc: pd.DataFrame
    wind_conc: pd.DataFrame
    inconsistent: pd.DataFrame
    seed: int


def run_analysis(df: pd.DataFrame, y0: int = 2014, y1: int = 2024, gdp_year: int = 2022, min_twh: float = 5.0, seed: int = 0,
                 k_range: range = range(4, 9)) -> Analysis:
    countries, aggregates = cl.split_entities(df)
    mix0, mix1 = cl.electricity_mix(countries, y0, min_twh), cl.electricity_mix(countries, y1, min_twh)
    changes = an.decarbonization_change(countries, y0, y1, min_twh)
    clusters = an.fit_clusters(mix1, k_range=k_range, seed=seed)
    return Analysis(
        df=df, countries=countries, aggregates=aggregates, world=an.world_series(df), y0=y0, y1=y1, gdp_year=gdp_year, min_twh=min_twh,
        mix0=mix0, mix1=mix1, changes=changes, intensity=an.carbon_intensity_change(countries, y0, y1, changes),
        adoption=an.adoption_speed(countries, min_twh=min_twh), wealth=an.wealth_relationships(countries, gdp_year, seed=seed),
        clusters=clusters, transitions=an.cluster_transitions(clusters, mix0, clusters.labels), coal=an.coal_peaks(countries, last_year=y1),
        solar_conc=an.concentration(countries, "solar_electricity", 5, [y0, (y0 + y1) // 2, y1]),
        wind_conc=an.concentration(countries, "wind_electricity", 5, [y0, (y0 + y1) // 2, y1]),
        inconsistent=cl.inconsistent_share_rows(countries, min_twh), seed=seed)


def _wind_solar_leader(a: Analysis) -> dict:
    ws = (a.clusters.centers["wind"] + a.clusters.centers["solar"]).sort_values(ascending=False)
    return {"name": a.clusters.names[int(ws.index[0])], "share": round(float(ws.iloc[0]))}


def _rich_spread(a: Analysis) -> dict:
    """How different are the electricity mixes of the richest countries? (10th to 90th percentile of low-carbon share.)"""
    sub = a.countries[(a.countries["year"] == a.gdp_year) & (a.countries["electricity_generation"] >= 1.0)].copy()
    sub["gdp_pc"] = cl.gdp_per_capita(sub)
    sub = sub.dropna(subset=["gdp_pc", "low_carbon_share_elec"])
    rich = sub[sub["gdp_pc"] >= sub["gdp_pc"].quantile(0.75)]
    p10, p90 = rich["low_carbon_share_elec"].quantile([0.1, 0.9])
    return {"n": int(len(rich)), "p10": round(float(p10)), "p90": round(float(p90))}


def compute_facts(a: Analysis) -> dict:
    w, y0, y1 = a.world, a.y0, a.y1
    c = a.countries
    n_checked, n_over, max_total = cl.double_count_check(c, y1, a.min_twh)
    reached = a.adoption[a.adoption["reached"]]
    ch = a.changes
    slope, intercept = np.polyfit(a.intensity["change_pp"], a.intensity["ci_change"], 1)
    rho, lo, hi, n_ci = an.spearman_ci(a.intensity["change_pp"], a.intensity["ci_change"], seed=a.seed)
    cp = a.coal
    past = cp[cp["past_peak"]]
    members = {a.clusters.names[cid]: list(a.mix1.loc[a.clusters.labels[a.clusters.labels == cid].index, "generation_twh"].nlargest(4).index)
               for cid in sorted(a.clusters.names)}
    cov = lambda col, y: int(c[(c["year"] == y)][col].notna().sum())
    diag = sum(int(a.transitions.loc[n, n]) for n in a.transitions.index if n in a.transitions.columns)
    return {
        "years": {"start": y0, "end": y1, "gdp_year": a.gdp_year, "min_twh": a.min_twh},
        "data": {
            "rows": int(len(a.df)), "entities": int(a.df["country"].nunique()), "countries": int(c["country"].nunique()),
            "aggregates": int(a.aggregates["country"].nunique()), "first_year": int(a.df["year"].min()), "last_year": int(a.df["year"].max()),
            "countries_with_electricity": {str(y): cov("electricity_generation", y) for y in (y1 - 1, y1, y1 + 1)},
            "countries_with_gdp_last_year": cov("gdp", a.gdp_year), "last_gdp_year": int(c.dropna(subset=["gdp"])["year"].max()),
            "double_count": {"checked": n_checked, "over_100": n_over, "max_total": round(max_total, 1)},
            "inconsistent_rows": int(len(a.inconsistent)),
            "inconsistent_years": [int(a.inconsistent["year"].min()), int(a.inconsistent["year"].max())] if len(a.inconsistent) else None,
            "countries_compared": int(len(ch)),
        },
        "world": {
            "low_carbon_start": round(float(w.loc[y0, "low_carbon_share_elec"]), 1), "low_carbon_end": round(float(w.loc[y1, "low_carbon_share_elec"]), 1),
            "coal_share_start": round(float(w.loc[y0, "coal_share_elec"]), 1), "coal_share_end": round(float(w.loc[y1, "coal_share_elec"]), 1),
            "intensity_start": round(float(w.loc[y0, "carbon_intensity_elec"])), "intensity_end": round(float(w.loc[y1, "carbon_intensity_elec"])),
            "generation_start": round(float(w.loc[y0, "total"])), "generation_end": round(float(w.loc[y1, "total"])),
            "wind_solar_start": round(float(w.loc[y0, "wind_share_elec"] + w.loc[y0, "solar_share_elec"]), 1),
            "wind_solar_end": round(float(w.loc[y1, "wind_share_elec"] + w.loc[y1, "solar_share_elec"]), 1),
            "coal_twh_end": round(float(w.loc[y1, "coal"])), "coal_twh_start": round(float(w.loc[y0, "coal"])),
            "coal_peak_year": int(w["coal"].idxmax()), "coal_peak_twh": round(float(w["coal"].max())),
            "coal_growth_pct": round((float(w.loc[y1, "coal"]) / float(w.loc[y0, "coal"]) - 1) * 100, 1),
            "generation_growth_pct": round((float(w.loc[y1, "total"]) / float(w.loc[y0, "total"]) - 1) * 100),
        },
        "changes": {
            "gained_10pp": int((ch["change_pp"] >= 10).sum()), "lost": int((ch["change_pp"] < 0).sum()),
            "top": [{"country": k, "start": round(float(r.low_carbon_start), 1), "end": round(float(r.low_carbon_end), 1), "change": round(float(r.change_pp), 1)}
                    for k, r in ch.head(6).iterrows()],
            "bottom": [{"country": k, "start": round(float(r.low_carbon_start), 1), "end": round(float(r.low_carbon_end), 1), "change": round(float(r.change_pp), 1)}
                       for k, r in ch.tail(4).iloc[::-1].iterrows()],
        },
        "intensity": {"n": int(n_ci), "slope": round(float(slope), 1), "intercept": round(float(intercept), 1), "rho": round(rho, 2), "ci": [round(lo, 2), round(hi, 2)]},
        "adoption": {
            "took_off": int(len(a.adoption)), "reached_10": int(len(reached)), "censored": int((~a.adoption["reached"]).sum()),
            "median_years": float(reached["years"].median()), "q1": float(reached["years"].quantile(0.25)), "q3": float(reached["years"].quantile(0.75)),
            "recent_median": float(reached[reached["takeoff_year"] >= 2015]["years"].median()), "recent_n": int((reached["takeoff_year"] >= 2015).sum()),
            "early_median": float(reached[reached["takeoff_year"] < 2010]["years"].median()), "early_n": int((reached["takeoff_year"] < 2010).sum()),
        },
        "wealth": [{k: (round(float(v), 2) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()} for r in a.wealth.to_dict("records")],
        "rich": _rich_spread(a),
        "clusters": {
            "k": int(a.clusters.k), "silhouette": round(a.clusters.silhouette, 3), "stability": round(a.clusters.stability, 3),
            "scores": [{"k": int(r.k), "silhouette": round(float(r.silhouette), 3), "stability": round(float(r.stability), 3)} for r in a.clusters.scores.itertuples()],
            "sizes": {a.clusters.names[i]: int(n) for i, n in a.clusters.sizes.items()}, "members": members,
            "most_wind_solar": _wind_solar_leader(a),
            "moved": int(a.transitions.to_numpy().sum() - diag), "transition_total": int(a.transitions.to_numpy().sum()),
        },
        "coal": {
            "countries": int(len(cp)), "past_peak": int(len(past)), "past_peak_before_2015": int((past["peak_year"] < 2015).sum()),
            "biggest": [{"country": k, "peak_year": int(r.peak_year), "peak": round(float(r.peak_twh)), "latest": round(float(r.latest_twh))}
                        for k, r in past.assign(d=lambda x: x["peak_twh"] - x["latest_twh"]).sort_values("d", ascending=False).head(4).iterrows()],
        },
        "concentration": {
            "solar_start": round(float(a.solar_conc.iloc[0]["top_share"]) * 100), "solar_end": round(float(a.solar_conc.iloc[-1]["top_share"]) * 100),
            "solar_start_countries": a.solar_conc.iloc[0]["top_countries"], "solar_countries": a.solar_conc.iloc[-1]["top_countries"],
            "wind_end": round(float(a.wind_conc.iloc[-1]["top_share"]) * 100), "wind_countries": a.wind_conc.iloc[-1]["top_countries"],
        },
    }


def render_insights(f: dict) -> str:
    y0, y1, w, d = f["years"]["start"], f["years"]["end"], f["world"], f["data"]
    wealth = {r["outcome"].split(" (")[0]: r for r in f["wealth"]}
    lc, per = wealth["Low-carbon share of electricity"], wealth["Electricity per person"]
    verdict = "no clear relationship" if lc["ci_low"] <= 0 <= lc["ci_high"] else ("a weak relationship" if abs(lc["rho"]) < 0.4 else "a clear relationship")
    top, bottom, ci, ad, cl_, coal, con = f["changes"]["top"], f["changes"]["bottom"], f["intensity"], f["adoption"], f["clusters"], f["coal"], f["concentration"]
    lines = [
        "# Findings", "",
        f"*Generated from the data by `energy_eda`. Electricity only. Years compared: {y0} and {y1}; countries with at least {f['years']['min_twh']:g} TWh of generation.*", "",
        "## 1. The world is greening its electricity, but coal is still setting records", "",
        f"- The low-carbon share of world electricity rose from **{w['low_carbon_start']}% to {w['low_carbon_end']}%** between {y0} and {y1}, and wind plus solar from {w['wind_solar_start']}% to {w['wind_solar_end']}%.",
        f"- Carbon intensity fell from {w['intensity_start']} to **{w['intensity_end']} gCO2 per kWh**.",
        f"- Coal's share fell from {w['coal_share_start']}% to {w['coal_share_end']}%, **yet coal generation hit its record in {w['coal_peak_year']}: {w['coal_peak_twh']:,} TWh** (up {w['coal_growth_pct']}% from {w['coal_twh_start']:,} TWh in {y0}). Total generation grew {w['generation_growth_pct']}% over the decade, so coal's output rose even as its share fell.", "",
        "## 2. Who decarbonized", "",
        f"- Of {d['countries_compared']} countries compared, **{f['changes']['gained_10pp']} gained 10 or more points** of low-carbon share and **{f['changes']['lost']} went backwards**.",
        "- Biggest gains: " + "; ".join(f"{r['country']} ({r['start']}% → {r['end']}%)" for r in top) + ".",
        "- Biggest losses: " + "; ".join(f"{r['country']} ({r['start']}% → {r['end']}%)" for r in bottom) + ".",
        f"- A cleaner mix means a cleaner kilowatt-hour: across {ci['n']} countries each extra point of low-carbon share went with **{ci['slope']} gCO2/kWh** (Spearman ρ = {ci['rho']}, 95% CI {ci['ci'][0]} to {ci['ci'][1]}).", "",
        "## 3. How fast can a new source take off", "",
        f"- {ad['took_off']} grids of meaningful size moved wind plus solar past 1% of their electricity; {ad['reached_10']} have since passed 10% (the rest are still on the way). The median time from 1% to 10% was **{ad['median_years']:g} years** (middle half: {ad['q1']:g} to {ad['q3']:g}).",
        f"- Later starters look quicker: takeoffs since 2015 reached 10% in a median **{ad['recent_median']:g} years** (n = {ad['recent_n']}), against {ad['early_median']:g} years for takeoffs before 2010 (n = {ad['early_n']}). Recent takeoffs are, by construction, still being observed, so the recent figure only counts the fast ones.", "",
        "## 4. Wealth does not buy a clean grid", "",
        f"- Across countries in {f['years']['gdp_year']}, GDP per person shows {verdict} with the low-carbon share of electricity (Spearman ρ = {lc['rho']}, 95% CI {lc['ci_low']} to {lc['ci_high']}), but a strong one with how much electricity each person uses (ρ = {per['rho']}).",
        f"- Among the richest quarter of countries ({f['rich']['n']} of them) the low-carbon share ranges from {f['rich']['p10']}% to {f['rich']['p90']}% (10th to 90th percentile): wealth explains *how much* electricity a country uses, not *how clean* it is.", "",
        "## 5. Country archetypes", "",
        f"- Clustering the {y1} electricity mixes gives **{cl_['k']} archetypes** (silhouette {cl_['silhouette']}, stability {cl_['stability']}; stability is the average agreement between clusterings of random 80% subsamples, 1 = identical).",
        *[f"  - **{name}**: {cl_['sizes'][name]} countries, including {', '.join(m)}." for name, m in cl_["members"].items()],
        f"- The archetype that leans most on wind and solar is **{cl_['most_wind_solar']['name']}** (about {cl_['most_wind_solar']['share']}% of its electricity on average): the most diversified group, not a separate 'renewables' cluster.",
        f"- Re-assigning each country's {y0} mix to today's archetypes, **{cl_['moved']} of {cl_['transition_total']} countries** now sit in a different archetype.", "",
        "## 6. Peak coal", "",
        f"- {coal['past_peak']} of {coal['countries']} coal-using countries now generate at least 25% less coal power than at their peak, **{coal['past_peak_before_2015']} of them peaking before 2015**.",
        "- Largest declines: " + "; ".join(f"{r['country']} ({r['peak']:,} TWh in {r['peak_year']} → {r['latest']:,} TWh)" for r in coal["biggest"]) + ".", "",
        "## 7. A concentrated build-out", "",
        f"- Five countries ({con['solar_countries']}) produced **{con['solar_end']}% of the world's solar electricity** in {y1}. The share was {con['solar_start']}% in {y0} too, but with a different line-up ({con['solar_start_countries']}). For wind the top five ({con['wind_countries']}) produced {con['wind_end']}%.", "",
        "## Data-quality notes", "",
        f"- The file has {d['rows']:,} rows for {d['entities']} entities, but only **{d['countries']} are countries**; the other {d['aggregates']} are regions, income groups or historical entities and must be excluded from rankings.",
        f"- Adding the separate biofuel column to the eight source shares double-counts biofuel (it is already inside `other_renewables`): shares exceeded 100% for **{d['double_count']['over_100']} of {d['double_count']['checked']} countries** in {y1} (up to {d['double_count']['max_total']}%).",
        f"- {d['inconsistent_rows']} country-year rows have shares not summing to 100%" + (f", all between {d['inconsistent_years'][0]} and {d['inconsistent_years'][1]}." if d["inconsistent_years"] else "."),
        f"- Coverage thins at the recent edge: {d['countries_with_electricity'][str(y1)]} countries have {y1} electricity data but only {d['countries_with_electricity'][str(y1 + 1)]} have {y1 + 1}, and GDP stops at {d['last_gdp_year']} ({d['countries_with_gdp_last_year']} countries in {f['years']['gdp_year']}). That is why the analysis ends at {y1} and the wealth comparison uses {f['years']['gdp_year']}.", "",
        "## Caveats", "",
        "- This is about **electricity**, not total energy: transport, heating and industry are not covered.",
        "- Shares describe where electricity is generated, not where it is consumed; cross-border trade is ignored.",
        "- Correlations are descriptive. They show which countries are alike, not what caused a change.",
        "- Cluster names come from each cluster's dominant sources; neighbouring clusters overlap and a few countries sit near the boundary.",
        "- Tiny grids are excluded by the minimum-size rule, and results depend on it (see the sensitivity checks in the notebook).", ""]
    return "\n".join(lines)


def make_figures(a: Analysis, out: str | Path) -> dict[str, Path]:
    out = Path(out)
    slope, intercept = np.polyfit(a.intensity["change_pp"], a.intensity["ci_change"], 1)
    rho = an.spearman_ci(a.intensity["change_pp"], a.intensity["ci_change"], seed=a.seed)[0]
    return {
        "coverage": plots.plot_coverage(a.countries, out / "01_coverage.png"),
        "world_mix": plots.plot_world_mix(a.world, out / "02_world_mix.png"),
        "coal_share_vs_total": plots.plot_coal_share_vs_total(a.world, out / "03_share_vs_absolute.png"),
        "leaders": plots.plot_leaders_laggards(a.changes, out / "04_leaders_laggards.png"),
        "replacements": plots.plot_replacements(a.changes, out / "05_what_replaced_what.png"),
        "adoption": plots.plot_adoption_curves(a.countries, a.adoption, out / "06_adoption_curves.png"),
        "wealth": plots.plot_wealth(a.countries, a.gdp_year, a.wealth, out / "07_wealth.png"),
        "clusters": plots.plot_clusters(a.clusters, a.mix1, out / "08_archetypes.png"),
        "transitions": plots.plot_transitions(a.transitions, out / "09_transitions.png", a.y0, a.y1),
        "intensity": plots.plot_intensity(a.intensity, float(slope), float(intercept), rho, out / "10_carbon_intensity.png"),
        "coal_peaks": plots.plot_coal_peaks(a.coal, out / "11_coal_peaks.png"),
        "concentration": plots.plot_concentration(a.solar_conc, a.wind_conc, out / "12_concentration.png"),
    }


def run_all(df: pd.DataFrame, out_dir: str | Path = "reports", **kwargs) -> dict:
    a = run_analysis(df, **kwargs)
    out = Path(out_dir)
    make_figures(a, out / "figures")
    facts = compute_facts(a)
    (out / "facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    (out / "insights.md").write_text(render_insights(facts), encoding="utf-8")
    return facts
