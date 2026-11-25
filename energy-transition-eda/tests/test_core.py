"""Fast tests on small synthetic data where the right answer can be worked out by hand."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from energy_eda import analysis as an
from energy_eda import cleaning as cl
from energy_eda import data

SAMPLE = Path(__file__).parent / "fixtures" / "energy_sample.csv"


def rows(country, year, shares, twh=50.0, **extra):
    """One country-year with source shares given as a dict (the rest are 0)."""
    r = {"country": country, "year": year, "iso_code": "XXX", "electricity_generation": twh,
         **{f"{s}_share_elec": float(shares.get(s, 0.0)) for s in cl.MIX_SOURCES}, "biofuel_share_elec": 0.0}
    r.update(extra)
    return r


# ---- data and cleaning -------------------------------------------------------------------------------------------
def test_real_sample_loads_and_a_broken_file_is_refused(tmp_path):
    df = data.load(SAMPLE)
    assert df["country"].nunique() == 41 and "World" in set(df["country"])
    with pytest.raises(ValueError, match="missing columns"):
        data.validate_schema(df.drop(columns=["gdp"]))
    with pytest.raises(ValueError, match="duplicate"):
        data.validate_schema(pd.concat([df, df.head(1)]))


def test_aggregates_are_separated_from_countries():
    countries, aggregates = cl.split_entities(data.load(SAMPLE))
    assert "World" in set(aggregates["country"]) and "World" not in set(countries["country"])
    assert countries["iso_code"].notna().all()


def test_a_mix_adds_up_to_100_and_bad_rows_are_dropped_or_raise():
    good = rows("A", 2024, {"coal": 60, "gas": 40})
    bad = rows("B", 2024, {"coal": 60, "gas": 30})                 # adds up to 90
    small = rows("C", 2024, {"coal": 100}, twh=1.0)                # too small a grid
    df = pd.DataFrame([good, bad, small])
    mix = cl.electricity_mix(df, 2024)
    assert list(mix.index) == ["A"] and mix.attrs["dropped"] == ["B"]
    assert mix[cl.MIX_SOURCES].sum(axis=1).iloc[0] == pytest.approx(100)
    with pytest.raises(ValueError, match="do not add up"):
        cl.electricity_mix(df, 2024, on_mismatch="raise")
    assert len(cl.inconsistent_share_rows(df)) == 1


def test_real_mixes_are_consistent_and_the_biofuel_double_count_is_detected():
    countries, _ = cl.split_entities(data.load(SAMPLE))
    mix = cl.electricity_mix(countries, 2024, min_twh=1.0)
    assert len(mix) > 30 and np.allclose(mix[cl.MIX_SOURCES].sum(axis=1), 100, atol=0.5)
    checked, over, worst = cl.double_count_check(countries, 2024)
    assert over >= 1 and worst > 100.5


def test_gdp_per_capita_ignores_zero_population():
    out = cl.gdp_per_capita(pd.DataFrame({"gdp": [100.0, 50.0], "population": [10.0, 0.0]}))
    assert out.iloc[0] == 10.0 and np.isnan(out.iloc[1])


# ---- analyses ----------------------------------------------------------------------------------------------------
def test_decarbonization_change_by_hand():
    df = pd.DataFrame([rows("X", 2014, {"coal": 80, "gas": 20}), rows("X", 2024, {"coal": 40, "gas": 20, "wind": 40}),
                       rows("Y", 2014, {"hydro": 50, "coal": 50}), rows("Y", 2024, {"hydro": 40, "coal": 60})])
    ch = an.decarbonization_change(df, 2014, 2024)
    assert list(ch.index) == ["X", "Y"]
    assert ch.loc["X", "change_pp"] == 40 and ch.loc["Y", "change_pp"] == -10
    assert ch.loc["X", "d_coal"] == -40 and ch.loc["X", "d_wind_solar"] == 40
    assert np.allclose(ch[[c for c in ch.columns if c.startswith("d_")]].sum(axis=1), 0)      # shares always sum to 100


def test_adoption_speed_handles_reached_censored_and_tiny_grids():
    years = range(1990, 2025)
    ws = lambda f: [f(y) for y in years]
    a_share = ws(lambda y: 0 if y < 2000 else min(1.5 + (y - 2000) * 1.7, 12))        # 1% in 2000, 10% in 2005
    b_share = ws(lambda y: 0 if y < 2010 else 1 + (y - 2010) * 0.35)                   # 1% in 2010, still under 10% in 2024
    frames = []
    for name, share, twh in (("A", a_share, 100.0), ("B", b_share, 100.0), ("Tiny", a_share, 1.0)):
        frames.append(pd.DataFrame({"country": name, "year": list(years), "wind_share_elec": share, "solar_share_elec": 0.0, "electricity_generation": twh}))
    out = an.adoption_speed(pd.concat(frames))
    assert set(out.index) == {"A", "B"}                                               # the 1 TWh grid is excluded
    assert out.loc["A", "takeoff_year"] == 2000 and out.loc["A", "years"] == 5 and bool(out.loc["A", "reached"])
    assert out.loc["B", "takeoff_year"] == 2010 and not out.loc["B", "reached"] and out.loc["B", "years"] == 14


def test_spearman_ci_is_reproducible_and_sensible():
    x = np.arange(30.0)
    rho, lo, hi, n = an.spearman_ci(x, x * 2 + 1, n_boot=200)
    assert (rho, n) == (1.0, 30) and lo <= rho <= hi
    rng = np.random.default_rng(1)
    noisy = x + rng.normal(0, 8, 30)
    a, b = an.spearman_ci(x, noisy, seed=3, n_boot=300), an.spearman_ci(x, noisy, seed=3, n_boot=300)
    assert a == b and a[1] <= a[0] <= a[2] and 0.3 < a[0] < 0.95
    with pytest.raises(ValueError):
        an.spearman_ci([1, 2, 3], [1, 2, 3])


def planted_mix(seed=0):
    rng = np.random.default_rng(seed)
    centers = {"coal": {"coal": 70, "gas": 20, "hydro": 10}, "hydro": {"hydro": 80, "gas": 10, "wind": 10},
               "nuclear": {"nuclear": 60, "gas": 20, "wind": 10, "solar": 10}}
    vecs, truth = [], []
    for i, (name, c) in enumerate(centers.items()):
        base = np.array([c.get(s, 0.0) for s in cl.MIX_SOURCES]) + 0.5
        for _ in range(20):
            vecs.append(rng.dirichlet(base * 30) * 100)
            truth.append(i)
    mix = pd.DataFrame(vecs, columns=cl.MIX_SOURCES, index=[f"c{i}" for i in range(60)])
    mix["generation_twh"] = 50.0
    return mix, np.array(truth)


def test_clustering_recovers_planted_archetypes_and_names_them():
    mix, truth = planted_mix()
    res = an.fit_clusters(mix, k_range=range(2, 6), n_boot=10)
    assert res.k == 3 and res.stability > 0.9
    assert adjusted_rand_score(truth, res.labels.to_numpy()) > 0.9
    assert set(res.names.values()) == {"Coal-dominated", "Hydro-dominated", "Nuclear-dominated"}
    assert res.sizes.sum() == 60 and (res.predict(mix) == res.labels).all()
    again = an.fit_clusters(mix, k_range=range(2, 6), n_boot=10)
    assert (again.labels == res.labels).all()                                         # same seed, same answer


def test_cluster_naming_rule():
    assert an.name_cluster(pd.Series({"coal": 70, "gas": 30, "oil": 0, "nuclear": 0, "hydro": 0, "wind": 0, "solar": 0, "other_renewables": 0})) == "Coal-dominated"
    mixed = pd.Series({"coal": 20, "gas": 30, "oil": 0, "nuclear": 25, "hydro": 10, "wind": 10, "solar": 5, "other_renewables": 0})
    assert an.name_cluster(mixed) == "Gas + Nuclear"


def test_coal_peaks_and_concentration():
    def coal(country, values, start=1985):
        return pd.DataFrame({"country": country, "year": range(start, start + len(values)), "coal_electricity": values})
    df = pd.concat([coal("Past", [100.0] * 10 + [50.0] * 30), coal("Rising", list(np.linspace(10, 90, 40))),
                    coal("Tiny", [0.5] * 40), coal("Stale", [100.0] * 20 + [10.0] * 5)])
    cp = an.coal_peaks(df, last_year=2024)
    assert set(cp.index) == {"Past", "Rising"}                                       # tiny peak and stale data excluded
    assert bool(cp.loc["Past", "past_peak"]) and cp.loc["Past", "decline"] == pytest.approx(0.5) and not cp.loc["Rising", "past_peak"]
    solar = pd.DataFrame({"country": list("ABCDEF"), "year": 2024, "solar_electricity": [50, 30, 10, 5, 3, 2]})
    conc = an.concentration(solar, "solar_electricity", top_n=3)
    assert conc.loc[2024, "top_share"] == pytest.approx(0.9) and conc.loc[2024, "top_countries"] == "A, B, C"
