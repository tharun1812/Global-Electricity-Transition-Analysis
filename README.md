# Energy transition: who is decarbonizing electricity, and how fast?

An exploratory data analysis of the global electricity mix, built on the [Our World in Data energy dataset](https://github.com/owid/energy-data)
(23,377 rows, 220 countries, 1900 to 2025). It asks which countries cleaned up their electricity between
2014 and 2024, how fast wind and solar can take off, whether wealth buys a cleaner grid, and what kinds of electricity
mix exist.

**Start here:** [`notebooks/energy_transition_eda.ipynb`](notebooks/energy_transition_eda.ipynb) (executed, with all outputs) or the
generated summary [`reports/insights.md`](reports/insights.md).

![Archetypes](reports/figures/08_archetypes.png)

## Key findings

All numbers below are computed from the data by the code in this repository.

1. **The mix is greening, but coal is still setting records.** The low-carbon share of world electricity rose from 32.8% to 40.9%
   (2014 to 2024) and carbon intensity fell from 545 to 471 gCO2/kWh. Yet coal generation hit an
   all-time high in 2024 (10,539 TWh), because total generation grew 30%. A falling share is not a falling total.
2. **Who moved most:** Estonia (11% → 56%), Lebanon (1% → 44%), Netherlands (15% → 54%). **21 of 113 countries went backwards**, including Ghana (64% → 36%), Laos (100% → 77%), Mozambique (99% → 83%) (hydro-reliant grids whose demand outgrew their water power).
3. **A cleaner mix really does mean a cleaner kilowatt-hour:** each extra point of low-carbon share went with -7.3 gCO2/kWh
   (Spearman ρ = -0.93, 95% bootstrap CI -0.96 to -0.88, 113 countries).
4. **Wealth does not buy a clean grid.** GDP per person has no clear relationship with the low-carbon share (ρ = -0.09, 95% CI -0.25 to 0.08),
   but a strong one with electricity use per person (ρ = 0.89). Among the richest quarter of countries the low-carbon share ranges from 2% to 92%.
5. **Wind and solar take about 9 years to go from 1% to 10%** of a grid (middle half: 5 to 11; 57 of 93 grids have got there).
   Countries still on the way are treated as *censored*, not averaged in.
6. **Five electricity archetypes** (k-means, silhouette 0.312, stability 0.83): Hydro-dominated (38), Gas-dominated (34), Coal-dominated (21), Nuclear + Gas (20), Oil-dominated (9). Only 13 of 113 countries changed archetype in the decade.
7. **43 of 76 coal-using countries are at least 25% below their coal peak**, but five countries produce 68% of the world's solar power.

## Data problems found and handled

- **Only 220 of 314 entities are countries.** The rest are regions, income groups and "World"; ranking them together double-counts. Entities without an ISO code are excluded.
- **The obvious set of share columns double-counts.** `other_renewables_share_elec` already includes biofuel, so adding the biofuel column too pushes 73 of 122 countries above 100% (up to 122.4%). The code uses the eight non-overlapping sources and checks that every mix sums to 100%.
- **Source shares are inconsistent before 2000** (107 country-years, none later), so the analysis uses 2014 and 2024.
- **Recent coverage is thin:** 90 countries have 2025 electricity data against 196 for 2024, and GDP stops at 2022. The analysis ends at 2024; wealth comparisons use 2022.
- **Tiny grids distort rankings** (one solar farm can move a small island from 1% to 10%), so comparisons use grids of at least 5 TWh, with a sensitivity check in the notebook.

## Methods

Rank correlations with bootstrap confidence intervals; k-means on square-root-transformed shares with *k* chosen by silhouette among stable solutions
(adjusted Rand index across 80% subsamples); right-censoring in the adoption-speed analysis; per-source decomposition of each country's change; PCA for the
archetype map; sensitivity checks on the grid-size cut-off and on the number of clusters.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m energy_eda download        # fetches the dataset (about 9 MB) into data/
python -m energy_eda analyze         # figures, facts.json and insights.md into reports/
python -m energy_eda notebook        # rebuilds and executes the notebook
python -m pytest -q                  # 16 tests, about 30 seconds
```

## Repository layout

```
src/energy_eda/   data.py (download, validation) | cleaning.py | analysis.py | plots.py | report.py | notebook.py | cli.py
notebooks/        energy_transition_eda.ipynb (executed)
reports/          insights.md, facts.json, figures/ (12 charts)
tests/            synthetic-data unit tests and an end-to-end test on a small real-data sample (tests/fixtures)
data/             codebook (the raw dataset is downloaded on demand)
```

## Limitations

- Electricity only: transport, heating and industry are not covered, and shares describe where electricity is generated, not consumed.
- Correlations are descriptive, not causal; cluster boundaries are fuzzy and the cluster names are labels for dominant sources.
- Results depend on the 5 TWh grid-size cut-off and the 2014 to 2024 window (the notebook shows how much).

## Data

Our World in Data energy dataset: <https://github.com/owid/energy-data>. See the repository there for the licence and for citations of the underlying sources (Ember, Energy Institute and others).
