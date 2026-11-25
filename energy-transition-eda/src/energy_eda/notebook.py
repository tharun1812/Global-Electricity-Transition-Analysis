"""Builds the analysis notebook from code and executes it, so the committed .ipynb always shows real outputs."""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

SETUP = '''import sys
sys.path.insert(0, "../src")
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from IPython.display import Image, Markdown, display
from energy_eda import analysis as an, cleaning as cl, data, report

pd.set_option("display.precision", 1)
FIG = "../reports/figures/"
df = data.load("../data/owid-energy-data.csv")
a = report.run_analysis(df)                     # all the analyses, run once
report.make_figures(a, FIG)                     # regenerate the charts shown below
facts = report.compute_facts(a)
print(f"{len(df):,} rows, {df['country'].nunique()} entities, years {df['year'].min()}-{df['year'].max()}")'''

CELLS = [
    ("md", """# Energy transition: who is decarbonizing electricity, and how fast?

**Data:** [Our World in Data energy dataset](https://github.com/owid/energy-data) (country-level electricity by source, 1985 to 2024).
**Questions:** which countries cleaned up their electricity most between 2014 and 2024, how fast can wind and solar take off, does wealth
buy a cleaner grid, and what kinds of electricity mix exist?

Everything below is computed by the `energy_eda` package in this repository; the final cell prints the findings document."""),
    ("code", SETUP),
    ("md", """## 1. Data quality first
Before any analysis: which rows are real countries, how complete is the recent data, and can the source shares be trusted?"""),
    ("code", """countries, aggregates = cl.split_entities(df)
print(f"{countries['country'].nunique()} countries vs {aggregates['country'].nunique()} aggregates (regions, income groups, 'World', historical entities)")
print("examples of aggregates:", sorted(aggregates['country'].unique())[:6])
n, over, worst = cl.double_count_check(countries, 2024)
print(f"\\nAdding the biofuel column to the eight sources would push {over} of {n} countries above 100% (up to {worst:.0f}%): biofuel is already inside 'other renewables'.")
bad = cl.inconsistent_share_rows(countries)
print(f"{len(bad)} country-years have shares that do not sum to 100%, all in {bad['year'].min()}-{bad['year'].max()}.")
display(Image(FIG + "01_coverage.png"))"""),
    ("md", "## 2. The world: shares fall, totals do not\nA falling share is not a falling total."),
    ("code", """display(Image(FIG + "02_world_mix.png"))
display(Image(FIG + "03_share_vs_absolute.png"))
a.world[["coal", "total", "coal_share_elec", "low_carbon_share_elec", "carbon_intensity_elec"]].loc[[1990, 2000, 2010, 2014, 2024]]"""),
    ("md", """## 3. Who decarbonized between 2014 and 2024?
Low-carbon share = nuclear + hydro + wind + solar + bio/other renewables. Only grids of at least 5 TWh are compared, so tiny systems do not dominate."""),
    ("code", """display(a.changes[["low_carbon_start", "low_carbon_end", "change_pp"]].head(8))
display(a.changes[["low_carbon_start", "low_carbon_end", "change_pp"]].tail(5))
display(Image(FIG + "04_leaders_laggards.png"))
display(Image(FIG + "05_what_replaced_what.png"))
display(Image(FIG + "10_carbon_intensity.png"))"""),
    ("md", """## 4. How fast can wind and solar take off?
For each country: years from first reaching 1% to first reaching 10% of electricity. Countries that have not reached 10% yet are *censored*
(we only know they took at least that long), so they are reported separately, not averaged in."""),
    ("code", """r = a.adoption[a.adoption["reached"]]
print(f"{len(a.adoption)} grids took off; {len(r)} reached 10%; {(~a.adoption['reached']).sum()} still on the way")
print(f"median years from 1% to 10%: {r['years'].median():g} (quartiles {r['years'].quantile(.25):g} to {r['years'].quantile(.75):g})")
display(r.head(8))
display(Image(FIG + "06_adoption_curves.png"))"""),
    ("md", "## 5. Does wealth buy a cleaner grid?\nRank correlations with GDP per person, with bootstrap 95% confidence intervals."),
    ("code", """display(a.wealth.round(2))
display(Image(FIG + "07_wealth.png"))"""),
    ("md", """## 6. Electricity archetypes
K-means on the square-root of each country's source shares. *k* is chosen by silhouette among values whose clusters are **stable**
(agreement across random 80% subsamples). All values tried are shown."""),
    ("code", """display(a.clusters.scores.round(3))
for cid, name in a.clusters.names.items():
    top = a.mix1.loc[a.clusters.labels[a.clusters.labels == cid].index, "generation_twh"].nlargest(5).index.tolist()
    print(f"{name:20s} {a.clusters.sizes[cid]:3d} countries, e.g. {', '.join(top)}")
display(Image(FIG + "08_archetypes.png"))
display(Image(FIG + "09_transitions.png"))"""),
    ("md", "## 7. Peak coal and a concentrated build-out"),
    ("code", """display(a.coal[a.coal["past_peak"]].head(6).round(0))
display(Image(FIG + "11_coal_peaks.png"))
display(Image(FIG + "12_concentration.png"))"""),
    ("md", """## 8. Sensitivity: do the conclusions depend on my choices?
Two checks: the minimum grid size used for rankings, and the number of clusters."""),
    ("code", """import numpy as np
from sklearn.metrics import adjusted_rand_score
base = set(a.changes.head(10).index)
rows = []
for m in (1, 5, 10, 20):
    ch = an.decarbonization_change(a.countries, 2014, 2024, min_twh=m)
    rows.append({"min grid size (TWh)": m, "countries compared": len(ch), "top-10 overlap with default": len(base & set(ch.head(10).index))})
display(pd.DataFrame(rows))
k5, k6 = an.fit_clusters(a.mix1, k=5, n_boot=5), an.fit_clusters(a.mix1, k=6, n_boot=5)
print(f"agreement between the 5-cluster and 6-cluster solutions (adjusted Rand index): {adjusted_rand_score(k5.labels, k6.labels):.2f}")"""),
    ("md", "## 9. Findings\nGenerated from the same analysis object, so every number below is computed, not typed."),
    ("code", """display(Markdown(report.render_insights(facts)))"""),
]


def build_and_run(data_path: str = "data/owid-energy-data.csv", out: str | Path = "notebooks/energy_transition_eda.ipynb") -> Path:
    nb = new_notebook(metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
    nb.cells = [new_markdown_cell(src) if kind == "md" else new_code_cell(src) for kind, src in CELLS]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(out.parent.resolve())}}).execute()
    nbformat.write(nb, out)
    return out
