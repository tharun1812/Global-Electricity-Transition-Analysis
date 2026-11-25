# Findings

*Generated from the data by `energy_eda`. Electricity only. Years compared: 2014 and 2024; countries with at least 5 TWh of generation.*

## 1. The world is greening its electricity, but coal is still setting records

- The low-carbon share of world electricity rose from **32.8% to 40.9%** between 2014 and 2024, and wind plus solar from 3.8% to 15.0%.
- Carbon intensity fell from 545 to **471 gCO2 per kWh**.
- Coal's share fell from 40.4% to 34.1%, **yet coal generation hit its record in 2024: 10,539 TWh** (up 9.7% from 9,604 TWh in 2014). Total generation grew 30% over the decade, so coal's output rose even as its share fell.

## 2. Who decarbonized

- Of 113 countries compared, **38 gained 10 or more points** of low-carbon share and **21 went backwards**.
- Biggest gains: Estonia (11.2% → 55.8%); Lebanon (1.2% → 43.8%); Netherlands (15.4% → 53.7%); Belarus (0.8% → 37.3%); Venezuela (55.9% → 91.1%); Denmark (56.1% → 89.2%).
- Biggest losses: Ghana (63.6% → 36.2%); Laos (100.0% → 76.7%); Mozambique (98.7% → 83.4%); Myanmar (63.0% → 47.9%).
- A cleaner mix means a cleaner kilowatt-hour: across 113 countries each extra point of low-carbon share went with **-7.3 gCO2/kWh** (Spearman ρ = -0.93, 95% CI -0.96 to -0.88).

## 3. How fast can a new source take off

- 93 grids of meaningful size moved wind plus solar past 1% of their electricity; 57 have since passed 10% (the rest are still on the way). The median time from 1% to 10% was **9 years** (middle half: 5 to 11).
- Later starters look quicker: takeoffs since 2015 reached 10% in a median **3.5 years** (n = 12), against 12 years for takeoffs before 2010 (n = 21). Recent takeoffs are, by construction, still being observed, so the recent figure only counts the fast ones.

## 4. Wealth does not buy a clean grid

- Across countries in 2022, GDP per person shows no clear relationship with the low-carbon share of electricity (Spearman ρ = -0.09, 95% CI -0.25 to 0.08), but a strong one with how much electricity each person uses (ρ = 0.89).
- Among the richest quarter of countries (37 of them) the low-carbon share ranges from 2% to 92% (10th to 90th percentile): wealth explains *how much* electricity a country uses, not *how clean* it is.

## 5. Country archetypes

- Clustering the 2024 electricity mixes gives **5 archetypes** (silhouette 0.312, stability 0.83; stability is the average agreement between clusterings of random 80% subsamples, 1 = identical).
  - **Hydro-dominated**: 38 countries, including Brazil, Canada, Norway, Colombia.
  - **Gas-dominated**: 34 countries, including Iran, Mexico, Italy, Egypt.
  - **Coal-dominated**: 21 countries, including China, India, Japan, Germany.
  - **Nuclear + Gas**: 20 countries, including United States, Russia, South Korea, France.
  - **Oil-dominated**: 9 countries, including Puerto Rico, Cuba, Honduras, Senegal.
- The archetype that leans most on wind and solar is **Nuclear + Gas** (about 24% of its electricity on average): the most diversified group, not a separate 'renewables' cluster.
- Re-assigning each country's 2014 mix to today's archetypes, **13 of 113 countries** now sit in a different archetype.

## 6. Peak coal

- 43 of 76 coal-using countries now generate at least 25% less coal power than at their peak, **29 of them peaking before 2015**.
- Largest declines: United States (2,038 TWh in 1999 → 652 TWh); United Kingdom (211 TWh in 1991 → 2 TWh); Germany (315 TWh in 1986 → 106 TWh); Canada (116 TWh in 2000 → 25 TWh).

## 7. A concentrated build-out

- Five countries (China, United States, India, Japan, Germany) produced **68% of the world's solar electricity** in 2024. The share was 68% in 2014 too, but with a different line-up (Germany, United States, Japan, China, Italy). For wind the top five (China, United States, Germany, Brazil, United Kingdom) produced 71%.

## Data-quality notes

- The file has 23,377 rows for 314 entities, but only **220 are countries**; the other 94 are regions, income groups or historical entities and must be excluded from rankings.
- Adding the separate biofuel column to the eight source shares double-counts biofuel (it is already inside `other_renewables`): shares exceeded 100% for **73 of 122 countries** in 2024 (up to 122.4%).
- 107 country-year rows have shares not summing to 100%, all between 1985 and 1999.
- Coverage thins at the recent edge: 196 countries have 2024 electricity data but only 90 have 2025, and GDP stops at 2022 (165 countries in 2022). That is why the analysis ends at 2024 and the wealth comparison uses 2022.

## Caveats

- This is about **electricity**, not total energy: transport, heating and industry are not covered.
- Shares describe where electricity is generated, not where it is consumed; cross-border trade is ignored.
- Correlations are descriptive. They show which countries are alike, not what caused a change.
- Cluster names come from each cluster's dominant sources; neighbouring clusters overlap and a few countries sit near the boundary.
- Tiny grids are excluded by the minimum-size rule, and results depend on it (see the sensitivity checks in the notebook).
