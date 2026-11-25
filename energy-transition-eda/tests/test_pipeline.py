"""End to end on the real-data sample: the whole pipeline runs, writes every output, and is reproducible."""
import json
import re
from pathlib import Path

import pytest

from energy_eda import data, report
from energy_eda.cli import main

SAMPLE = Path(__file__).parent / "fixtures" / "energy_sample.csv"


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("reports")
    facts = report.run_all(data.load(SAMPLE), out, min_twh=1.0)
    return out, facts


def test_every_output_is_written(run):
    out, _ = run
    figures = sorted((out / "figures").glob("*.png"))
    assert len(figures) == 12 and all(f.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and f.stat().st_size > 5000 for f in figures)
    assert (out / "facts.json").exists() and (out / "insights.md").exists()


def test_findings_document_has_the_sections_and_real_numbers(run):
    out, facts = run
    text = (out / "insights.md").read_text()
    for heading in ("## 1. The world is greening", "## 2. Who decarbonized", "## 5. Country archetypes", "## Data-quality notes", "## Caveats"):
        assert heading in text
    assert f"{facts['world']['low_carbon_end']}%" in text
    assert not re.search(r"\bnan\b", text.lower()), "a missing value leaked into the findings"


def test_facts_are_internally_consistent(run):
    _, facts = run
    assert 0 < facts["world"]["low_carbon_start"] < facts["world"]["low_carbon_end"] < 100
    assert facts["changes"]["top"][0]["change"] >= facts["changes"]["top"][-1]["change"]
    assert facts["changes"]["bottom"][0]["change"] <= facts["changes"]["bottom"][-1]["change"]
    assert sum(facts["clusters"]["sizes"].values()) == facts["data"]["countries_compared"] or facts["clusters"]["k"] >= 2
    assert facts["intensity"]["slope"] < 0 and facts["intensity"]["rho"] < -0.5       # cleaner mix, lower carbon intensity
    json.dumps(facts)                                                                  # serialisable


def test_the_pipeline_is_reproducible(run, tmp_path):
    _, facts = run
    again = report.run_all(data.load(SAMPLE), tmp_path, min_twh=1.0)
    assert again == facts


def test_command_line_analyze(tmp_path, capsys):
    assert main(["analyze", "--data", str(SAMPLE), "--out", str(tmp_path), "--min-twh", "1"]) == 0
    assert "wrote" in capsys.readouterr().out and (tmp_path / "insights.md").exists()
