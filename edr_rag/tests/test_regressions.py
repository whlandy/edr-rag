"""Regression tests for review findings fixed after the initial import."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from edr_rag import extract as extract_mod
from edr_rag import normalize
from edr_rag.p1.search import JsonlCorpus
from edr_rag.p2.api import EdrRagAPI
from edr_rag.p3.server import EdrRagAgent


def test_keyword_search_is_case_insensitive():
    corpus = JsonlCorpus([
        {"id": "dns", "title": "Configure DNS", "content": "Use HTTP Proxy"},
    ])

    assert corpus.keyword_search("dns")[0]["id"] == "dns"
    assert corpus.keyword_search("http")[0]["id"] == "dns"


def test_action_search_is_case_insensitive(tmp_path: Path):
    action = {
        "action_id": "Network.ConfigureDNS",
        "title": "Configure DNS",
        "module": "Network",
        "preconditions": [],
        "procedure": [],
    }
    (tmp_path / "actions.jsonl").write_text(json.dumps(action) + "\n", encoding="utf-8")

    assert EdrRagAPI(tmp_path).action_search("dns")[0]["id"] == action["action_id"]


def test_generated_feedback_ids_are_unique(tmp_path: Path):
    agent = EdrRagAgent(str(tmp_path))

    first = agent.capture_feedback({"manual_id": "m"})
    second = agent.capture_feedback({"manual_id": "m"})

    assert first != second


def test_ingest_replaces_old_extraction_tree(tmp_path: Path, monkeypatch):
    fixtures = []
    for name in ("old", "new"):
        fixture = tmp_path / name
        fixture.mkdir()
        (fixture / f"{name}.html").write_text(
            f"<html><head><title>{name}</title></head><body><h1>{name}</h1></body></html>",
            encoding="utf-8",
        )
        fixtures.append(fixture)

    calls = iter(fixtures)

    def fake_extract(_chm: Path, out: Path):
        # Deliberately do not clear `out`; run_ingest must provide a fresh tree.
        for source in next(calls).iterdir():
            shutil.copy2(source, out / source.name)
        return "sha256:fixture", True

    monkeypatch.setattr(extract_mod, "extract_chm", fake_extract)
    workdir = tmp_path / "work"
    normalize.run_ingest(tmp_path / "fake.chm", "m", "p", workdir)
    normalize.run_ingest(tmp_path / "fake.chm", "m", "p", workdir)

    records = [json.loads(line) for line in (workdir / "sections.jsonl").read_text().splitlines()]
    assert [record["title"] for record in records] == ["new"]
    assert not (workdir / "extracted" / "old.html").exists()


def test_semantic_ids_are_namespaced():
    first = normalize.make_semantic_id(section_path=["DNS"], title="", manual_id="manual-a")
    second = normalize.make_semantic_id(section_path=["DNS"], title="", manual_id="manual-b")
    assert first != second
