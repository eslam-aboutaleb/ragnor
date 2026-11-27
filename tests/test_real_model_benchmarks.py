"""Plumbing of the real-model benchmark: settings, cost cap, Qdrant path, workflow safety.

The models themselves are only reachable from the manual workflow with secrets.
These tests use real components: a real deterministic embedder, an in-process
Qdrant engine (qdrant-client local mode), the real EX-02 corpus and the real
CLI. They never call a hosted model.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from benchmarks.pdf_extraction.manifest import load_manifest
from benchmarks.pdf_extraction.probes import embed as hashing_embed
from benchmarks.pdf_extraction.probes import load_probes
from benchmarks.pdf_extraction.providers import SDKProvider
from benchmarks.real_models import config as cfg
from benchmarks.real_models import run as bench_run
from benchmarks.real_models.budget import BudgetExceededError, CostMeter
from benchmarks.real_models.config import ConfigError, Settings
from benchmarks.real_models.models import LiteLLMEmbedder, ModelCallError
from benchmarks.real_models.retrieval import brute_force, embed_corpus, qdrant_search

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "benchmarks/pdf_extraction/fixtures"
WORKFLOW = ROOT / ".github/workflows/real-model-benchmarks.yml"
SECRET_VALUE = "sk-test-not-a-real-key-123456"


def batch_embed(texts):
    return [hashing_embed(t) for t in texts]


# -- settings ----------------------------------------------------------------


def test_settings_use_generic_names_and_fall_back_to_the_shared_key():
    env = {
        cfg.API_KEY: SECRET_VALUE,
        cfg.EMBEDDING_MODEL: "provider-x/embed-1",
        cfg.VLM_MODEL: "provider-y/vision-2",
        cfg.VLM_API_BASE: "https://example.invalid/v1",
    }
    s = Settings.from_env(env)
    assert s.embedding.api_key == SECRET_VALUE and s.embedding.api_base is None
    assert s.vlm.api_base == "https://example.invalid/v1"
    assert s.qdrant_url is None


def test_public_view_and_errors_never_contain_secret_values():
    s = Settings.from_env({cfg.API_KEY: SECRET_VALUE, cfg.EMBEDDING_MODEL: "p/m"})
    assert SECRET_VALUE not in json.dumps(s.public())
    with pytest.raises(ConfigError) as err:
        Settings.from_env({cfg.API_KEY: SECRET_VALUE}).require("embeddings")
    assert cfg.EMBEDDING_MODEL in str(err.value)
    assert SECRET_VALUE not in str(err.value)


def test_each_suite_names_what_it_needs():
    empty = Settings.from_env({})
    empty.require("docling")  # local, needs nothing
    for suite in ("embeddings", "qdrant", "vlm"):
        with pytest.raises(ConfigError):
            empty.require(suite)
    with pytest.raises(ConfigError, match=cfg.QDRANT_URL):
        Settings.from_env({cfg.API_KEY: "k", cfg.EMBEDDING_MODEL: "p/m"}).require("qdrant")


# -- cost cap ----------------------------------------------------------------


def test_cost_cap_refuses_before_a_call_would_exceed_it():
    meter = CostMeter(0.05)
    meter.reserve(0.02)
    meter.charge("vlm", 0.04)
    with pytest.raises(BudgetExceededError):
        meter.reserve(0.02)
    assert meter.to_dict()["spent_usd"] == 0.04


@pytest.mark.parametrize("cap", [0, -1])
def test_cost_cap_must_be_positive(cap):
    with pytest.raises(ValueError, match="max_usd"):
        CostMeter(cap)


def test_embedder_sends_nothing_when_the_cap_is_too_low():
    endpoint = Settings.from_env({cfg.API_KEY: SECRET_VALUE, cfg.EMBEDDING_MODEL: "p/m"}).embedding
    embedder = LiteLLMEmbedder(endpoint, CostMeter(1e-9))
    with pytest.raises(BudgetExceededError):
        embedder.embed(["x" * 5000])
    assert embedder.latencies_ms == []


def test_failed_call_error_has_no_provider_text_or_key():
    endpoint = Settings.from_env(
        {
            cfg.API_KEY: SECRET_VALUE,
            cfg.API_BASE: "http://127.0.0.1:9",  # closed port: a real failed request
            cfg.EMBEDDING_MODEL: "openai/text-embedding-test",
        }
    ).embedding
    with pytest.raises(ModelCallError) as err:
        LiteLLMEmbedder(endpoint, CostMeter(1.0)).embed(["hello"])
    assert SECRET_VALUE not in str(err.value) and "127.0.0.1" not in str(err.value)


# -- retrieval and Qdrant ------------------------------------------------------


@pytest.fixture(scope="module")
def index():
    manifest = load_manifest(FIXTURES / "manifest.jsonl")
    probes = load_probes(FIXTURES / "probes.json")
    chunks, vectors = embed_corpus(manifest, FIXTURES, SDKProvider(engine="auto"), batch_embed)
    return chunks, vectors, probes


def test_brute_force_recall_on_the_real_corpus(index):
    chunks, vectors, probes = index
    result = brute_force(chunks, vectors, probes, batch_embed)
    assert result["probes"] == 8 and result["indexed_chunks"] == len(chunks)
    assert 0.5 <= result["recall_at_k"] <= 1 and 0 < result["mrr"] <= 1


def test_qdrant_search_matches_brute_force_and_cleans_up(index):
    from qdrant_client import QdrantClient

    chunks, vectors, probes = index
    client = QdrantClient(":memory:")
    result = qdrant_search(client, chunks, vectors, probes, batch_embed)
    expected = brute_force(chunks, vectors, probes, batch_embed)
    assert result["recall_at_k"] == expected["recall_at_k"]
    assert result["mrr"] == expected["mrr"]
    assert result["queries_timed"] == 8 * 50 and result["query_ms_p95"] >= result["query_ms_p50"]
    assert client.get_collections().collections == []


# -- CLI ----------------------------------------------------------------------


def test_cli_rejects_unknown_suites_and_caps_above_the_hard_limit(capsys):
    assert bench_run.main(["--suites", "nope"]) == 2
    assert bench_run.main(["--suites", "docling", "--max-cost-usd", "11"]) == 2
    assert bench_run.main(["--suites", "docling", "--max-cost-usd", "0"]) == 2


def test_dry_run_lists_missing_settings_by_name_only(capsys, monkeypatch):
    for name in cfg.ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(cfg.API_KEY, SECRET_VALUE)
    assert bench_run.main(["--suites", "embeddings,vlm", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert cfg.EMBEDDING_MODEL in out and cfg.VLM_MODEL in out
    assert SECRET_VALUE not in out


def test_unconfigured_suites_are_skipped_not_failed(tmp_path, monkeypatch):
    for name in (*cfg.ENV_NAMES, cfg.QDRANT_URL):
        monkeypatch.delenv(name, raising=False)
    code = bench_run.main(["--suites", "embeddings,qdrant,vlm", "--out", str(tmp_path)])
    report = json.loads((tmp_path / "results.json").read_text())
    assert code == 0
    assert {r["status"] for r in report["results"].values()} == {"skipped"}
    assert "skipped" in (tmp_path / "summary.md").read_text()


# -- workflow safety -------------------------------------------------------------


@pytest.fixture(scope="module")
def workflow_text():
    return WORKFLOW.read_text()


@pytest.fixture(scope="module")
def workflow():
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(WORKFLOW.read_text())


def test_workflow_is_dispatch_only(workflow):
    triggers = workflow.get(True) or workflow.get("on")  # PyYAML parses "on" as True
    assert set(triggers) == {"workflow_dispatch"}


def test_workflow_runs_only_from_main_with_read_only_token(workflow):
    assert workflow["permissions"] == {"contents": "read"}
    job = workflow["jobs"]["benchmark"]
    assert job["if"] == "github.ref == 'refs/heads/main'"
    assert job["timeout-minutes"] <= 60


def test_workflow_has_a_qdrant_service_container(workflow):
    service = workflow["jobs"]["benchmark"]["services"]["qdrant"]
    assert service["image"].startswith("qdrant/qdrant:v")
    assert "6333:6333" in service["ports"]


def test_every_action_is_pinned_to_a_commit_sha(workflow_text):
    uses = re.findall(r"uses:\s*(\S+)", workflow_text)
    assert uses and all(re.fullmatch(r".+@[0-9a-f]{40}", u) for u in uses)


def test_secrets_are_only_referenced_in_step_env_with_known_names(workflow_text):
    refs = re.findall(r"\$\{\{\s*secrets\.([A-Z0-9_]+)\s*\}\}", workflow_text)
    assert set(refs) == set(cfg.SECRET_NAMES) == {"LITELLM_API_KEY", "LITELLM_API_BASE"}
    for line in workflow_text.splitlines():
        if "secrets." in line and not line.lstrip().startswith("#"):
            assert re.match(
                r"\s+LITELLM_API_(KEY|BASE): \$\{\{ secrets\.LITELLM_API_(KEY|BASE) \}\}$", line
            ), line


def test_workflow_never_echoes_or_traces_secrets(workflow_text):
    assert "set -x" not in workflow_text and "env |" not in workflow_text
    assert "printenv" not in workflow_text
    assert "echo ${{" not in workflow_text


def test_user_inputs_reach_the_shell_only_through_env(workflow_text):
    for line in workflow_text.splitlines():
        if "inputs." in line and not line.lstrip().startswith("#"):
            stripped = line.strip()
            assert stripped.startswith(
                (
                    "SUITES:",
                    "MAX_COST:",
                    "MAX_PAGES:",
                    "EMBED_MODEL:",
                    "VLM_MODEL:",
                    "RAGNOR_BENCH_",
                    "if:",
                )
            ), line


def test_workflow_uploads_results_and_writes_a_job_summary(workflow_text):
    assert "upload-artifact@" in workflow_text and "GITHUB_STEP_SUMMARY" in workflow_text


def test_models_are_inputs_with_gateway_alias_defaults(workflow):
    triggers = workflow.get(True) or workflow.get("on")
    inputs = triggers["workflow_dispatch"]["inputs"]
    assert inputs["embedding_model"]["default"] == "openai/rag-embedding"
    assert inputs["vlm_model"]["default"] == "openai/general-reasoning"


def test_workflow_secret_list_matches_the_docs():
    docs = (ROOT / "docs/real-model-benchmarks.md").read_text()
    for name in cfg.SECRET_NAMES:
        assert f"`{name}`" in docs
    assert "RAGNOR_BENCH_API_KEY" not in docs
