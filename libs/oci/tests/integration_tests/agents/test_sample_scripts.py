# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Integration-style coverage for the top-level Deep Research sample scripts."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[5]
SAMPLES_DIR = ROOT / "samples"
DEEP_RESEARCH_DIR = SAMPLES_DIR / "11-deepresearch"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeEmbeddingModel:
    def embed_documents(self, texts):
        return [[float(index + 1)] for index, _ in enumerate(texts)]


class FakeStore:
    instances: list["FakeStore"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.connected_with = None
        self.bulk_insert_calls = []
        FakeStore.instances.append(self)

    def connect(self, embedding_model):
        self.connected_with = embedding_model

    def bulk_insert(self, records, embeddings):
        self.bulk_insert_calls.append((records, embeddings))
        return len(records)


class FakeAgent:
    def __init__(self, *, datastores, **kwargs):
        self.datastores = datastores
        self.kwargs = kwargs
        self.prompts = []

    def invoke(self, payload):
        self.prompts.append(payload["messages"][-1]["content"])
        return {"messages": [SimpleNamespace(content="sample deep research output")]}


def _set_common_env(monkeypatch):
    monkeypatch.setenv(
        "OCI_COMPARTMENT_ID",
        "ocid1.compartment.oc1..exampleuniqueID",
    )
    monkeypatch.setenv("OCI_REGION", "us-ashburn-1")
    monkeypatch.setenv("OCI_AUTH_TYPE", "API_KEY")
    monkeypatch.setenv("OCI_CONFIG_PROFILE", "API_KEY_AUTH")


def _mock_sample_dependencies(
    monkeypatch,
    module,
    *,
    store_attr,
    store_class,
    agent_factory,
):
    monkeypatch.setattr(
        module,
        "load_huggingface_dataset",
        lambda *args, **kwargs: [{}],
    )
    monkeypatch.setattr(
        module,
        "OCIGenAIEmbeddings",
        lambda **kwargs: FakeEmbeddingModel(),
    )
    monkeypatch.setattr(module, store_attr, store_class)
    monkeypatch.setattr(module, "create_deep_research_agent", agent_factory)


def test_adb_sample_runs_end_to_end(monkeypatch, tmp_path):
    module = _load_module(
        "adb_sample",
        DEEP_RESEARCH_DIR / "adb_multi_store_huggingface_example.py",
    )
    FakeStore.instances = []
    created_agents = []

    def fake_agent_factory(**kwargs):
        agent = FakeAgent(**kwargs)
        created_agents.append(agent)
        return agent

    _mock_sample_dependencies(
        monkeypatch,
        module,
        store_attr="ADB",
        store_class=FakeStore,
        agent_factory=fake_agent_factory,
    )
    _set_common_env(monkeypatch)
    monkeypatch.setenv("ADB_DSN", "deepresearch_low")
    monkeypatch.setenv("ADB_USER", "ADMIN")
    monkeypatch.setenv("ADB_PASSWORD", "secret")
    monkeypatch.setenv("ADB_WALLET_LOCATION", str(tmp_path))

    output = tmp_path / "adb-report.md"
    exit_code = module.main(
        ["--limit", "1", "--run-id", "testrun", "--output", str(output)]
    )

    assert exit_code == 0
    assert output.read_text(encoding="utf-8") == "sample deep research output"
    assert len(FakeStore.instances) == 2
    assert all(store.connected_with is not None for store in FakeStore.instances)
    assert all(len(store.bulk_insert_calls) == 1 for store in FakeStore.instances)
    assert created_agents
    assert set(created_agents[0].datastores) == {"medical_research", "news_research"}


def test_opensearch_sample_runs_end_to_end(monkeypatch, tmp_path):
    module = _load_module(
        "opensearch_sample",
        DEEP_RESEARCH_DIR / "opensearch_multi_index_huggingface_example.py",
    )
    FakeStore.instances = []
    created_agents = []

    def fake_agent_factory(**kwargs):
        agent = FakeAgent(**kwargs)
        created_agents.append(agent)
        return agent

    _mock_sample_dependencies(
        monkeypatch,
        module,
        store_attr="OpenSearch",
        store_class=FakeStore,
        agent_factory=fake_agent_factory,
    )
    _set_common_env(monkeypatch)
    monkeypatch.setenv("OPENSEARCH_ENDPOINT", "https://opensearch.example.com:9200")
    monkeypatch.setenv("OPENSEARCH_USERNAME", "admin")
    monkeypatch.setenv("OPENSEARCH_PASSWORD", "secret")

    output = tmp_path / "opensearch-report.md"
    exit_code = module.main(
        ["--limit", "1", "--run-id", "testrun", "--output", str(output)]
    )

    assert exit_code == 0
    assert output.read_text(encoding="utf-8") == "sample deep research output"
    assert len(FakeStore.instances) == 2
    assert all(store.connected_with is not None for store in FakeStore.instances)
    assert all(len(store.bulk_insert_calls) == 1 for store in FakeStore.instances)
    assert created_agents
    assert set(created_agents[0].datastores) == {"medical_research", "news_research"}
