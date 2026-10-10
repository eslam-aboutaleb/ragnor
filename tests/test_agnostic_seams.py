"""Domain-agnostic seam tests.

Parametrizes two fake host domains -- medical and legal -- with
different settings, embedding models and dimensions, chunker
strategies, table names, tsquery configs, and domain vocabulary,
and proves every public ragit seam accepts them without any
domain-specific code in the library: the settings protocol, the
embedding provider registry, the dimension registry, the chunker
registry, the vector store registry, the in-memory store, the
hybrid retriever, the ingestion pipeline, the evaluation runner,
the heuristic judge, pgvector constructor validation, and the
public API surface.

Everything runs offline: fake embedders, fake providers,
in-memory stores, and injected retrieval functions. No network,
no database, no API keys.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import pkgutil
import re
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import pytest

import ragit
from ragit.chunking import Chunker, SlidingWindowChunker, chunker_registry
from ragit.config import RagSettings
from ragit.embeddings import (
    FALLBACK_DIMENSION,
    EmbeddingFunction,
    embedding_registry,
    get_embedding_dimension,
    get_embedding_function,
    register_dimension,
)
from ragit.evaluation import (
    Difficulty,
    EvalConfig,
    EvalSample,
    RetrievalFunction,
    evaluate_answer_heuristic,
    run_evaluation,
)
from ragit.ingestion import (
    IngestionPipeline,
    IngestionSnapshot,
    RawDocument,
    VersionRecord,
)
from ragit.retrieval import HybridRetriever, RetrieverSpec, as_dicts
from ragit.stores import (
    InMemoryVectorStore,
    PgVectorStore,
    VectorStore,
    get_vector_store,
    vector_store_registry,
)
from ragit.validation import validate_embedding


@dataclass(frozen=True)
class DomainConfig:
    """Everything a host domain configures ragit with."""

    name: str
    embedding_model: str
    embedding_dimension: int
    embedding_provider: str
    vector_store_provider: str
    chunker_name: str
    chunk_size: int
    overlap: int
    table_name: str
    id_field: str
    tsquery_config: str
    distance_threshold: float
    domain_terms: tuple[str, ...]
    chunker_sample_text: str
    chunker_expected_sections: tuple[str, ...]
    store_documents: tuple[str, ...]
    store_query: str
    store_expected_keyword: str
    ingestion_document: str
    ingestion_source_name: str
    ingestion_query: str
    eval_queries: tuple[tuple[str, str, str], ...]
    judge_gold: str
    judge_partial: str
    judge_missing: str


MEDICAL_CONFIG = DomainConfig(
    name="medical",
    embedding_model="med-embed-v1",
    embedding_dimension=128,
    embedding_provider="medical-provider",
    vector_store_provider="medical-backend",
    chunker_name="sliding-window",
    chunk_size=50,
    overlap=10,
    table_name="medical_chunks",
    id_field="chunk_id",
    tsquery_config="english",
    distance_threshold=1.3,
    domain_terms=("diagnosis", "symptom", "treatment", "prognosis"),
    chunker_sample_text=(
        "The patient presents with a persistent cough and fever. "
        "The diagnosis is pneumonia. The treatment is antibiotics and rest. "
        "The prognosis is good with treatment. Follow up in two weeks."
    ),
    chunker_expected_sections=(),
    store_documents=(
        "The patient presents with a persistent cough and fever. "
        "The diagnosis is pneumonia. The treatment is antibiotics and rest.",
        "Common symptoms of dehydration include dry mouth and fatigue. "
        "The treatment is fluid replacement.",
        "A headache can be a symptom of tension. "
        "The diagnosis requires a neurological exam.",
    ),
    store_query="what is the treatment for pneumonia",
    store_expected_keyword="pneumonia",
    ingestion_document=(
        "The patient presents with a persistent cough and fever. "
        "The diagnosis is pneumonia. The treatment is antibiotics and rest. "
        "The prognosis is good with treatment. Follow up in two weeks. "
        "The symptoms include cough and fever. The treatment plan is clear. "
        "The patient responds well to antibiotics. The fever subsides "
        "within days. The prognosis remains good."
    ),
    ingestion_source_name="medical_notes.md",
    ingestion_query="what is the treatment for pneumonia",
    eval_queries=(
        (
            "what is the treatment for pneumonia",
            "treatment",
            "The treatment is antibiotics and rest.",
        ),
        (
            "what symptoms indicate dehydration",
            "symptoms",
            "Symptoms include dry mouth and fatigue.",
        ),
        (
            "how is a headache diagnosed",
            "diagnosis",
            "The diagnosis requires a neurological exam.",
        ),
    ),
    judge_gold="The symptom is fever and the treatment is antibiotics.",
    judge_partial="The symptom is fever.",
    judge_missing="The patient is resting comfortably.",
)

LEGAL_CONFIG = DomainConfig(
    name="legal",
    embedding_model="legal-embed-v2",
    embedding_dimension=64,
    embedding_provider="legal-provider",
    vector_store_provider="legal-backend",
    chunker_name="markdown",
    chunk_size=80,
    overlap=16,
    table_name="legal_clauses",
    id_field="clause_id",
    tsquery_config="simple",
    distance_threshold=1.3,
    domain_terms=("liability", "breach", "indemnification", "jurisdiction"),
    chunker_sample_text=(
        "# Legal Agreement\n\n"
        "## Liability\n\n"
        "The liability of the parties is capped at the fees paid.\n\n"
        "## Breach\n\n"
        "A breach of this agreement triggers indemnification.\n\n"
        "## Jurisdiction\n\n"
        "The jurisdiction shall be the courts of New York."
    ),
    chunker_expected_sections=("Legal Agreement", "Liability", "Breach", "Jurisdiction"),
    store_documents=(
        "The liability of the parties is capped at the fees paid. "
        "A breach of this agreement triggers indemnification.",
        "The governing law and jurisdiction shall be the courts of New York. "
        "Any breach must be notified in writing.",
        "Indemnification shall cover all claims arising from negligence. "
        "The liability is limited as stated herein.",
    ),
    store_query="what happens on breach of the agreement",
    store_expected_keyword="breach",
    ingestion_document=(
        "# Legal Services Agreement\n\n"
        "## Liability\n\n"
        "The liability of the parties is capped at the fees paid.\n\n"
        "## Breach\n\n"
        "A breach of this agreement triggers indemnification.\n\n"
        "## Jurisdiction\n\n"
        "The jurisdiction shall be the courts of New York."
    ),
    ingestion_source_name="legal_agreement.md",
    ingestion_query="what triggers indemnification on breach",
    eval_queries=(
        (
            "what happens on breach of the agreement",
            "breach",
            "A breach triggers indemnification.",
        ),
        (
            "how is liability limited",
            "liability",
            "The liability is capped at the fees paid.",
        ),
        (
            "which courts have jurisdiction",
            "jurisdiction",
            "The jurisdiction shall be the courts of New York.",
        ),
    ),
    judge_gold="The liability is capped and the breach triggers indemnification.",
    judge_partial="The liability is capped.",
    judge_missing="The contract was signed today.",
)

DOMAINS: tuple[DomainConfig, ...] = (MEDICAL_CONFIG, LEGAL_CONFIG)

register_dimension(MEDICAL_CONFIG.embedding_model, MEDICAL_CONFIG.embedding_dimension)
register_dimension(LEGAL_CONFIG.embedding_model, LEGAL_CONFIG.embedding_dimension)


class DomainSettings:
    """Host settings satisfying the ``RagSettings`` protocol for one domain."""

    def __init__(self, config: DomainConfig) -> None:
        """Initialize every field ragit reads at runtime."""
        self.embedding_model = config.embedding_model
        self.embedding_provider = config.embedding_provider
        self.rag_distance_threshold = config.distance_threshold
        self.vector_store_provider = config.vector_store_provider
        self.openai_api_key = ""
        self.openai_api_base = ""
        self.embedding_drain_interval_seconds = 5.0
        self.embedding_drain_batch_size = 10
        self.job_stale_after_seconds = 300.0
        self.worker_id = f"{config.name}-worker"
        self.ingest_on_startup = False
        self.database_url = ""


class FakeDomainEmbedder:
    """Deterministic token-hashing embedder tagged with its model.

    Tokens hash into vector dimensions, so texts that share
    vocabulary get close L2 distance -- bag-of-words similarity
    without any network calls. The dimension comes from the
    domain's registered embedding model.
    """

    def __init__(self, model_name: str, dimension: int) -> None:
        """Store the model tag and the output dimension."""
        self.model_name = model_name
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        """Return the normalized token-hash embedding of ``text``."""
        vector = [0.0] * self.dimension
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:2], "big") % self.dimension
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0.0:
            vector = [value / norm for value in vector]
        return vector

    async def __call__(self, input: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        return [self.embed(text) for text in input]

    async def embed_query(self, input: str | list[str]) -> list[float]:
        """Embed a query."""
        return self.embed(input if isinstance(input, str) else input[0])

    async def embed_documents(self, input: list[str]) -> list[list[float]]:
        """Embed a batch of documents."""
        return [self.embed(text) for text in input]


class MedicalEmbedder(FakeDomainEmbedder):
    """Medical-domain embedder."""


class LegalEmbedder(FakeDomainEmbedder):
    """Legal-domain embedder."""


class _DomainEmbeddingProvider:
    """Base factory building a domain embedder from host settings."""

    embedder_cls: type[FakeDomainEmbedder]

    def __init__(self, settings: RagSettings) -> None:
        """Store the settings the factory reads."""
        self._settings = settings

    def create(self) -> FakeDomainEmbedder:
        """Return the domain embedder for the configured model."""
        model = self._settings.embedding_model
        return self.embedder_cls(model, get_embedding_dimension(model))


@embedding_registry.register("medical-provider")
class MedicalEmbeddingProvider(_DomainEmbeddingProvider):
    """Builds the medical embedder from medical settings."""

    embedder_cls = MedicalEmbedder


@embedding_registry.register("legal-provider")
class LegalEmbeddingProvider(_DomainEmbeddingProvider):
    """Builds the legal embedder from legal settings."""

    embedder_cls = LegalEmbedder


class SharedIndexBackend(VectorStore):
    """Vector backend over a shared in-memory index per table name.

    One shared :class:`InMemoryVectorStore` per table name keeps
    the pipeline's writes visible to the retriever's reads within
    this process; a production backend shares a database server
    instead, and the host wiring code is identical.
    """

    _indexes: dict[str, InMemoryVectorStore] = {}

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        table_name: str = "documents",
        id_field: str = "id",
        embedding_dim: int | None = None,
        session_factory: Any = None,
        settings: RagSettings | None = None,
        embedding_fn: EmbeddingFunction | None = None,
    ) -> None:
        """Initialize the backend, reusing the shared index for ``table_name``."""
        backend = SharedIndexBackend._indexes.get(table_name)
        if backend is None:
            backend = InMemoryVectorStore(
                table_name=table_name,
                id_field=id_field,
                embedding_dim=embedding_dim,
                embedding_fn=embedding_fn,
            )
            SharedIndexBackend._indexes[table_name] = backend
        self._backend = backend
        self.table_name = table_name
        self.id_field = id_field
        self.embedding_dim = embedding_dim

    async def hybrid_search(  # noqa: PLR0913, PLR0917
        self,
        query: str,
        embedding: list[float],
        n_results: int,
        threshold: float,
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
        session: Any = None,
        strict: bool = False,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        """Hybrid search delegated to the shared in-memory index."""
        return await self._backend.hybrid_search(
            query=query,
            embedding=embedding,
            n_results=n_results,
            threshold=threshold,
            text_field=text_field,
            metadata_fields=metadata_fields,
            session=session,
            strict=strict,
            **filters,
        )

    async def count(
        self,
        session: Any = None,
        strict: bool = False,
        **filters: Any,
    ) -> int:
        """Count documents in the shared index."""
        return await self._backend.count(session=session, strict=strict, **filters)

    async def upsert(  # noqa: PLR0913, PLR0917
        self,
        documents: list[dict[str, Any]],
        session: Any = None,
        text_field: str = "text",
        embedding_field: str = "embedding",
        id_field: str | None = None,
        extra_fields: dict[str, Any] | None = None,
    ) -> None:
        """Upsert documents into the shared index."""
        await self._backend.upsert(
            documents,
            session=session,
            text_field=text_field,
            embedding_field=embedding_field,
            id_field=id_field,
            extra_fields=extra_fields,
        )


@vector_store_registry.register("medical-backend")
class MedicalVectorBackend(SharedIndexBackend):
    """Medical-domain vector backend over the shared in-memory index."""


@vector_store_registry.register("legal-backend")
class LegalVectorBackend(SharedIndexBackend):
    """Legal-domain vector backend over the shared in-memory index."""


class _NullSession:
    """No-op session: the offline pipeline commits nothing."""

    async def commit(self) -> None:
        """No-op commit (there is no database to commit to)."""


@asynccontextmanager
async def _null_session_factory() -> AsyncIterator[_NullSession]:
    """Yield a no-op session (the pipeline runs without a database)."""
    yield _NullSession()


class MemoryVersionStore:
    """In-memory ``VersionStore`` (the test keeps no version table)."""

    def __init__(self) -> None:
        """Initialize with no active version."""
        self.active: VersionRecord | None = None

    async def find_active(self, session: Any) -> VersionRecord | None:
        """Return the active version, or ``None``."""
        return self.active

    async def close_active(self, session: Any) -> None:
        """Retire the active version."""
        self.active = None

    async def create_version(
        self,
        session: Any,
        snapshot: IngestionSnapshot,
    ) -> VersionRecord:
        """Insert a new version row carrying ``snapshot``."""
        self.active = VersionRecord(
            version_id=uuid.uuid4().hex[:8],
            snapshot=snapshot,
            source_id="source-document",
        )
        return self.active

    async def delete_chunks(self, session: Any, version_id: str) -> None:
        """Delete the chunks belonging to ``version_id`` (no-op)."""


class _StaticSource:
    """Document source yielding a fixed document set."""

    def __init__(self, documents: list[RawDocument]) -> None:
        """Store the documents this source provides."""
        self._documents = documents

    async def load(self) -> list[RawDocument]:
        """Return the documents this source provides."""
        return list(self._documents)


@pytest.fixture(autouse=True)
def _isolate_shared_indexes() -> Iterator[None]:
    """Isolate the shared in-memory indexes between tests."""
    SharedIndexBackend._indexes.clear()
    yield
    SharedIndexBackend._indexes.clear()


def _config_ids(config: DomainConfig) -> str:
    """Return the pytest id for a domain config."""
    return config.name


def _other_domain(config: DomainConfig) -> DomainConfig:
    """Return the other domain in the matrix."""
    return next(domain for domain in DOMAINS if domain.name != config.name)


def _build_chunker(config: DomainConfig) -> Chunker:
    """Resolve the domain's chunker from the chunking registry."""
    chunker_cls = chunker_registry.get(config.chunker_name)
    if config.chunker_name == "markdown":
        return chunker_cls(
            SlidingWindowChunker(config.chunk_size, config.overlap),
            source_name=config.name,
        )
    return chunker_cls(chunk_size=config.chunk_size, overlap=config.overlap)


def _eval_dataset(config: DomainConfig) -> list[EvalSample]:
    """Build the domain's evaluation dataset."""
    return [
        EvalSample(
            query=query,
            expected_sections=[section],
            gold_answer=gold_answer,
            category=config.name,
            difficulty=Difficulty.EASY,
        )
        for query, section, gold_answer in config.eval_queries
    ]


def _make_retrieval_fn(config: DomainConfig) -> RetrievalFunction:
    """Build a fake retrieval function answering from the domain's test set."""
    sections = {query: section for query, section, _ in config.eval_queries}

    async def retrieval_fn(query: str) -> list[dict[str, Any]]:
        """Return one canned result carrying the expected section."""
        section = sections.get(query, "unknown")
        return [
            {
                "id": f"{config.name}-{section}",
                "document": f"{section} documentation for the {config.name} domain",
                "metadata": {"section": section},
                "distance": 0.1,
                "keyword_score": 1.0,
                "_rrf_score": 0.9,
            }
        ]

    return retrieval_fn


def _public_api_names() -> set[str]:
    """Return every public name the ragit package and its submodules export."""
    names = set(dir(ragit))
    for module_info in pkgutil.iter_modules(ragit.__path__):
        module = importlib.import_module(f"ragit.{module_info.name}")
        names.update(getattr(module, "__all__", ()))
        names.update(name for name in dir(module) if not name.startswith("_"))
    return names


def test_domains_use_distinct_configurations() -> None:
    """The two domains differ on every axis the seams are parametrized over."""
    assert MEDICAL_CONFIG.embedding_dimension != LEGAL_CONFIG.embedding_dimension
    assert MEDICAL_CONFIG.embedding_model != LEGAL_CONFIG.embedding_model
    assert MEDICAL_CONFIG.table_name != LEGAL_CONFIG.table_name
    assert MEDICAL_CONFIG.tsquery_config != LEGAL_CONFIG.tsquery_config
    assert MEDICAL_CONFIG.chunker_name != LEGAL_CONFIG.chunker_name
    assert set(MEDICAL_CONFIG.domain_terms).isdisjoint(LEGAL_CONFIG.domain_terms)


@pytest.mark.parametrize("config", DOMAINS, ids=_config_ids)
def test_domain_settings_satisfy_the_rag_settings_protocol(config: DomainConfig) -> None:
    """Each domain's settings class satisfies the ``RagSettings`` protocol."""
    settings = DomainSettings(config)
    assert isinstance(settings, RagSettings)
    assert settings.embedding_model == config.embedding_model
    assert settings.embedding_provider == config.embedding_provider
    assert settings.vector_store_provider == config.vector_store_provider
    assert settings.rag_distance_threshold == config.distance_threshold
    assert settings.worker_id == f"{config.name}-worker"


@pytest.mark.parametrize("config", DOMAINS, ids=_config_ids)
def test_dimension_registry_resolves_per_domain(config: DomainConfig) -> None:
    """The dimension registry resolves each domain's model to its dimension."""
    assert get_embedding_dimension(config.embedding_model) == config.embedding_dimension
    assert get_embedding_dimension("totally-unknown-model") == FALLBACK_DIMENSION


@pytest.mark.parametrize("config", DOMAINS, ids=_config_ids)
def test_embedding_provider_seam_resolves_per_domain(config: DomainConfig) -> None:
    """The embedding registry resolves the right provider per domain config."""
    settings = DomainSettings(config)
    embedder = get_embedding_function(settings)
    assert isinstance(embedder, EmbeddingFunction)
    assert isinstance(embedder, FakeDomainEmbedder)
    assert embedder.model_name == config.embedding_model
    assert embedder.dimension == config.embedding_dimension

    provider_cls = embedding_registry.get(config.embedding_provider)
    assert provider_cls(settings).create().__class__ is embedder.__class__

    other = _other_domain(config)
    other_embedder = get_embedding_function(DomainSettings(other))
    assert other_embedder.__class__ is not embedder.__class__
    assert other_embedder.dimension == other.embedding_dimension


@pytest.mark.parametrize("config", DOMAINS, ids=_config_ids)
async def test_dimension_validation_rejects_mismatched_dimensions(
    config: DomainConfig,
) -> None:
    """Dimension validation rejects embeddings sized for the other domain."""
    other = _other_domain(config)
    with pytest.raises(ValueError):
        validate_embedding(
            [0.1] * other.embedding_dimension,
            config.embedding_dimension,
            label="document embedding",
        )

    store = InMemoryVectorStore(
        table_name=config.table_name,
        id_field=config.id_field,
        embedding_dim=config.embedding_dimension,
    )
    with pytest.raises(ValueError):
        await store.upsert(
            [
                {
                    "id": "mismatched",
                    "text": "wrong dimension",
                    "embedding": [0.1] * other.embedding_dimension,
                }
            ]
        )


@pytest.mark.parametrize("config", DOMAINS, ids=_config_ids)
def test_chunker_seam_resolves_per_domain(config: DomainConfig) -> None:
    """The chunking registry resolves the domain's chunker strategy."""
    chunker = _build_chunker(config)
    assert isinstance(chunker, Chunker)
    assert chunker.__class__ is chunker_registry.get(config.chunker_name)

    chunks = chunker.chunk(config.chunker_sample_text)
    assert chunks
    for chunk in chunks:
        assert chunk.text.strip()

    if config.chunker_name == "markdown":
        sections = {chunk.metadata["section"] for chunk in chunks}
        assert sections == set(config.chunker_expected_sections)
        assert all(chunk.metadata["chunk_id"].startswith("chunk_") for chunk in chunks)
        assert chunker.version == "section-aware-v1"
    else:
        assert chunker.config == {"chunk_size": config.chunk_size, "overlap": config.overlap}
        assert chunker.version == "sliding-window-v1"


def test_vector_store_registry_resolves_per_domain() -> None:
    """The store registry resolves a distinct backend class per domain."""
    medical = get_vector_store(
        table_name=MEDICAL_CONFIG.table_name,
        settings=DomainSettings(MEDICAL_CONFIG),
    )
    legal = get_vector_store(
        table_name=LEGAL_CONFIG.table_name,
        settings=DomainSettings(LEGAL_CONFIG),
    )
    assert isinstance(medical, SharedIndexBackend)
    assert isinstance(legal, SharedIndexBackend)
    assert medical.__class__ is not legal.__class__
    assert vector_store_registry.get("medical-backend") is MedicalVectorBackend
    assert vector_store_registry.get("legal-backend") is LegalVectorBackend


@pytest.mark.parametrize("config", DOMAINS, ids=_config_ids)
async def test_memory_store_seam_upsert_search_per_domain(config: DomainConfig) -> None:
    """The in-memory store upserts and searches each domain's corpus."""
    settings = DomainSettings(config)
    embedder = get_embedding_function(settings)
    store = InMemoryVectorStore(
        table_name=config.table_name,
        id_field=config.id_field,
        embedding_dim=config.embedding_dimension,
        embedding_fn=embedder,
    )

    texts = list(config.store_documents)
    embeddings = await embedder.embed_documents(texts)
    documents = [
        {
            "id": f"{config.name}-{index}",
            "text": text,
            "metadata": {"domain": config.name},
            "embedding": embedding,
        }
        for index, (text, embedding) in enumerate(zip(texts, embeddings, strict=True))
    ]
    await store.upsert(documents)
    assert await store.count() == len(documents)
    assert await store.count(domain=config.name) == len(documents)

    query_embedding = await embedder.embed_query(config.store_query)
    results = await store.hybrid_search(
        query=config.store_query,
        embedding=query_embedding,
        n_results=3,
        threshold=2.0,
        metadata_fields=["domain"],
    )
    assert results
    assert config.store_expected_keyword in results[0]["document"].lower()
    assert results[0]["metadata"]["domain"] == config.name
    assert results[0]["_rrf_score"] > 0.0

    auto_results = await store.hybrid_search(
        query=config.store_query,
        embedding=[],
        n_results=3,
        threshold=2.0,
    )
    assert auto_results
    assert config.store_expected_keyword in auto_results[0]["document"].lower()


@pytest.mark.parametrize("config", DOMAINS, ids=_config_ids)
async def test_retriever_seam_top_k_per_domain(config: DomainConfig) -> None:
    """The hybrid retriever retrieves each domain's top-k results."""
    settings = DomainSettings(config)
    embedder = get_embedding_function(settings)
    store = get_vector_store(
        table_name=config.table_name,
        id_field=config.id_field,
        embedding_dim=config.embedding_dimension,
        settings=settings,
        embedding_fn=embedder,
    )
    assert isinstance(store, SharedIndexBackend)

    texts = list(config.store_documents)
    embeddings = await embedder.embed_documents(texts)
    await store.upsert(
        [
            {
                "id": f"{config.name}-{index}",
                "text": text,
                "metadata": {"domain": config.name},
                "embedding": embedding,
            }
            for index, (text, embedding) in enumerate(zip(texts, embeddings, strict=True))
        ]
    )

    retriever = HybridRetriever(
        spec=RetrieverSpec(
            table_name=config.table_name,
            id_field=config.id_field,
            metadata_fields=["domain"],
        ),
        settings=settings,
    )

    results = await retriever.retrieve(config.store_query, n_results=3)
    assert results
    assert config.store_expected_keyword in results[0].document.lower()
    assert results[0].metadata["domain"] == config.name

    filtered = await retriever.retrieve(config.store_query, n_results=3, domain=config.name)
    assert filtered

    as_dict_results = as_dicts(results)
    assert as_dict_results[0]["id"] == results[0].id
    assert as_dict_results[0]["document"] == results[0].document
    assert as_dict_results[0]["metadata"]["domain"] == config.name


@pytest.mark.parametrize("config", DOMAINS, ids=_config_ids)
async def test_ingestion_pipeline_seam_per_domain(config: DomainConfig) -> None:
    """The ingestion pipeline runs each domain's document set end to end."""
    settings = DomainSettings(config)
    embedder = get_embedding_function(settings)
    store = get_vector_store(
        table_name=config.table_name,
        id_field=config.id_field,
        embedding_dim=config.embedding_dimension,
        settings=settings,
        embedding_fn=embedder,
    )
    pipeline = IngestionPipeline(
        store=store,
        embedder=embedder,
        chunker=_build_chunker(config),
        session_factory=_null_session_factory,
        settings=settings,
    )
    version_store = MemoryVersionStore()
    source = _StaticSource(
        [RawDocument(source_name=config.ingestion_source_name, content=config.ingestion_document)]
    )

    count = await pipeline.run(source, version_store)
    assert count > 0
    assert await store.count() == count

    results = await store.hybrid_search(
        query=config.ingestion_query,
        embedding=await embedder.embed_query(config.ingestion_query),
        n_results=count,
        threshold=2.0,
        metadata_fields=["source", "version_id"],
    )
    assert results
    for result in results:
        assert result["metadata"]["source"] == config.ingestion_source_name
        assert result["metadata"]["version_id"]

    rerun = await pipeline.run(source, version_store)
    assert rerun == count


@pytest.mark.parametrize("config", DOMAINS, ids=_config_ids)
async def test_evaluation_seam_retrieval_metrics_per_domain(config: DomainConfig) -> None:
    """``run_evaluation`` computes retrieval metrics for each domain's dataset."""
    result = await run_evaluation(
        EvalConfig(mode="retrieval-only", judge="heuristic", n_results=5),
        retrieval_fn=_make_retrieval_fn(config),
        dataset=_eval_dataset(config),
        settings=DomainSettings(config),
    )
    assert result.retrieval_metrics["total_queries"] == len(config.eval_queries)
    assert result.retrieval_metrics["errors"] == 0
    assert result.retrieval_metrics["recall_at_k"] == 1.0
    assert result.retrieval_metrics["precision_at_k"] == 1.0
    assert result.retrieval_metrics["mrr"] == 1.0
    assert result.per_sample_results


@pytest.mark.parametrize("config", DOMAINS, ids=_config_ids)
async def test_evaluation_seam_end_to_end_per_domain(config: DomainConfig) -> None:
    """``run_evaluation`` computes answer metrics for each domain's dataset."""
    gold = {query: gold_answer for query, _, gold_answer in config.eval_queries}

    async def answer_fn(query: str, context: str) -> str:
        """Simulate a faithful answer generator."""
        return gold.get(query, context)

    result = await run_evaluation(
        EvalConfig(mode="end-to-end", judge="heuristic", n_results=5),
        retrieval_fn=_make_retrieval_fn(config),
        dataset=_eval_dataset(config),
        settings=DomainSettings(config),
        answer_fn=answer_fn,
    )
    assert result.retrieval_metrics["errors"] == 0
    assert result.answer_metrics["total_evaluated"] == len(config.eval_queries)
    assert result.answer_metrics["errors"] == 0
    assert result.answer_metrics["avg_overall"] > 0.0
    assert result.summary


@pytest.mark.parametrize("config", DOMAINS, ids=_config_ids)
def test_heuristic_judge_domain_terms_boost_per_domain(config: DomainConfig) -> None:
    """Domain terms present in the gold answer boost the heuristic judge's score."""
    query = f"what are the key {config.name} facts"
    domain_terms = set(config.domain_terms)

    full = evaluate_answer_heuristic(
        query=query,
        answer=config.judge_gold,
        gold_answer=config.judge_gold,
        retrieved_context=config.judge_gold,
        domain_terms=domain_terms,
    )
    partial = evaluate_answer_heuristic(
        query=query,
        answer=config.judge_partial,
        gold_answer=config.judge_gold,
        retrieved_context=config.judge_gold,
        domain_terms=domain_terms,
    )
    missing = evaluate_answer_heuristic(
        query=query,
        answer=config.judge_missing,
        gold_answer=config.judge_gold,
        retrieved_context=config.judge_gold,
        domain_terms=domain_terms,
    )

    assert full.completeness == 1.0
    assert 0.0 < partial.completeness < 1.0
    assert missing.completeness == 0.0
    assert full.overall > partial.overall > missing.overall

    neutral = evaluate_answer_heuristic(
        query=query,
        answer=config.judge_partial,
        gold_answer=config.judge_gold,
        retrieved_context=config.judge_gold,
    )
    assert neutral.completeness != partial.completeness


@pytest.mark.parametrize("config", DOMAINS, ids=_config_ids)
def test_pgvector_store_accepts_domain_configuration(config: DomainConfig) -> None:
    """PgVectorStore accepts each domain's table, id, tsquery, and dimension."""
    store = PgVectorStore(
        table_name=config.table_name,
        id_field=config.id_field,
        embedding_dim=config.embedding_dimension,
        tsquery_config=config.tsquery_config,
    )
    assert store.table_name == config.table_name
    assert store.id_field == config.id_field
    assert store.embedding_dim == config.embedding_dimension
    assert store.tsquery_config == config.tsquery_config


@pytest.mark.parametrize("config", DOMAINS, ids=_config_ids)
def test_pgvector_store_rejects_unsafe_table_name(config: DomainConfig) -> None:
    """PgVectorStore rejects a table name with unsafe characters."""
    with pytest.raises(ValueError):
        PgVectorStore(
            table_name=f"{config.table_name}; DROP TABLE {config.table_name}",
            id_field=config.id_field,
            embedding_dim=config.embedding_dimension,
            tsquery_config=config.tsquery_config,
        )


@pytest.mark.parametrize("config", DOMAINS, ids=_config_ids)
def test_pgvector_store_rejects_unsafe_id_field(config: DomainConfig) -> None:
    """PgVectorStore rejects an id field with unsafe characters."""
    with pytest.raises(ValueError):
        PgVectorStore(
            table_name=config.table_name,
            id_field="unsafe id",
            embedding_dim=config.embedding_dimension,
            tsquery_config=config.tsquery_config,
        )


@pytest.mark.parametrize("config", DOMAINS, ids=_config_ids)
def test_pgvector_store_rejects_unsafe_tsquery_config(config: DomainConfig) -> None:
    """PgVectorStore rejects tsquery configs that are not lowercase identifiers."""
    with pytest.raises(ValueError):
        PgVectorStore(
            table_name=config.table_name,
            id_field=config.id_field,
            embedding_dim=config.embedding_dimension,
            tsquery_config="English",
        )
    with pytest.raises(ValueError):
        PgVectorStore(
            table_name=config.table_name,
            id_field=config.id_field,
            embedding_dim=config.embedding_dimension,
            tsquery_config="simple config",
        )


@pytest.mark.parametrize("config", DOMAINS, ids=_config_ids)
def test_pgvector_store_requires_embedding_dimension(config: DomainConfig) -> None:
    """PgVectorStore rejects construction without an embedding dimension."""
    with pytest.raises(ValueError):
        PgVectorStore(
            table_name=config.table_name,
            id_field=config.id_field,
            embedding_dim=None,
            tsquery_config=config.tsquery_config,
        )


def test_public_api_surface_has_no_domain_specific_symbols() -> None:
    """No domain-specific symbols leak into the ragit public API surface."""
    banned_terms = ("insurance", "medical", "legal", "policy", "financial", "banking")
    for name in _public_api_names():
        lowered = name.lower()
        for term in banned_terms:
            assert term not in lowered, (
                f"domain-specific symbol {name!r} leaks into the public API"
            )
