"""Compatibility import for :mod:`ragnor.client.builder`."""

from __future__ import annotations

# ruff: noqa: PLC0414 - explicit aliases declare typed compatibility exports

import sys
from typing import TYPE_CHECKING

from ragnor.client import builder as _implementation

if TYPE_CHECKING:
    from ragnor.client.builder import (
        Any as Any,
        AsyncCloseable as AsyncCloseable,
        AsyncIterator as AsyncIterator,
        BackendFeature as BackendFeature,
        BackendUnavailableError as BackendUnavailableError,
        BuilderSettings as BuilderSettings,
        Chunker as Chunker,
        ConfigurationError as ConfigurationError,
        DEFAULT_CHUNK_SIZE as DEFAULT_CHUNK_SIZE,
        DEFAULT_EMBEDDING_MODEL as DEFAULT_EMBEDDING_MODEL,
        DEFAULT_OVERLAP as DEFAULT_OVERLAP,
        DeploymentMode as DeploymentMode,
        Document as Document,
        DocumentStore as DocumentStore,
        EmbeddingFunction as EmbeddingFunction,
        EventSink as EventSink,
        HybridRetrieval as HybridRetrieval,
        HybridRetriever as HybridRetriever,
        InMemoryVectorStore as InMemoryVectorStore,
        InMemoryVersionStore as InMemoryVersionStore,
        IngestionPipeline as IngestionPipeline,
        IngestionSnapshot as IngestionSnapshot,
        LexicalIndex as LexicalIndex,
        ListDocumentSource as ListDocumentSource,
        OperationTimeoutError as OperationTimeoutError,
        RagSettings as RagSettings,
        Ragnor as Ragnor,
        RagnorBuilder as RagnorBuilder,
        Reranker as Reranker,
        RetrievalStrategy as RetrievalStrategy,
        RetrieverSpec as RetrieverSpec,
        SearchResult as SearchResult,
        Sequence as Sequence,
        SlidingWindowChunker as SlidingWindowChunker,
        StoreCapabilities as StoreCapabilities,
        VectorIndex as VectorIndex,
        VectorStore as VectorStore,
        VersionRecord as VersionRecord,
        VersionStore as VersionStore,
        asynccontextmanager as asynccontextmanager,
        asyncio as asyncio,
        cast as cast,
        dataclass as dataclass,
        has_registered_embedding_dimension as has_registered_embedding_dimension,
        register_dimension as register_dimension,
        sanitize_message as sanitize_message,
    )

# Keep module identity, private helpers and patch targets compatible too.
sys.modules[__name__] = _implementation
