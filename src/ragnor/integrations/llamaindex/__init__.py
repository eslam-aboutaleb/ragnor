"""LlamaIndex compatibility (optional extra ``ragnor[llamaindex]``).

Inbound adapters bring LlamaIndex components into ragnor:

* :class:`LlamaIndexDocumentSource` - a LlamaIndex reader or ``Document``
  list as a ragnor ``DocumentSource``.
* :class:`LlamaIndexChunker` - a LlamaIndex ``TextSplitter`` as a ragnor
  ``Chunker``.

Outbound adapters expose ragnor to LlamaIndex:

* :class:`RagnorRetriever` - a ragnor retrieval strategy as a LlamaIndex
  ``BaseRetriever``.
* :class:`RagnorVectorStore` - a ragnor ``VectorStore`` as a LlamaIndex
  vector store.

Both outbound adapters require an isolation scope (tenant/owner) at
construction and enforce it themselves; callers cannot override it per query.
The core ``ragnor`` package never imports this module.
"""

from __future__ import annotations

try:
    import llama_index.core  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "ragnor.integrations.llamaindex needs llama-index-core. "
        "Install it with: pip install 'ragnor[llamaindex]'"
    ) from exc

from ragnor.integrations.llamaindex.chunker import LlamaIndexChunker
from ragnor.integrations.llamaindex.retriever import RagnorRetriever
from ragnor.integrations.llamaindex.source import LlamaIndexDocumentSource
from ragnor.integrations.llamaindex.vector_store import RagnorVectorStore

__all__ = [
    "LlamaIndexChunker",
    "LlamaIndexDocumentSource",
    "RagnorRetriever",
    "RagnorVectorStore",
]
