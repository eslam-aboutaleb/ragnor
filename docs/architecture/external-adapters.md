# Writing an external Ragnor adapter

Ragnor defines structural Python ports (Protocols) for ingestion sources, chunkers, tokenizers, embedding providers, search and storage capabilities, job processors and event sinks. A host may inject implementations directly; a registry is optional. There is no vendor-specific base class requirement.

## Example: host chunker + embedder

```python
from ragnor.builder import RagnorBuilder
from ragnor.types import Chunk, Document

class PipeChunker:
    version = "pipe-v1"
    config = {"separator": "|"}

    def chunk(self, text: str) -> list[Chunk]:
        return [
            Chunk(id=f"part-{i}", text=piece)
            for i, piece in enumerate(text.split("|")) if piece
        ]

# host_embedder implements async __call__, embed_query and embed_documents
rag = await RagnorBuilder.local(host_embedder).with_chunker(PipeChunker()).build()
await rag.ingest([Document(id="policy", text="refund|insurance terms")])
```

The example uses ephemeral local defaults; supply a real store, version store and session factory for durable deployments (see [builder composition](builder-composition.md)).

## Conformance expectations

| Adapter | Contract / required checks |
|---|---|
| Chunker | `Chunker` structure, nonempty text, version and config, deterministic content (IDs may be generated) |
| Tokenizer | `Tokenizer` structure, integer non-negative counts, exact-mode claims verified against actual model tokenizer |
| Embedding provider | `EmbeddingFunction` async methods, batch cardinality, vector dimensions, finite floats; missing provider rejected |
| Document source | `load()` returns typed document records; identity/provenance preserved |
| Vector store | Capability negotiation, filter and strict-mode semantics, CRUD atomicity where promised |
| Job processor/store | `JobProcessor`, `EmbeddingJobStore`; durable jobs must prove locking with real DB processes |
| Event sink | `EventSink.emit(TelemetryEvent)`, privacy policy, non-blocking / failure isolation as required by host |
| Authorization | Host scope authorizer; fail closed on unsupported mandatory scopes |

## Clean Architecture requirements

- Import Ragnor *ports/types* to implement your adapter. Application use cases must not import your concrete adapter package.
- Vendor clients, credentials, network connections and model weights remain in your own adapter; don't add them to Ragnor's mandatory dependencies.
- Validate capabilities at composition time. A store without tenant filter support cannot silently serve protected data.
- Document resource ownership and async loop affinity. Host-supplied sessions/clients remain host-owned unless agreed otherwise.
- Add negative tests and integration tests under real infrastructure for durability, timeouts, cancellation and distributed correctness.
- Do not claim a tokenizer or backend is exact/durable based only on `isinstance(..., Protocol)`.
