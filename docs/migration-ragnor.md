# Migrating to Ragnor

Version 0.4.0 uses **Ragnor** as its project name, **`ragnor`** as its PyPI
distribution, and **`ragnor`** as its Python import package.

## Update applications

Install `ragnor` with the same extras your application previously used:

```bash
python -m pip install 'ragnor[pgvector]'
```

Update dependency manifests and lockfiles to use `ragnor` instead of
`ragit-toolkit`. Replace imports of `ragit` with `ragnor`, including dotted module
paths in configuration, monkeypatch targets, and worker launch commands.

The convenience API is now:

```python
from ragnor.builder import Ragnor, RagnorBuilder
from ragnor.types import Document

rag = await RagnorBuilder.local(embedder=my_embedder).build()
await rag.ingest([Document(id='policy', text='The refund policy covers water damage.')])
results = await rag.search('refund policy')
```

`RagitBuilder` becomes `RagnorBuilder`, and `Ragit` becomes `Ragnor`. The previous
import package is not included in the new distribution. Update applications
before removing the old distribution. Stop and restart workers together with
application upgrades; do not rely on serialized Python objects containing old
module paths across this rename.

## Stored data and observability

Host-owned database schemas, configuration fields, document IDs, and revision
formats remain unchanged. Qdrant retains the original deterministic ID namespace,
so existing points can still be retrieved, updated, and deleted without reindexing.

Default logger names and OpenTelemetry instrumentation names, metric names, and
attribute prefixes now use `ragnor`. Update dashboards and log filters accordingly.
The default MCP server name is `ragnor-knowledge`, and the builder's default worker
ID is `ragnor-builder`. Explicit host-provided names and worker IDs are unchanged.
