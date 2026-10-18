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
The SonarCloud project key retains its existing external identity.

## Release process

The repository is `eslam-aboutaleb/ragnor`. Store a PyPI API token authorized to
publish `ragnor` as the repository Actions secret `PYPI_API_TOKEN`. The `pypi`
environment requires the repository owner's approval before uploading.

1. Merge the migration or release change only after **CI Status Check** succeeds.
2. Ensure the package version matches the proposed `v<version>` release tag.
3. Publish a GitHub release from a commit contained in `main`.
4. The Publish workflow checks the name and tag, runs the complete qualification
   workflow, then calls the pinned shared PyPI workflow to build and validate the
   distributions and publish with `PYPI_API_TOKEN` after approval. The token is
   available only in the upload job. Token publication does not generate PyPI
   attestations; PyPI OIDC publication requires an upload job in the consuming
   repository's own workflow.

Manual dispatch of **Publish** runs qualification without uploading to PyPI. Tag
pushes alone do not publish, preventing duplicate uploads when creating a release.
