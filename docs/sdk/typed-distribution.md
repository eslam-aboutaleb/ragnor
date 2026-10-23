# Typed SDK distribution — PEP 561

The public Ragnor package includes `py.typed` in wheel and sdist artifacts,
and the Python 3.11/3.12 GitHub Actions typing contract validates installation
**outside the source checkout**. The stable dataclass surface is checked with
`mypy --strict` using `typing.assert_type`.

```python
from ragnor import Document, RagnorBuilder, SearchResult
document: Document = Document(id="policy", text="Coverage")
```

`py.typed` declares inline package annotations to consuming type-checkers,
not that all dynamic adapters and integrations have strict/complete typing.
The SDK's internal `Any` APIs remain subject to later incremental tightening.
An adapter can be dynamically valid but still require typed contract tests
for its chosen port. This PR does not promise sound static typing for every
provider or remove Python's runtime extension patterns.

The base wheel does not depend on mypy; it is installed only in the CI
typing environment. Existing users retain all legacy imports and dynamic
integration options. License expression remains unchanged until #19 legal
review; do not imply an ELv2 migration has already occurred.
