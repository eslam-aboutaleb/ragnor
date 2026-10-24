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
for its chosen port.

The repository also ships an **independently authored typed SDK consumer**
(`tests/typing/adapter_consumer.py`) containing host-supplied implementations
of the `Tokenizer`, `Chunker`, `EmbeddingFunction`, `VectorIndex`,
`LexicalIndex`, `DocumentStore`, `JobProcessor`, `JobEventSink`, and
`GenerationVectorReader` interfaces. GitHub Actions type-checks this consumer
against a wheel installed outside the source checkout, rather than inheriting
source-tree imports.

To verify the contract locally against a built wheel:

```bash
python -m pip install build mypy
python -m build --wheel
python -m pip install --no-deps dist/ragnor-*.whl
python -m mypy --strict --follow-imports=silent tests/typing/sdk_consumer.py tests/typing/adapter_consumer.py
```

The same job intentionally type-checks `invalid_sdk_consumer.py` and
requires mypy to reject its incompatible public API arguments. A passing
positive-only consumer would not establish that published annotations are
actually enforceable. This PR does not promise sound static typing for every
provider or remove Python's runtime extension patterns.

The base wheel does not depend on mypy; it is installed only in the CI
typing environment. Existing users retain all legacy imports and dynamic
integration options. Current source distributions declare `Elastic-2.0` and
include LICENSE and NOTICE. Previously published artifacts retain their original license terms.
