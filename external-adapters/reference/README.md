# ragnor-reference-adapters

Independently authored reference adapters for [Ragnor](https://github.com/eslam-aboutaleb/ragnor)
ports: chunker, tokenizer, embedder, event sink and document source. This
package proves each major adapter category is implementable by a third
party, against Ragnor's public API only, and passes the published
conformance suite (`ragnor.conformance`) without editing Ragnor.

The implementations are deterministic and intentionally simple: they are
conformance evidence and a starting template, not production components.

## Run the conformance suite against an installed Ragnor wheel

```bash
python -m pip install /path/to/ragnor-0.4.1-py3-none-any.whl  # a built or released Ragnor wheel
python -m pip install -e ".[test]"
python -m pytest
```

No Ragnor source checkout is required: the suite imports only
`ragnor.conformance` from the installed distribution and this package's own
adapters.

## Injecting the adapters

Adapters are plain instances, injected through the documented seams (for
example `RagnorBuilder.local(embedder).with_chunker(...)` or a production
composition root). This package registers no global state and Ragnor never
imports it.
