# Typed SDK distribution — PEP 561

Ragnor includes inline type annotations and a `py.typed` marker in its wheel
and source distribution. Type checkers can use these annotations without
installing the source checkout.

```python
from ragnor import Document, RagnorBuilder, SearchResult

document: Document = Document(id="policy", text="Coverage")
```

To check an application with mypy:

```bash
python -m pip install ragnor mypy
python -m mypy --strict your_application.py
```

Mypy is a development tool, not a Ragnor runtime dependency. Dynamic adapter
interfaces can contain `Any`; add annotations to host implementations and
check them against their chosen protocols. The package does not require a
particular host framework or type checker.

Current distributions declare `Elastic-2.0` and include LICENSE and NOTICE.
Previously published artifacts retain their original license terms.
