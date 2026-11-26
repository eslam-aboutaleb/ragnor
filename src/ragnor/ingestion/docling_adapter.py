"""Optional Docling adapter emitting ragnor page evidence into the verifier gate.

Docling (MIT licensed) converts layout and tables with local models; the
default installation pulls PyTorch and is deliberately an opt-in extra.
Conversion runs inside the bounded SDK worker, which never downloads models:
hosts provision them ahead of time and pass ``artifacts_path``.
"""

from __future__ import annotations

from pathlib import Path

from ragnor.ingestion.extraction_types import PDFExtractionLimits, PageVerifier
from ragnor.ingestion.pdf_extractors import _PDFAdapter


class DoclingPDFExtractor(_PDFAdapter):
    """Convert PDF pages with Docling inside the sandboxed SDK worker.

    Docling output is unverified: converted pages stay PARTIAL until a host
    verifier accepts their exact evidence. Pages Docling leaves empty keep
    their pypdf baseline status, so unknown pages still fail closed. Table
    geometry is coarse (table-level boxes, no word positions).
    """

    id = "docling"
    version = "1"
    _engine = "docling"

    def __init__(  # noqa: PLR0913 -- explicit optional provider configuration
        self,
        *,
        artifacts_path: str | Path | None = None,
        do_ocr: bool = False,
        do_table_structure: bool = True,
        limits: PDFExtractionLimits | None = None,
        verifier: PageVerifier | None = None,
        policy: str = "strict",
    ) -> None:
        if artifacts_path is not None and not isinstance(artifacts_path, (str, Path)):
            raise ValueError("artifacts_path must be a path or None")
        if type(do_ocr) is not bool or type(do_table_structure) is not bool:
            raise ValueError("Docling pipeline switches must be boolean")
        self._configure(
            {
                "artifacts_path": str(Path(artifacts_path).resolve()) if artifacts_path else None,
                "do_ocr": do_ocr,
                "do_table_structure": do_table_structure,
            },
            limits,
            verifier,
            policy,
        )
