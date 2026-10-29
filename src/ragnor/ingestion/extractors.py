"""Optional text extraction; source provenance is the caller's responsibility."""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from enum import StrEnum
from html.parser import HTMLParser
from io import BytesIO
from typing import Any

from ragnor.ingestion.extraction_types import (
    ExtractionRegion,
    ExtractionTable,
    ExtractionWord,
    extraction_evidence_hash,
)


class PageExtractionStatus(StrEnum):
    EXTRACTED = "extracted"
    VERIFIED_BLANK = "verified_blank"
    NEEDS_OCR = "needs_ocr"
    PARTIAL = "partial"
    FAILED = "failed"
    POLICY_EXCLUDED = "policy_excluded"


@dataclass(frozen=True)
class PageExtractionOutcome:
    """Page evidence; diagnostics never include provider exception messages."""

    page: int
    status: PageExtractionStatus
    text: str = field(default="", repr=False)
    reason: str = ""
    has_images: bool = False
    has_drawing: bool = False
    provider: str = "pdf"
    version: str = "3"
    words: tuple[ExtractionWord, ...] = field(default=(), repr=False)
    tables: tuple[ExtractionTable, ...] = field(default=(), repr=False)
    regions: tuple[ExtractionRegion, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def evidence_sha256(self) -> str:
        return extraction_evidence_hash(self)

    @property
    def complete(self) -> bool:
        return self.status in {
            PageExtractionStatus.EXTRACTED,
            PageExtractionStatus.VERIFIED_BLANK,
            PageExtractionStatus.POLICY_EXCLUDED,
        }


class Extractor:
    """Base extractor identity/version/options."""

    id: str = "text"
    version: str = "1"
    options: dict[str, Any] = {}

    def extract(self, data: bytes, *, name: str = "") -> str:
        return data.decode("utf-8", errors="replace")


@dataclass
class TextExtractor(Extractor):
    id: str = "text"
    version: str = "1"
    options: dict[str, Any] = field(default_factory=dict)
    encoding: str = "utf-8"

    def __post_init__(self) -> None:
        self.options = {**self.options, "encoding": self.encoding}

    def extract(self, data: bytes, *, name: str = "") -> str:
        return data.decode(self.encoding)


@dataclass
class PDFExtractor(Extractor):
    """Extract typed PDF page evidence with optional pypdf; no built-in OCR.

    ``options['max_pages']`` bounds accepted documents (default 1000). Exceeding
    the bound raises rather than returning a silently truncated document.
    Connectors enforce ``policy`` (strict by default) with ``extract_outcomes``;
    legacy text views do not certify completeness.
    """

    id: str = "pdf"
    version: str = "3"
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.options = {
            "max_pages": 1000,
            "max_bytes": 50_000_000,
            "policy": "strict",
            **self.options,
        }
        limit = self.options["max_pages"]
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("PDF max_pages must be a positive integer")
        byte_limit = self.options["max_bytes"]
        if isinstance(byte_limit, bool) or not isinstance(byte_limit, int) or byte_limit < 1:
            raise ValueError("PDF max_bytes must be a positive integer")
        if self.options["policy"] not in {"strict", "review", "allow_partial"}:
            raise ValueError("PDF policy must be strict, review, or allow_partial")

    def extract_pages(self, data: bytes) -> list[tuple[int, str]]:
        """Legacy text view, including empty pages; does not certify completeness.

        Connectors use ``extract_outcomes`` to enforce the extraction policy.
        """
        reader = self._reader(data)
        return [
            (number, (page.extract_text() or "").strip())
            for number, page in enumerate(reader.pages, start=1)
        ]

    def _reader(self, data: bytes) -> Any:
        if len(data) > self.options["max_bytes"]:
            raise ValueError("PDF exceeds max_bytes")
        try:
            from pypdf import PdfReader  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "PDF extraction requires pypdf; install 'ragnor[pdf]' "
                "or 'pypdf'. OCR for image-only PDFs is not implemented."
            ) from exc

        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            raise ValueError("Encrypted PDF requires host decryption")
        count = len(reader.pages)
        limit = self.options["max_pages"]
        if count > limit:
            raise ValueError(f"PDF has {count} pages, exceeding max_pages={limit}")
        return reader

    def extract_outcomes(self, data: bytes) -> list[PageExtractionOutcome]:
        """Conservatively distinguish blank pages from unverified visual content.

        Text alone cannot verify raster content on the same page. Hosts needing
        OCR/layout verification should inject an extractor with this capability.
        """
        reader = self._reader(data)
        if not reader.pages:
            raise ValueError("PDF has no pages")
        return [self._page_outcome(page, number) for number, page in enumerate(reader.pages, 1)]

    def _page_outcome(self, page: Any, number: int) -> PageExtractionOutcome:
        try:
            width, height = float(page.mediabox.width), float(page.mediabox.height)
            if not all(math.isfinite(value) and value > 0 for value in (width, height)):
                raise ValueError("Invalid page dimensions")
            text = (page.extract_text() or "").strip()
            resources = page.get("/Resources", {})
            resources = resources.get_object() if hasattr(resources, "get_object") else resources
            xobjects = resources.get("/XObject", {})
            xobjects = xobjects.get_object() if hasattr(xobjects, "get_object") else xobjects
            images = any(obj.get_object().get("/Subtype") == "/Image" for obj in xobjects.values())
            contents = page.get_contents()
            operations = contents.operations if contents is not None else []
            # Unknown operators and form XObjects are not evidence of blankness.
            state_only = {
                b"q",
                b"Q",
                b"cm",
                b"w",
                b"J",
                b"j",
                b"M",
                b"d",
                b"ri",
                b"i",
                b"gs",
                b"CS",
                b"cs",
                b"SC",
                b"SCN",
                b"sc",
                b"scn",
                b"G",
                b"g",
                b"RG",
                b"rg",
                b"K",
                b"k",
            }
            content = any(op not in state_only for _, op in operations)
            paint = {
                b"S",
                b"s",
                b"f",
                b"F",
                b"f*",
                b"B",
                b"B*",
                b"b",
                b"b*",
                b"sh",
                b"INLINE IMAGE",
                b"Do",
            }
            drawing = any(op in paint for _, op in operations)
            text_only = state_only | {
                b"BT",
                b"ET",
                b"Tc",
                b"Tw",
                b"Tz",
                b"TL",
                b"Tf",
                b"Tr",
                b"Ts",
                b"Td",
                b"TD",
                b"Tm",
                b"T*",
                b"Tj",
                b"TJ",
                b"'",
                b'"',
                b"BMC",
                b"BDC",
                b"EMC",
                b"MP",
                b"DP",
            }
            unknown = any(op not in text_only for _, op in operations)
            forms = any(obj.get_object().get("/Subtype") != "/Image" for obj in xobjects.values())
            annotations = bool(page.get("/Annots"))
            if text and not (images or forms or annotations or unknown):
                status, reason = PageExtractionStatus.EXTRACTED, "text_extracted"
            elif text:
                status, reason = PageExtractionStatus.PARTIAL, "unverified_visual_content"
            elif not (images or forms or content or annotations):
                status, reason = PageExtractionStatus.VERIFIED_BLANK, "no_content_operations"
            else:
                status, reason = PageExtractionStatus.NEEDS_OCR, "nonblank_without_text"
            return PageExtractionOutcome(
                number, status, text, reason, images, drawing, self.id, self.version
            )
        except Exception:
            return PageExtractionOutcome(
                number,
                PageExtractionStatus.FAILED,
                reason="page_extraction_failed",
                provider=self.id,
                version=self.version,
            )

    def extract(self, data: bytes, *, name: str = "") -> str:
        return "\n\n".join(text for _, text in self.extract_pages(data) if text)


class _HTMLTextParser(HTMLParser):
    """Render visible prose, suppressing boilerplate before emitting markup."""

    _ignored = {"head", "script", "style", "nav", "footer", "aside", "noscript", "template"}
    _roles = {"navigation", "banner", "contentinfo", "complementary"}
    _void = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
    _blocks = {
        "address",
        "article",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "header",
        "hr",
        "li",
        "main",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tr",
        "ul",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    }
    _headings = {"h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.stack: list[tuple[str, bool, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        suppressed = (
            bool(self.stack and self.stack[-1][1])
            or tag in self._ignored
            or bool(set((attributes.get("role") or "").lower().split()) & self._roles)
            or "hidden" in attributes
            or (attributes.get("aria-hidden") or "").lower() == "true"
        )
        href = attributes.get("href") if tag == "a" else None
        if not suppressed:
            if tag in self._blocks:
                self.parts.append("\n")
            if tag in self._headings:
                self.parts.append("#" * int(tag[1]) + " ")
            if tag in {"td", "th"}:
                self.parts.append(" ")
        if tag not in self._void:
            self.stack.append((tag, suppressed, href))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in self._void:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        # Match the actual open element so stray closing tags cannot unhide text.
        index = next(
            (i for i in range(len(self.stack) - 1, -1, -1) if self.stack[i][0] == tag),
            None,
        )
        if index is None:
            return
        closing = self.stack[index:]
        del self.stack[index:]
        for closed_tag, suppressed, href in reversed(closing):
            if suppressed:
                continue
            if href and href.strip():
                self.parts.append(f" ({href.strip()})")
            if closed_tag in self._blocks:
                self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.stack or not self.stack[-1][1]:
            self.parts.append(re.sub(r"\s+", " ", data))


@dataclass
class HTMLExtractor(Extractor):
    """Conservative HTML boilerplate remover preserving headings and link targets."""

    id: str = "html"
    version: str = "2"
    options: dict[str, Any] = field(default_factory=dict)

    def extract(self, data: bytes, *, name: str = "") -> str:
        parser = _HTMLTextParser()
        parser.feed(data.decode("utf-8", errors="replace"))
        parser.close()
        lines = (
            re.sub(r"[^\S\n]+", " ", line).strip() for line in "".join(parser.parts).splitlines()
        )
        return "\n".join(line for line in lines if line)
