"""Optional text extraction; source provenance is the caller's responsibility."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from io import BytesIO
from typing import Any


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
    """Extract PDF pages with optional pypdf; image-only pages need external OCR.

    ``options['max_pages']`` bounds accepted documents (default 1000). Exceeding
    the bound raises rather than returning a silently truncated document.
    """

    id: str = "pdf"
    version: str = "2"
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.options = {"max_pages": 1000, **self.options}
        limit = self.options["max_pages"]
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("PDF max_pages must be a positive integer")

    def extract_pages(self, data: bytes) -> list[tuple[int, str]]:
        """Return every real page as ``(1-based number, text)``, including blanks."""
        try:
            from pypdf import PdfReader  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "PDF extraction requires pypdf; install 'ragnor[pdf]' "
                "or 'pypdf'. OCR for image-only PDFs is not implemented."
            ) from exc

        reader = PdfReader(BytesIO(data))
        count = len(reader.pages)
        limit = self.options["max_pages"]
        if count > limit:
            raise ValueError(f"PDF has {count} pages, exceeding max_pages={limit}")
        return [
            (number, (page.extract_text() or "").strip())
            for number, page in enumerate(reader.pages, start=1)
        ]

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
