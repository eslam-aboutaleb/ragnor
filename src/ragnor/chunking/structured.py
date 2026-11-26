"""Structure-aware chunking with source-span provenance (EX-06).

:class:`StructuredDocumentChunker` chunks a typed, reading-ordered list of
regions (headings, paragraphs, tables, figures, formulas, captions) instead of
raw text. It keeps each table, figure and formula atomic together with its
caption, splits oversized tables only on complete row boundaries (repeating the
header, caption and section path and marking continuations), and maps every
chunk to the exact page/region/row/cell sources it came from.

Two kinds of provenance are kept apart on purpose:

* ``source_spans`` in chunk metadata: the exact source regions whose content
  appears in the chunk body. Several regions are never merged into one bbox.
* ``context_spans`` / ``retrieval_context``: header cells, caption and section
  path repeated on every table partition so the chunk is self-describing. These
  are retrieval context, not additional sources.

The input types here are small, dependency-free dataclasses. They are the seam
for the typed structured-document model of EX-03 (#61): an EX-03 document is
adapted into :class:`StructuredDocument`, and :func:`structured_from_pages`
adapts today's :class:`~ragnor.ingestion.extraction_types.PageExtractionOutcome`.

The existing ``Chunker.chunk(text)`` API is unchanged: :meth:`chunk` parses
Markdown (headings H1-H6, pipe tables, paragraphs) into a structured document
and calls :meth:`chunk_document`, so the chunker can be dropped into the current
ingestion pipeline as a fallback-compatible replacement for
:class:`~ragnor.chunking.markdown.MarkdownSectionChunker`.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ragnor.core.types import Chunk
from ragnor.evidence.tokenizer import ApproximateTokenizer, Tokenizer

BBox = tuple[float, float, float, float]

CHUNKER_VERSION = "structured-v1"
"""Bump when chunk boundaries, text projection or identity derivation change."""

REGION_KINDS = frozenset({"heading", "paragraph", "list", "table", "figure", "formula", "caption"})
_ATOMIC_KINDS = frozenset({"table", "figure", "formula"})
_ASSET_KEYS = frozenset({"html", "json", "crop", "image"})


class StructuredChunkingError(ValueError):
    """Input or budget problem; messages never include document content."""


class OversizeUnitError(StructuredChunkingError):
    """A row or word cannot fit the token budget and would have to be split."""


@dataclass(frozen=True)
class TableRow:
    """One body row. ``cell_bboxes`` (optional) maps each cell to its page box."""

    cells: tuple[str, ...]
    page: int | None = None
    bbox: BBox | None = None
    cell_bboxes: tuple[BBox | None, ...] | None = None


@dataclass(frozen=True)
class StructuredRegion:
    """One reading-ordered region of a document.

    ``level`` applies to headings (1-6). ``header`` and ``rows`` apply to
    tables. ``caption_for`` on a ``caption`` region names the ``region_id`` it
    captions. ``assets`` holds storage references (URIs or keys) to full
    HTML/JSON renderings or immutable figure crops, never image bytes.
    """

    region_id: str
    kind: str
    text: str = ""
    page: int | None = None
    bbox: BBox | None = None
    level: int = 0
    caption: str | None = None
    caption_for: str | None = None
    header: tuple[str, ...] = ()
    rows: tuple[TableRow, ...] = ()
    assets: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.region_id or not isinstance(self.region_id, str):
            raise StructuredChunkingError("region_id must be a non-empty string")
        if self.kind not in REGION_KINDS:
            raise StructuredChunkingError(f"unknown region kind: {self.kind!r}")
        if self.kind == "heading" and not 1 <= self.level <= 6:
            raise StructuredChunkingError("heading level must be between 1 and 6")
        if self.kind == "table" and not self.rows and not self.header:
            raise StructuredChunkingError("table regions need a header or rows")
        for key, value in self.assets.items():
            if key not in _ASSET_KEYS or not isinstance(value, str):
                raise StructuredChunkingError(
                    "assets must map html/json/crop/image to string storage references"
                )
            if len(value) > 2048:
                raise StructuredChunkingError("asset reference is too long to be a reference")


@dataclass(frozen=True)
class StructuredDocument:
    """Reading-ordered regions plus the identity inputs chunk ids depend on.

    ``fingerprint`` identifies the source revision and the normalizer output
    (for example the evidence hash). ``access`` is copied unchanged onto every
    chunk so tenant/authorization decisions stay possible at retrieval time.
    """

    document_id: str
    fingerprint: str
    regions: tuple[StructuredRegion, ...]
    access: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.document_id or not self.fingerprint:
            raise StructuredChunkingError("document_id and fingerprint are required")
        ids = [region.region_id for region in self.regions]
        if len(set(ids)) != len(ids):
            raise StructuredChunkingError("region ids must be unique within a document")


@dataclass
class _Unit:
    """An indivisible piece of content with its exact sources."""

    kind: str
    text: str
    spans: list[dict[str, Any]]
    region: StructuredRegion | None = None
    caption_region: StructuredRegion | None = None


def _bbox_list(bbox: BBox | None) -> list[float] | None:
    return None if bbox is None else [float(v) for v in bbox]


def _span(region: StructuredRegion, **extra: Any) -> dict[str, Any]:
    span: dict[str, Any] = {
        "region_id": region.region_id,
        "kind": region.kind,
        "page": region.page,
        "bbox": _bbox_list(region.bbox),
    }
    span.update(extra)
    return span


def _md_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _md_row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(_md_cell(c) for c in cells) + " |"


class StructuredDocumentChunker:
    """Chunk a :class:`StructuredDocument` under a token budget.

    Args:
        tokenizer: Token counter. Defaults to the conservative
            :class:`ApproximateTokenizer`.
        max_tokens: Budget for the full chunk text, including the repeated
            section path, caption and header.
        strict: When true a hard limit is promised, so the tokenizer must be
            exact (same rule as :class:`TokenAwareChunker`) and any unit that
            cannot fit raises :class:`OversizeUnitError`. When false the
            approximate counter may be used; metadata then records
            ``token_count_exact: False`` so an estimate is never presented as a
            guarantee, and an atomic unit larger than the budget is emitted
            alone with ``exceeds_budget: True``.
    """

    def __init__(
        self,
        tokenizer: Tokenizer | None = None,
        *,
        max_tokens: int = 512,
        strict: bool = False,
        source_name: str = "document",
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        tokenizer = tokenizer if tokenizer is not None else ApproximateTokenizer()
        if strict and not tokenizer.is_exact:
            raise ValueError(
                "strict mode requires an exact tokenizer; approximate counting "
                "cannot guarantee a hard token limit"
            )
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.strict = strict
        self.source_name = source_name

    # -- Chunker protocol -------------------------------------------------

    @property
    def version(self) -> str:
        return CHUNKER_VERSION

    @property
    def config(self) -> dict[str, Any]:
        return {
            "chunk_size": self.max_tokens,
            "overlap": 0,
            "unit": "model_tokens" if self.tokenizer.is_exact else "approximate_tokens",
            "tokenizer": self.tokenizer.name,
            "is_exact": self.tokenizer.is_exact,
            "strict": self.strict,
        }

    def chunk(self, text: str) -> list[Chunk]:
        """Chunk Markdown text (the existing ``Chunker`` API)."""
        document = markdown_to_structured(text, source_name=self.source_name)
        return self.chunk_document(document)

    # -- structured API ---------------------------------------------------

    def chunk_document(
        self,
        document: StructuredDocument,
        token_budget: int | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> list[Chunk]:
        """Chunk a structured document.

        ``token_budget`` and ``tokenizer`` override the constructor values for
        this call; the strict-mode exactness rule is re-checked for an override.
        """
        if token_budget is not None or tokenizer is not None:
            run = StructuredDocumentChunker(
                tokenizer or self.tokenizer,
                max_tokens=token_budget or self.max_tokens,
                strict=self.strict,
                source_name=self.source_name,
            )
            return run.chunk_document(document)
        return list(self._chunk(document))

    def _count(self, text: str) -> int:
        return self.tokenizer.count_tokens(text)

    def _chunk(self, document: StructuredDocument) -> Iterable[Chunk]:
        by_id = {region.region_id: region for region in document.regions}
        captions: dict[str, StructuredRegion] = {}
        for region in document.regions:
            if region.kind == "caption" and region.caption_for:
                target = by_id.get(region.caption_for)
                if target is None or target.kind not in _ATOMIC_KINDS:
                    raise StructuredChunkingError("caption_for must name a table/figure/formula")
                if target.region_id in captions:
                    raise StructuredChunkingError("a region can have only one caption region")
                captions[target.region_id] = region
        consumed = {c.region_id for c in captions.values()}

        path: list[tuple[int, str]] = []
        path_regions: list[StructuredRegion] = []
        opened = False
        pending: list[_Unit] = []
        out: list[Chunk] = []
        emitted_for_section = False

        def section_path() -> list[str]:
            return [title for _, title in path]

        def flush() -> None:
            nonlocal pending, emitted_for_section
            if pending:
                out.extend(self._pack(document, section_path(), path_regions, pending))
                emitted_for_section = True
            pending = []

        def close_section(child_follows: bool = False) -> None:
            nonlocal emitted_for_section
            flush()
            # A heading directly followed by a deeper heading is already
            # represented in the child's section path.
            if opened and not emitted_for_section and not child_follows:
                # Heading-only section: keep it findable instead of dropping it.
                out.extend(self._pack(document, section_path(), path_regions, []))
            emitted_for_section = False

        for region in document.regions:
            if region.region_id in consumed:
                continue
            if region.kind == "heading":
                close_section(child_follows=bool(path) and region.level > path[-1][0])
                while path and path[-1][0] >= region.level:
                    path.pop()
                    path_regions.pop()
                path.append((region.level, region.text.strip()))
                path_regions.append(region)
                opened = True
                continue
            pending.append(self._unit(region, captions.get(region.region_id)))
        close_section()
        return out

    def _unit(self, region: StructuredRegion, caption: StructuredRegion | None) -> _Unit:
        if region.kind == "table":
            return _Unit("table", "", [], region, caption)
        caption_text = (caption.text.strip() if caption else region.caption or "").strip()
        body = region.text.strip()
        if region.kind in {"figure", "formula"}:
            parts = [p for p in (caption_text, body) if p]
            text = "\n".join(parts)
            if region.kind == "figure" and not text:
                text = f"[figure {region.region_id}]"
        else:
            text = body
        spans = [_span(region)]
        if caption is not None:
            spans.append(_span(caption))
        return _Unit(region.kind, text, spans, region, caption)

    # -- packing ----------------------------------------------------------

    def _prefix(self, path: Sequence[str]) -> str:
        return " > ".join(path)

    def _compose(self, path: Sequence[str], body: str) -> str:
        prefix = self._prefix(path)
        return f"{prefix}\n\n{body}" if prefix and body else (prefix or body)

    def _pack(
        self,
        document: StructuredDocument,
        path: Sequence[str],
        path_regions: Sequence[StructuredRegion],
        units: list[_Unit],
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        section_spans = [_span(r, role="section_path") for r in path_regions]
        group: list[_Unit] = []

        def emit_group() -> None:
            nonlocal group
            if not group:
                return
            body = "\n\n".join(u.text for u in group)
            chunks.append(
                self._make(
                    document,
                    path,
                    self._compose(path, body),
                    kind=group[0].kind if len(group) == 1 else "mixed",
                    spans=[s for u in group for s in u.spans],
                    context_spans=list(section_spans),
                    boundary={"units": [u.region.region_id for u in group if u.region]},
                    extra={},
                    assets=[u.region for u in group if u.region],
                )
            )
            group = []

        if not units:
            if path_regions:
                last = path_regions[-1]
                chunks.append(
                    self._make(
                        document,
                        path,
                        self._compose(path, ""),
                        kind="heading",
                        spans=[_span(last)],
                        context_spans=section_spans[:-1],
                        boundary={"heading": last.region_id},
                        extra={},
                        assets=[],
                    )
                )
            return chunks

        for unit in units:
            if unit.kind == "table":
                emit_group()
                chunks.extend(self._table_chunks(document, path, unit, section_spans))
                continue
            pieces = self._split_text_unit(path, unit)
            if len(pieces) > 1:
                # An oversized paragraph is split on word boundaries; every
                # piece keeps the original region as its (exact) source.
                emit_group()
                for part_index, piece in enumerate(pieces):
                    spans = [{**s, "part": part_index, "parts": len(pieces)} for s in unit.spans]
                    group.append(_Unit(unit.kind, piece, spans, unit.region, unit.caption_region))
                    emit_group()
                continue
            candidate = "\n\n".join([u.text for u in group] + [unit.text])
            if group and self._count(self._compose(path, candidate)) > self.max_tokens:
                emit_group()
            group.append(unit)
        emit_group()
        return chunks

    def _split_text_unit(self, path: Sequence[str], unit: _Unit) -> list[str]:
        """Return one piece if it fits, else greedy word groups that fit."""
        if self._count(self._compose(path, unit.text)) <= self.max_tokens:
            return [unit.text]
        if unit.kind in _ATOMIC_KINDS:
            if self.strict:
                raise OversizeUnitError("an atomic figure or formula exceeds the strict budget")
            return [unit.text]  # emitted alone, flagged by _make
        words = unit.text.split(" ")
        pieces: list[str] = []
        current: list[str] = []
        for word in words:
            trial = " ".join([*current, word])
            if current and self._count(self._compose(path, trial)) > self.max_tokens:
                pieces.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
            too_big = self._count(self._compose(path, " ".join(current))) > self.max_tokens
            if too_big and self.strict:
                raise OversizeUnitError("a single word exceeds the strict budget")
        if current:
            pieces.append(" ".join(current))
        return pieces

    def _table_chunks(
        self,
        document: StructuredDocument,
        path: Sequence[str],
        unit: _Unit,
        section_spans: list[dict[str, Any]],
    ) -> list[Chunk]:
        table = unit.region
        assert table is not None
        caption = (
            unit.caption_region.text.strip()
            if unit.caption_region is not None
            else (table.caption or "").strip()
        )
        header_lines: list[str] = []
        if table.header:
            header_lines = [_md_row(table.header), _md_row(["---"] * len(table.header))]
        rows = list(table.rows)
        context_spans = [_span(table, role="table_header")] if table.header else []
        if unit.caption_region is not None:
            context_spans.append(_span(unit.caption_region, role="caption"))

        def render(first: int, last: int, part: int, parts: int) -> str:
            lines: list[str] = []
            if caption:
                lines.append(caption)
            if parts > 1:
                if part:
                    lines.append(f"(continued: part {part + 1} of {parts})")
                else:
                    lines.append(f"(part 1 of {parts})")
            lines.extend(header_lines)
            lines.extend(_md_row(r.cells) for r in rows[first:last])
            return self._compose(path, "\n".join(lines))

        # Partition rows greedily on complete-row boundaries.
        ranges: list[tuple[int, int]] = []
        start = 0
        if not rows or self._count(render(0, len(rows), 0, 1)) <= self.max_tokens:
            ranges.append((0, len(rows)))
            start = len(rows)
        while start < len(rows):
            end = start
            while end < len(rows):
                # reserve space for the longest continuation marker
                trial = render(start, end + 1, 1, 9999)
                if self._count(trial) > self.max_tokens:
                    break
                end += 1
            if end == start:
                if self.strict:
                    raise OversizeUnitError(
                        "a table row with its header and caption exceeds the strict budget"
                    )
                end = start + 1  # emit the single row alone, flagged below
            ranges.append((start, end))
            start = end

        parts = len(ranges)
        chunks: list[Chunk] = []
        for part, (first, last) in enumerate(ranges):
            text = render(first, last, part, parts)
            spans = [self._row_span(table, index, rows[index]) for index in range(first, last)]
            if not rows:
                spans = [_span(table)]
            extra: dict[str, Any] = {
                "table": {
                    "region_id": table.region_id,
                    "row_start": first,
                    "row_end": last,
                    "total_rows": len(rows),
                    "header": list(table.header),
                    "caption": caption or None,
                },
                "retrieval_projection": {
                    "type": "table",
                    "caption": caption or None,
                    "header": list(table.header),
                    "section_path": list(path),
                },
            }
            if parts > 1:
                extra["continuation"] = {
                    "part": part + 1,
                    "parts": parts,
                    "continues_previous": part > 0,
                    "continues_next": part < parts - 1,
                }
            chunks.append(
                self._make(
                    document,
                    path,
                    text,
                    kind="table",
                    spans=spans,
                    context_spans=[*section_spans, *context_spans],
                    boundary={"table": table.region_id, "rows": [first, last]},
                    extra=extra,
                    assets=[table],
                )
            )
        return chunks

    @staticmethod
    def _row_span(table: StructuredRegion, index: int, row: TableRow) -> dict[str, Any]:
        span: dict[str, Any] = {
            "region_id": table.region_id,
            "kind": "table_row",
            "row": index,
            "page": row.page if row.page is not None else table.page,
            "bbox": _bbox_list(row.bbox if row.bbox is not None else table.bbox),
        }
        if row.cell_bboxes is not None:
            span["cells"] = [
                {"col": col, "bbox": _bbox_list(box)} for col, box in enumerate(row.cell_bboxes)
            ]
        return span

    # -- chunk construction -------------------------------------------------

    def _chunk_id(self, document: StructuredDocument, boundary: Mapping[str, Any]) -> str:
        payload = {
            "fingerprint": document.fingerprint,
            "document": document.document_id,
            "chunker": CHUNKER_VERSION,
            "config": self.config,
            "boundary": boundary,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        ).hexdigest()
        return f"sdc-{digest[:32]}"

    def _make(  # noqa: PLR0913
        self,
        document: StructuredDocument,
        path: Sequence[str],
        text: str,
        *,
        kind: str,
        spans: list[dict[str, Any]],
        context_spans: list[dict[str, Any]],
        boundary: dict[str, Any],
        extra: dict[str, Any],
        assets: Sequence[StructuredRegion],
    ) -> Chunk:
        tokens = self._count(text)
        exceeds = tokens > self.max_tokens
        if exceeds and self.strict:
            raise OversizeUnitError("chunk exceeds the strict token budget")
        # Disambiguate repeated units (e.g. split paragraph parts) by their spans.
        boundary = {**boundary, "spans": [_stable_span_key(s) for s in spans]}
        pages = sorted({s["page"] for s in [*spans, *context_spans] if s.get("page") is not None})
        storage: dict[str, dict[str, str]] = {}
        for region in assets:
            if region.assets:
                storage[region.region_id] = dict(region.assets)
        metadata: dict[str, Any] = {
            "chunker": CHUNKER_VERSION,
            "source": self.source_name,
            "document_id": document.document_id,
            "fingerprint": document.fingerprint,
            "kind": kind,
            "section_path": list(path),
            "pages": pages,
            "source_spans": spans,
            "context_spans": context_spans,
            "retrieval_context": self._prefix(path),
            "token_count": tokens,
            "token_count_exact": bool(self.tokenizer.is_exact),
            "tokenizer": self.tokenizer.name,
            "max_tokens": self.max_tokens,
        }
        if exceeds:
            metadata["exceeds_budget"] = True
        if storage:
            metadata["storage_refs"] = storage
        if document.access:
            metadata["access"] = dict(document.access)
        metadata.update(extra)
        return Chunk(id=self._chunk_id(document, boundary), text=text, metadata=metadata)


def _stable_span_key(span: Mapping[str, Any]) -> list[Any]:
    return [
        span.get("region_id"),
        span.get("row"),
        span.get("part"),
        span.get("role"),
    ]


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")


def _split_row(line: str) -> tuple[str, ...]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|") and not stripped.endswith("\\|"):
        stripped = stripped[:-1]
    cells = re.split(r"(?<!\\)\|", stripped)
    return tuple(c.strip().replace("\\|", "|") for c in cells)


def markdown_to_structured(text: str, *, source_name: str = "document") -> StructuredDocument:
    """Parse Markdown into a :class:`StructuredDocument` (fallback input path).

    Handles ATX headings H1-H6, pipe tables with a separator row, and
    blank-line separated paragraphs. Fenced code blocks stay in one paragraph
    and are not scanned for headings. No page or bbox information exists.
    """
    regions: list[StructuredRegion] = []
    lines = text.splitlines()
    i = 0
    paragraph: list[str] = []
    in_fence = False

    def next_id() -> str:
        return f"md-{len(regions)}"

    def flush_paragraph() -> None:
        if paragraph:
            body = "\n".join(paragraph).strip()
            if body:
                regions.append(StructuredRegion(next_id(), "paragraph", text=body))
        paragraph.clear()

    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            paragraph.append(line)
            i += 1
            continue
        if in_fence:
            paragraph.append(line)
            i += 1
            continue
        heading = _HEADING.match(line)
        if heading:
            flush_paragraph()
            regions.append(
                StructuredRegion(
                    next_id(), "heading", text=heading.group(2), level=len(heading.group(1))
                )
            )
            i += 1
            continue
        if "|" in line and i + 1 < len(lines) and _SEPARATOR.match(lines[i + 1]):
            flush_paragraph()
            header = _split_row(line)
            i += 2
            rows: list[TableRow] = []
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                rows.append(TableRow(_split_row(lines[i])))
                i += 1
            regions.append(StructuredRegion(next_id(), "table", header=header, rows=tuple(rows)))
            continue
        if not line.strip():
            flush_paragraph()
        else:
            paragraph.append(line)
        i += 1
    flush_paragraph()
    fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return StructuredDocument(
        document_id=source_name, fingerprint=fingerprint, regions=tuple(regions)
    )


def structured_from_pages(
    pages: Sequence[Any],
    *,
    document_id: str,
    fingerprint: str,
    header_rows: int = 1,
    access: Mapping[str, str] | None = None,
) -> StructuredDocument:
    """Adapt ``PageExtractionOutcome`` pages (EX-01) into a structured document.

    Page text becomes paragraphs (blank-line separated); each extracted table
    becomes a table region whose first ``header_rows`` grid rows form the
    header. Table cell boxes are carried through so rows keep cell citations.
    Tables extracted by a provider are normalized-page coordinates, as in
    :mod:`ragnor.ingestion.extraction_types`.
    """
    regions: list[StructuredRegion] = []
    for page in sorted(pages, key=lambda p: p.page):
        for index, block in enumerate(re.split(r"\n\s*\n", page.text or "")):
            if block.strip():
                regions.append(
                    StructuredRegion(
                        f"p{page.page}-text{index}", "paragraph", text=block.strip(), page=page.page
                    )
                )
        for t_index, table in enumerate(page.tables):
            grid: list[list[Any]] = [[None] * table.cols for _ in range(table.rows)]
            for cell in table.cells:
                grid[cell.row][cell.col] = cell
            texts = [[c.text if c is not None else "" for c in row] for row in grid]
            boxes = [[c.bbox if c is not None else None for c in row] for row in grid]
            header = tuple(texts[0]) if header_rows and texts else ()
            body = [
                TableRow(
                    cells=tuple(texts[r]),
                    page=page.page,
                    cell_bboxes=tuple(boxes[r]),
                )
                for r in range(header_rows, table.rows)
            ]
            regions.append(
                StructuredRegion(
                    f"p{page.page}-table{t_index}",
                    "table",
                    page=page.page,
                    bbox=table.bbox,
                    header=header,
                    rows=tuple(body),
                )
            )
    return StructuredDocument(
        document_id=document_id,
        fingerprint=fingerprint,
        regions=tuple(regions),
        access=dict(access or {}),
    )
