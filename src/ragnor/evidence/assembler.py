"""Deterministic extractive evidence assembly with optional compression.

Takes SearchHit objects and a ContextBudget to produce a structured,
budget-constrained EvidenceBundle with deduplication, span merging,
source diversity balancing, and provenance tracking.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable
from collections.abc import Mapping

from ragnor.evidence.tokenizer import ApproximateTokenizer, Tokenizer
from ragnor.evidence.types import (
    ContextBudget,
    EvidenceBundle,
    EvidenceItem,
    EvidenceStatus,
    SearchHit,
)
from ragnor.core.types import SearchResult

logger = logging.getLogger(__name__)


@runtime_checkable
class EvidenceCompressor(Protocol):
    """Optional model-based evidence compression protocol.

    Compressors MUST retain verifiable provenance references (source_id,
    document_id, revision, offsets, and citation metadata).
    """

    def compress(self, item: EvidenceItem, target_tokens: int) -> EvidenceItem:
        """Compress the item's text while preserving all provenance identifiers."""
        ...


class EvidenceAssembler:
    """Assembles retrieved search hits into a structured EvidenceBundle.

    Features:
    - Deterministic extractive assembly (default).
    - Exact chunk deduplication.
    - Safe span merging for overlapping/contiguous same-source chunks.
    - Optional source diversity balancing.
    - Budget-constrained selection with reserved room for host instructions,
      user input, and expected answer tokens.
    - Distinct statuses for empty search, backend failure, insufficient budget,
      and stale index conditions.
    """

    def __init__(  # noqa: PLR0913
        self,
        tokenizer: Tokenizer | None = None,
        compressor: EvidenceCompressor | None = None,
        *,
        merge_overlapping_spans: bool = True,
        balance_sources: bool = False,
        max_per_source: int | None = None,
        exact_required: bool = False,
    ) -> None:
        """Initialize the assembler.

        Args:
            tokenizer: Tokenizer protocol implementation for budget accounting.
                Defaults to a conservative ApproximateTokenizer.
            compressor: Optional model-based compression implementation.
            merge_overlapping_spans: Whether to safely merge contiguous or
                overlapping spans from the same document revision.
            balance_sources: Whether to balance candidate selection across sources.
            max_per_source: Optional maximum number of evidence items per source.
        """
        self.tokenizer: Tokenizer = tokenizer or ApproximateTokenizer()
        if exact_required and not self.tokenizer.is_exact:
            raise ValueError("exact tokenizer is required for hard evidence context limits")
        self.exact_required = exact_required
        self._warned_approximation = False
        self.compressor = compressor
        self.merge_overlapping_spans = merge_overlapping_spans
        self.balance_sources = balance_sources
        self.max_per_source = max_per_source

    def _coerce_hits(
        self,
        hits: Sequence[SearchHit | Mapping[str, Any] | SearchResult],
    ) -> list[SearchHit]:
        """Convert input sequences of dicts or SearchResults into SearchHit objects."""
        coerced: list[SearchHit] = []
        for h in hits:
            if isinstance(h, SearchHit):
                coerced.append(h)
            elif isinstance(h, SearchResult):
                coerced.append(SearchHit.from_search_result(h))
            elif isinstance(h, Mapping):
                coerced.append(SearchHit.from_dict(h))
            else:
                coerced.append(SearchHit.from_retrieved_document(h))
        return coerced

    def _deduplicate(self, hits: list[SearchHit]) -> tuple[list[SearchHit], int]:
        """Deduplicate exact chunks based on document id and normalized text."""
        seen: set[tuple[str, str, str]] = set()
        deduped: list[SearchHit] = []
        dropped = 0

        for hit in hits:
            norm_text = " ".join(hit.document.strip().split())
            key = (hit.source_id, hit.document_id, norm_text)
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            deduped.append(hit)

        return deduped, dropped

    @staticmethod
    def _create_merged_hit(
        current: SearchHit,
        nxt: SearchHit,
        combined_text: str,
        end_offset: int | None = None,
    ) -> SearchHit:
        """Construct a merged SearchHit from two overlapping chunks."""
        return SearchHit(
            id=current.id,
            document=combined_text,
            metadata=dict(current.metadata),
            score=max(current.score or 0.0, nxt.score or 0.0),
            stage=current.stage or nxt.stage,
            source_id=current.source_id,
            document_id=current.document_id,
            revision=current.revision,
            title=current.title or nxt.title,
            uri=current.uri or nxt.uri,
            page=current.page or nxt.page,
            section=current.section or nxt.section,
            line=current.line or nxt.line,
            start_offset=current.start_offset,
            end_offset=end_offset if end_offset is not None else current.end_offset,
            freshness=current.freshness or nxt.freshness,
        )

    def _merge_spans(self, hits: list[SearchHit]) -> tuple[list[SearchHit], int]:
        """Merge overlapping or contiguous same-source spans when safe."""
        if not self.merge_overlapping_spans or len(hits) < 2:
            return hits, 0

        # Group by (source_id, document_id, revision)
        groups: dict[tuple[str, str, str], list[SearchHit]] = defaultdict(list)
        for h in hits:
            groups[(h.source_id, h.document_id, h.revision)].append(h)

        merged_hits: list[SearchHit] = []
        total_merged_count = 0

        for group in groups.values():
            if len(group) == 1:
                merged_hits.append(group[0])
                continue

            # Check if offsets are available for span merging
            has_offsets = all(
                h.start_offset is not None and h.end_offset is not None for h in group
            )

            if has_offsets:
                # Sort by start_offset
                group.sort(key=lambda x: (x.start_offset or 0, x.end_offset or 0))
                current = group[0]
                merged_group_chunks = [current.id]

                for nxt in group[1:]:
                    assert current.start_offset is not None
                    assert current.end_offset is not None
                    assert nxt.start_offset is not None
                    assert nxt.end_offset is not None

                    # Overlapping or contiguous spans
                    if nxt.start_offset <= current.end_offset:
                        total_merged_count += 1
                        overlap = current.end_offset - nxt.start_offset
                        if nxt.end_offset > current.end_offset:
                            # Suffix of nxt extends current
                            added_text = (
                                nxt.document[overlap:] if overlap < len(nxt.document) else ""
                            )
                            combined_text = current.document + added_text
                            new_end = nxt.end_offset
                        else:
                            combined_text = current.document
                            new_end = current.end_offset

                        merged_group_chunks.append(nxt.id)
                        current = self._create_merged_hit(
                            current, nxt, combined_text, end_offset=new_end
                        )
                        current.metadata["merged_chunk_ids"] = list(merged_group_chunks)
                    else:
                        merged_hits.append(current)
                        current = nxt
                        merged_group_chunks = [current.id]
                merged_hits.append(current)
            else:
                # Fallback: check direct prefix/suffix overlap
                current = group[0]
                for nxt in group[1:]:
                    merged = False
                    for overlap_len in range(min(len(current.document), len(nxt.document)), 20, -1):
                        if current.document.endswith(nxt.document[:overlap_len]):
                            combined_text = current.document + nxt.document[overlap_len:]
                            total_merged_count += 1
                            current = self._create_merged_hit(current, nxt, combined_text)
                            merged = True
                            break
                    if not merged:
                        merged_hits.append(current)
                        current = nxt
                merged_hits.append(current)

        return merged_hits, total_merged_count

    def _balance_source_diversity(self, hits: list[SearchHit]) -> list[SearchHit]:
        """Interleave hits across sources or enforce per-source caps."""
        if not self.balance_sources and self.max_per_source is None:
            return hits

        by_source: dict[str, list[SearchHit]] = defaultdict(list)
        for h in hits:
            by_source[h.source_id].append(h)

        if self.max_per_source is not None:
            for src in by_source:
                by_source[src] = by_source[src][: self.max_per_source]

        if not self.balance_sources:
            flattened = [item for src_items in by_source.values() for item in src_items]
            flattened.sort(key=lambda x: x.score or 0.0, reverse=True)
            return flattened

        # Interleave across sources round-robin
        balanced: list[SearchHit] = []
        source_queues = [list(items) for items in by_source.values()]
        source_queues.sort(key=lambda q: (q[0].score or 0.0) if q else 0.0, reverse=True)

        while any(source_queues):
            for q in source_queues:
                if q:
                    balanced.append(q.pop(0))

        return balanced

    def assemble(  # noqa: PLR0912
        self,
        hits: Sequence[SearchHit | Mapping[str, Any] | SearchResult],
        budget: ContextBudget,
        *,
        backend_status: str | None = None,
        backend_failure: bool = False,
        is_stale_index: bool = False,
    ) -> EvidenceBundle:
        """Assemble search hits into a budget-enforced EvidenceBundle.

        Args:
            hits: Candidate search results from retrieval.
            budget: Caller-supplied token budget with reserved capacities.
            backend_status: Status returned by backend retrieval, if known.
            backend_failure: Flag indicating a backend/store outage.
            is_stale_index: Flag indicating index revision staleness.

        Returns:
            An :class:`~ragnor.evidence.types.EvidenceBundle` carrying selected
            evidence items and execution status.
        """
        if not self.tokenizer.is_exact and not self._warned_approximation:
            logger.warning(
                "EvidenceAssembler uses an approximate tokenizer; "
                "estimated budgets do not guarantee model context limits"
            )
            self._warned_approximation = True

        # Check backend failure
        if backend_failure or backend_status in (
            "failure",
            "error",
            EvidenceStatus.BACKEND_FAILURE,
        ):
            return EvidenceBundle(
                items=[],
                status=EvidenceStatus.BACKEND_FAILURE,
                total_tokens=0,
                budget=budget,
                status_detail=(
                    "Backend retrieval failed; abstaining from ungrounded evidence assembly."
                ),
            )

        # Check stale index
        if is_stale_index or backend_status == EvidenceStatus.STALE_INDEX:
            return EvidenceBundle(
                items=[],
                status=EvidenceStatus.STALE_INDEX,
                total_tokens=0,
                budget=budget,
                status_detail="Retrieved results originate from a stale index generation.",
            )

        # Check budget validity
        if not budget.is_valid:
            return EvidenceBundle(
                items=[],
                status=EvidenceStatus.INSUFFICIENT_BUDGET,
                total_tokens=0,
                budget=budget,
                status_detail=(
                    f"Insufficient context budget: total budget ({budget.total_budget}) "
                    f"is fully consumed by reserved tokens ({budget.reserved_total})."
                ),
            )

        # Coerce input hits
        parsed_hits = self._coerce_hits(hits)
        if not parsed_hits or backend_status == EvidenceStatus.EMPTY_SEARCH:
            return EvidenceBundle(
                items=[],
                status=EvidenceStatus.EMPTY_SEARCH,
                total_tokens=0,
                budget=budget,
                status_detail="Search completed successfully but returned zero results.",
            )

        # Deduplicate exact chunks
        deduped_hits, dropped_duplicates = self._deduplicate(parsed_hits)

        # Merge overlapping same-source spans
        merged_hits, merged_spans = self._merge_spans(deduped_hits)

        # Rank within each source before interleaving; preserve diversity during selection.
        merged_hits.sort(key=lambda x: x.score if x.score is not None else -999.0, reverse=True)
        balanced_hits = self._balance_source_diversity(merged_hits)

        # Budget selection and optional compression
        available_tokens = budget.available_evidence_tokens
        running_tokens = 0
        selected_items: list[EvidenceItem] = []
        is_truncated = False

        for hit in balanced_hits:
            if (
                budget.max_evidence_items is not None
                and len(selected_items) >= budget.max_evidence_items
            ):
                is_truncated = True
                break

            token_count = self.tokenizer.count_tokens(hit.document)
            item = EvidenceItem(
                citation_label="",  # Will assign deterministically after selection
                source_id=hit.source_id,
                document_id=hit.document_id,
                revision=hit.revision,
                text=hit.document,
                title=hit.title,
                uri=hit.uri,
                page=hit.page,
                section=hit.section,
                line=hit.line,
                start_offset=hit.start_offset,
                end_offset=hit.end_offset,
                retrieval_stage=hit.stage,
                score=hit.score,
                freshness=hit.freshness,
                token_count=token_count,
                is_merged="merged_chunk_ids" in hit.metadata,
                merged_chunk_ids=hit.metadata.get("merged_chunk_ids", [hit.id]),
            )

            item.citation_label = f"[{len(selected_items) + 1}]"
            rendered_tokens = self._rendered_tokens([*selected_items, item], budget)
            if self.compressor is not None and rendered_tokens > available_tokens:
                framing_tokens = max(0, rendered_tokens - token_count)
                target_room = available_tokens - framing_tokens
                if target_room > 0:
                    try:
                        compressed = self.compressor.compress(
                            deepcopy(item), target_tokens=target_room
                        )
                        provenance = (
                            "source_id",
                            "document_id",
                            "revision",
                            "start_offset",
                            "end_offset",
                        )
                        if any(
                            getattr(compressed, key) != getattr(item, key) for key in provenance
                        ):
                            raise ValueError("Compressor changed evidence provenance")
                        compressed.citation_label = item.citation_label
                        compressed.token_count = self.tokenizer.count_tokens(compressed.text)
                        item = compressed
                        rendered_tokens = self._rendered_tokens([*selected_items, item], budget)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Evidence compression failed: %s", exc)

            if rendered_tokens <= available_tokens:
                selected_items.append(item)
                running_tokens = rendered_tokens
            else:
                is_truncated = True

        # Check if budget was too small to fit any candidate
        if not selected_items:
            return EvidenceBundle(
                items=[],
                status=EvidenceStatus.INSUFFICIENT_BUDGET,
                total_tokens=0,
                budget=budget,
                is_truncated=True,
                dropped_duplicates_count=dropped_duplicates,
                merged_spans_count=merged_spans,
                status_detail="Candidate evidence items exceeded available context budget.",
            )

        # Assign deterministic citation labels: [1], [2], ...
        for idx, item in enumerate(selected_items, start=1):
            item.citation_label = f"[{idx}]"

        return EvidenceBundle(
            items=selected_items,
            status=EvidenceStatus.SUCCESS,
            total_tokens=running_tokens,
            budget=budget,
            is_truncated=is_truncated,
            dropped_duplicates_count=dropped_duplicates,
            merged_spans_count=merged_spans,
            status_detail=None,
        )

    def _rendered_tokens(self, items: list[EvidenceItem], budget: ContextBudget) -> int:
        context = EvidenceBundle(items, EvidenceStatus.SUCCESS, 0, budget).format_context()
        return self.tokenizer.count_tokens(context)
