"""Provider-neutral structured extraction contracts (zero dependencies).

Importing this package pulls in no engine, network client or optional
extra. Providers live in host code or third-party packages and are
registered explicitly on a :class:`ProviderRegistry`.
"""

from __future__ import annotations

from ragnor.extraction.adapters import from_page_outcomes, region_metadata, to_raw_documents
from ragnor.extraction.citation import (
    Citation,
    ResolvedCitation,
    citation_from_metadata,
    resolve_citation,
)
from ragnor.extraction.errors import (
    ExtractionConfigError,
    ExtractionError,
    ExtractionLimitError,
)
from ragnor.extraction.extractor import StructuredExtractor
from ragnor.extraction.figures import link_captions
from ragnor.extraction.fingerprint import (
    extractor_fingerprint,
    fingerprint_hash,
    provider_fingerprint,
    safe_provider_options,
)
from ragnor.extraction.ids import deterministic_region_id
from ragnor.extraction.layout import DEFAULT_LABELS, RegionNormalizer, normalize_docling
from ragnor.extraction.ports import (
    ArtifactStore,
    DocumentExtractor,
    ExtractionProvider,
    LayoutDetector,
    OCREngine,
    PageProfiler,
    PageRenderer,
    RegionExtractor,
    Verifier,
)
from ragnor.extraction.reading_order import reconcile_reading_order
from ragnor.extraction.registry import (
    ENTRY_POINT_GROUP,
    Capability,
    Mode,
    ProviderRegistry,
    ProviderSpec,
)
from ragnor.extraction.routes import apply_routes
from ragnor.extraction.serializers import (
    from_json,
    region_text,
    result_to_markdown,
    result_to_text,
    table_to_html,
    table_to_markdown,
    table_to_text,
    to_json,
)
from ragnor.extraction.tables import (
    link_table_continuations,
    logical_table,
    merge_declared_table_boxes,
    split_table_at,
    stack_tables,
    table_from_cell_dicts,
    table_from_grid,
)
from ragnor.extraction.types import (
    SCHEMA_VERSION,
    AssetRef,
    AttemptStatus,
    BoundingBox,
    Cell,
    CoordinateSpace,
    DerivedContent,
    DerivedKind,
    DocumentProfile,
    ExtractionAttempt,
    ExtractionResult,
    ExtractionStatus,
    HeaderScope,
    OrderState,
    Page,
    PageStatus,
    ReasonCode,
    Region,
    RegionKind,
    Table,
    TextSpan,
    Unit,
)

__all__ = [
    "Citation",
    "DEFAULT_LABELS",
    "RegionNormalizer",
    "ResolvedCitation",
    "apply_routes",
    "citation_from_metadata",
    "deterministic_region_id",
    "link_captions",
    "link_table_continuations",
    "logical_table",
    "merge_declared_table_boxes",
    "normalize_docling",
    "reconcile_reading_order",
    "resolve_citation",
    "split_table_at",
    "stack_tables",
    "table_from_cell_dicts",
    "table_from_grid",
    "ENTRY_POINT_GROUP",
    "SCHEMA_VERSION",
    "ArtifactStore",
    "AssetRef",
    "AttemptStatus",
    "BoundingBox",
    "Capability",
    "Cell",
    "CoordinateSpace",
    "DerivedContent",
    "DerivedKind",
    "DocumentExtractor",
    "DocumentProfile",
    "ExtractionAttempt",
    "ExtractionConfigError",
    "ExtractionError",
    "ExtractionLimitError",
    "ExtractionProvider",
    "ExtractionResult",
    "ExtractionStatus",
    "HeaderScope",
    "LayoutDetector",
    "Mode",
    "OCREngine",
    "OrderState",
    "Page",
    "PageProfiler",
    "PageRenderer",
    "PageStatus",
    "ProviderRegistry",
    "ProviderSpec",
    "ReasonCode",
    "Region",
    "RegionExtractor",
    "RegionKind",
    "StructuredExtractor",
    "Table",
    "TextSpan",
    "Unit",
    "Verifier",
    "extractor_fingerprint",
    "fingerprint_hash",
    "from_json",
    "from_page_outcomes",
    "provider_fingerprint",
    "region_metadata",
    "region_text",
    "result_to_markdown",
    "result_to_text",
    "safe_provider_options",
    "table_to_html",
    "table_to_markdown",
    "table_to_text",
    "to_json",
    "to_raw_documents",
]
