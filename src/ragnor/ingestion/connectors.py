"""Public connector API; optional SDKs are imported only when used."""

from ragnor.ingestion.connector_core import (
    ChangeEvent,
    ChangeEventSource,
    DiscoveryPage,
    IncompleteScanError,
    IncrementalSourceConnector,
    LoadResult,
    SkipRecord,
    SourceConnector,
    SourceError,
    SourceInaccessibleError,
    SourceItem,
    SourceNotFoundError,
    SourceStatus,
    TransientSourceError,
    collect_pages,
)
from ragnor.ingestion.directory_connector import (
    DirectoryConnector,
    FilesystemWatcher,
    PeriodicReconciler,
)
from ragnor.ingestion.extractors import (
    Extractor,
    HTMLExtractor,
    PDFExtractor,
    TextExtractor,
    PageExtractionOutcome,
    PageExtractionStatus,
)
from ragnor.ingestion.extraction_types import (
    EvidenceAllowlistVerifier,
    ExtractionCell,
    ExtractionRegion,
    ExtractionTable,
    ExtractionWord,
    PDFExtractionError,
    PDFExtractionLimits,
    PageVerifier,
)
from ragnor.ingestion.pdf_extractors import (
    AutoPDFExtractor,
    PdfPlumberExtractor,
    TesseractPDFExtractor,
)
from ragnor.ingestion.http_connector import Credential, HttpApiConnector, WebsiteConnector
from ragnor.ingestion.object_connector import ObjectStorageClient, ObjectStorageConnector
from ragnor.ingestion.sources import ConnectorDocumentSource, DocumentSourceConnector

DirectorySource = DirectoryConnector
DirectoryDocumentSource = DirectoryConnector
DirectorySourceConnector = DirectoryConnector
FileSystemConnector = DirectoryConnector
LocalDirectoryConnector = DirectoryConnector
FilesystemSourceConnector = DirectoryConnector
HttpConnector = HttpApiConnector
ApiConnector = HttpApiConnector
GenericHttpConnector = HttpApiConnector
ObjectStorageSource = ObjectStorageConnector
ObjectStoreConnector = ObjectStorageConnector
S3Connector = ObjectStorageConnector

__all__ = [
    "EvidenceAllowlistVerifier",
    "AutoPDFExtractor",
    "ExtractionCell",
    "ExtractionRegion",
    "ExtractionTable",
    "ExtractionWord",
    "PDFExtractionError",
    "PDFExtractionLimits",
    "PageVerifier",
    "PageExtractionOutcome",
    "PageExtractionStatus",
    "PdfPlumberExtractor",
    "TesseractPDFExtractor",
    "ApiConnector",
    "ChangeEvent",
    "ChangeEventSource",
    "ConnectorDocumentSource",
    "Credential",
    "DirectoryConnector",
    "DirectoryDocumentSource",
    "DirectorySource",
    "DirectorySourceConnector",
    "DiscoveryPage",
    "DocumentSourceConnector",
    "Extractor",
    "FileSystemConnector",
    "FilesystemSourceConnector",
    "FilesystemWatcher",
    "GenericHttpConnector",
    "HTMLExtractor",
    "HttpApiConnector",
    "HttpConnector",
    "IncompleteScanError",
    "IncrementalSourceConnector",
    "LoadResult",
    "LocalDirectoryConnector",
    "ObjectStorageClient",
    "ObjectStorageConnector",
    "ObjectStorageSource",
    "ObjectStoreConnector",
    "PDFExtractor",
    "PeriodicReconciler",
    "S3Connector",
    "SkipRecord",
    "SourceConnector",
    "SourceError",
    "SourceInaccessibleError",
    "SourceItem",
    "SourceNotFoundError",
    "SourceStatus",
    "TextExtractor",
    "TransientSourceError",
    "WebsiteConnector",
    "collect_pages",
]
