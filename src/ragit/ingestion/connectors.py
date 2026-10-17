"""Public connector API; optional SDKs are imported only when used."""

from ragit.ingestion.connector_core import (
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
from ragit.ingestion.directory_connector import (
    DirectoryConnector,
    FilesystemWatcher,
    PeriodicReconciler,
)
from ragit.ingestion.extractors import Extractor, HTMLExtractor, PDFExtractor, TextExtractor
from ragit.ingestion.http_connector import Credential, HttpApiConnector, WebsiteConnector
from ragit.ingestion.object_connector import ObjectStorageClient, ObjectStorageConnector
from ragit.ingestion.sources import ConnectorDocumentSource, DocumentSourceConnector

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
