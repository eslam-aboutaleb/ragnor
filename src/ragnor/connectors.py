"""Compatibility import for :mod:`ragnor.ingestion.connectors`."""

from __future__ import annotations

# ruff: noqa: PLC0414 - explicit aliases declare typed compatibility exports

import sys
from typing import TYPE_CHECKING

from ragnor.ingestion import connectors as _implementation

if TYPE_CHECKING:
    from ragnor.ingestion.connectors import (
        ApiConnector as ApiConnector,
        ChangeEvent as ChangeEvent,
        ChangeEventSource as ChangeEventSource,
        ConnectorDocumentSource as ConnectorDocumentSource,
        Credential as Credential,
        DirectoryConnector as DirectoryConnector,
        DirectoryDocumentSource as DirectoryDocumentSource,
        DirectorySource as DirectorySource,
        DirectorySourceConnector as DirectorySourceConnector,
        DiscoveryPage as DiscoveryPage,
        DocumentSourceConnector as DocumentSourceConnector,
        Extractor as Extractor,
        FileSystemConnector as FileSystemConnector,
        FilesystemSourceConnector as FilesystemSourceConnector,
        FilesystemWatcher as FilesystemWatcher,
        GenericHttpConnector as GenericHttpConnector,
        HTMLExtractor as HTMLExtractor,
        HttpApiConnector as HttpApiConnector,
        HttpConnector as HttpConnector,
        IncompleteScanError as IncompleteScanError,
        IncrementalSourceConnector as IncrementalSourceConnector,
        LoadResult as LoadResult,
        LocalDirectoryConnector as LocalDirectoryConnector,
        ObjectStorageClient as ObjectStorageClient,
        ObjectStorageConnector as ObjectStorageConnector,
        ObjectStorageSource as ObjectStorageSource,
        ObjectStoreConnector as ObjectStoreConnector,
        PDFExtractor as PDFExtractor,
        PeriodicReconciler as PeriodicReconciler,
        S3Connector as S3Connector,
        SkipRecord as SkipRecord,
        SourceConnector as SourceConnector,
        SourceError as SourceError,
        SourceInaccessibleError as SourceInaccessibleError,
        SourceItem as SourceItem,
        SourceNotFoundError as SourceNotFoundError,
        SourceStatus as SourceStatus,
        TextExtractor as TextExtractor,
        TransientSourceError as TransientSourceError,
        WebsiteConnector as WebsiteConnector,
        collect_pages as collect_pages,
    )

# Keep module identity, private helpers and patch targets compatible too.
sys.modules[__name__] = _implementation
