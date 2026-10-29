"""Generate tiny regression PDFs locally; no binary fixtures are checked in."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path


def create_smoke_corpus(output: Path) -> Path:
    from pypdf import PdfWriter  # noqa: PLC0415
    from pypdf.generic import (  # noqa: PLC0415
        DecodedStreamObject,
        DictionaryObject,
        NameObject,
        NumberObject,
    )

    cases = {
        "digital": [("First page", "text"), ("Third page", "text")],
        "blank": [("", "blank")],
        "image": [("First", "text"), ("", "image"), ("Third", "text")],
        "vector": [("First", "text"), ("", "vector"), ("Third", "text")],
        "overlay": [("First", "text"), ("overlay", "image"), ("Third", "text")],
    }
    output.mkdir(parents=True, exist_ok=True)
    documents = []
    for name, definitions in cases.items():
        writer = PdfWriter()
        references = []
        for number, (text, kind) in enumerate(definitions, 1):
            page = writer.add_blank_page(width=300, height=300)
            resources = DictionaryObject()
            operations = b""
            if text:
                resources[NameObject("/Font")] = DictionaryObject(
                    {
                        NameObject("/F1"): DictionaryObject(
                            {
                                NameObject("/Type"): NameObject("/Font"),
                                NameObject("/Subtype"): NameObject("/Type1"),
                                NameObject("/BaseFont"): NameObject("/Helvetica"),
                            }
                        )
                    }
                )
                operations += f"BT /F1 12 Tf 20 200 Td ({text}) Tj ET\n".encode()
            if kind == "image":
                image = DecodedStreamObject()
                image.set_data(b"\x00")
                image.update(
                    {
                        NameObject("/Type"): NameObject("/XObject"),
                        NameObject("/Subtype"): NameObject("/Image"),
                        NameObject("/Width"): NumberObject(1),
                        NameObject("/Height"): NumberObject(1),
                        NameObject("/ColorSpace"): NameObject("/DeviceGray"),
                        NameObject("/BitsPerComponent"): NumberObject(8),
                    }
                )
                resources[NameObject("/XObject")] = DictionaryObject(
                    {NameObject("/Im0"): writer._add_object(image)}
                )
                operations += b"q 100 0 0 100 0 0 cm /Im0 Do Q"
            elif kind == "vector":
                operations += b"0 0 100 100 re f"
            if operations:
                stream = DecodedStreamObject()
                stream.set_data(operations)
                page[NameObject("/Contents")] = writer._add_object(stream)
            page[NameObject("/Resources")] = resources
            status = (
                "partial"
                if text and kind == "image"
                else "extracted"
                if text
                else ("verified_blank" if kind == "blank" else "needs_ocr")
            )
            references.append({"page": number, "expected_status": status, "expected_text": text})
        data = io.BytesIO()
        writer.write(data)
        path = output / f"{name}.pdf"
        path.write_bytes(data.getvalue())
        documents.append(
            {
                "document_id": name,
                "family": name,
                "path": path.name,
                "license": "Elastic-2.0",
                "source_hash": hashlib.sha256(data.getvalue()).hexdigest(),
                "pages": references,
            }
        )
    path = output / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "annotation_review": "synthetic-only; pending human review",
                "documents": documents,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path
