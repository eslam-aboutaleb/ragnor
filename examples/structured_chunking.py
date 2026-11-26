"""Chunk a basic Markdown document and a structured document with source spans."""

from __future__ import annotations

from ragnor.chunking import (
    StructuredDocument,
    StructuredDocumentChunker,
    StructuredRegion,
    TableRow,
)
from ragnor.evidence.tokenizer import ExactTokenizerAdapter

tokenizer = ExactTokenizerAdapter(lambda text: len(text.split()), name="demo-words")
chunker = StructuredDocumentChunker(tokenizer, max_tokens=40, strict=True)

markdown = "# Report\n\n## Prices\n\n| Item | Qty |\n| --- | --- |\n" + "\n".join(
    f"| part-{i} | {i} |" for i in range(30)
)
for chunk in chunker.chunk(markdown):
    print(chunk.id, chunk.metadata["continuation"]["part"], chunk.metadata["table"]["row_start"])

document = StructuredDocument(
    document_id="invoice-1",
    fingerprint="evidence-sha256-goes-here",
    regions=(
        StructuredRegion("h", "heading", "Totals", level=1, page=3),
        StructuredRegion(
            "t",
            "table",
            page=3,
            bbox=(0.1, 0.2, 0.9, 0.5),
            header=("Item", "Amount"),
            rows=(TableRow(("Hosting", "10.00"), page=3), TableRow(("Support", "5.00"), page=4)),
        ),
    ),
    access={"tenant_id": "acme"},
)
for chunk in chunker.chunk_document(document):
    print([s["page"] for s in chunk.metadata["source_spans"]], chunk.metadata["access"])
