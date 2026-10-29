"""Acquire the full pinned public corpus and deterministic stress variants locally."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

from benchmarks.fetch_public_pdf_corpus import _download, reference_text
from benchmarks.quality_metrics import TableParser, validate_bbox

REVISION = "24702c61a2fb13325534be664653bc6e60250d13"
REFERENCE_SHA256 = "116a42c284890772048c5d21486882f2632da4947c3dd90434efef4f49b7d9fd"
BASE = f"https://huggingface.co/datasets/upstage/dp-bench/resolve/{REVISION}"


def acquire(output: Path) -> dict:
    from pypdf import PdfReader  # noqa: PLC0415

    output.mkdir(parents=True, exist_ok=True)
    raw = _download(
        f"{BASE}/dataset/reference.json", output / "reference.json", REFERENCE_SHA256, 5_000_000
    )
    reference = json.loads(raw)
    if len(reference) != 200:
        raise ValueError("Pinned source page count changed")
    index_url = (
        "https://huggingface.co/api/datasets/upstage/dp-bench/tree/"
        f"{REVISION}/dataset/pdfs?limit=1000"
    )
    with urlopen(index_url, timeout=60) as response:  # noqa: S310 -- fixed HTTPS endpoint
        index = json.loads(response.read(2_000_000))
    hashes = {Path(item["path"]).name: item["lfs"]["oid"] for item in index}
    if set(hashes) != set(reference):
        raise ValueError("Source index and reference documents disagree")

    def fetch(name):
        path = output / name
        # Reuse a previous local subset without duplicating network requests.
        existing = output.parent / "dpbench-40" / name
        if not path.exists() and existing.exists():
            shutil.copyfile(existing, path)
        data = _download(f"{BASE}/dataset/pdfs/{name}", path, hashes[name], 20_000_000)
        reader = PdfReader(io.BytesIO(data))
        if len(reader.pages) != 1:
            raise ValueError("Public reference expects one source page")
        page = reader.pages[0]
        width, height = float(page.mediabox.width), float(page.mediabox.height)
        regions = []
        for element in reference[name]["elements"]:
            polygon = element["coordinates"]
            # Source annotation pixels correspond to MediaBox rendering at 200 DPI.
            bbox = [
                min(p["x"] for p in polygon) / (width * 200 / 72),
                min(p["y"] for p in polygon) / (height * 200 / 72),
                max(p["x"] for p in polygon) / (width * 200 / 72),
                max(p["y"] for p in polygon) / (height * 200 / 72),
            ]
            # Small annotation edge rounding is tolerated; material overflow is an error.
            if any(x < -0.002 or x > 1.002 for x in bbox):
                raise ValueError("Annotation coordinates do not match the render transform")
            bbox = [max(0.0, min(1.0, x)) for x in bbox]
            validate_bbox(bbox)
            region = {
                "id": str(element["id"]),
                "category": element["category"],
                "bbox": bbox,
                "text": reference_text([element]),
            }
            if element["category"] == "Table":
                region["html"] = element["content"]["html"]
            regions.append(region)
        tags = sorted({region["category"] for region in regions})
        return {
            "document_id": Path(name).stem,
            "path": name,
            "source_hash": hashlib.sha256(data).hexdigest(),
            "family": "public",
            "tags": tags,
            "pages": [
                {
                    "page": 1,
                    "expected_text": reference_text(reference[name]["elements"]),
                    "regions": regions,
                }
            ],
            "license": "MIT declared by upstream dataset card",
            "source": f"{BASE}/dataset/pdfs/{name}",
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        documents = list(pool.map(fetch, sorted(reference)))
    return {
        "schema_version": 2,
        "dataset": "upstage/dp-bench",
        "revision": REVISION,
        "reference_sha256": REFERENCE_SHA256,
        "coordinate_convention": "normalized top-left MediaBox",
        "annotation_review": (
            "upstream references; four overlay spot checks; no independent full-corpus signoff"
        ),
        "documents": documents,
    }


def _complexity(document):
    counts = []
    for region in document["pages"][0]["regions"]:
        if region["category"] == "Table":
            try:
                table = TableParser(region["html"])
                counts.append(
                    sum(len(row) for row in table.rows)
                    + 10
                    * sum(
                        cell.rowspan > 1 or cell.colspan > 1 for row in table.rows for cell in row
                    )
                )
            except ValueError:
                counts.append(0)
    return max(counts, default=0)


def add_stress(manifest: dict, output: Path, count=10) -> None:
    from pypdf import PdfReader, PdfWriter  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    import pypdfium2 as pdfium  # noqa: PLC0415

    selected = sorted(
        manifest["documents"], key=lambda doc: (-_complexity(doc), doc["document_id"])
    )[:count]
    for original in selected:
        source = output / original["path"]
        for family in ["scan_200dpi", "scan_100dpi_jpeg35", "rotated_90", "mixed_two_page"]:
            path = output / f"{original['document_id']}-{family}.pdf"
            writer = PdfWriter()
            if family == "rotated_90":
                page = PdfReader(source).pages[0]
                page.rotate(90)
                writer.add_page(page)
                pages = json.loads(json.dumps(original["pages"]))
                for region in pages[0]["regions"]:
                    x0, y0, x1, y1 = region["bbox"]
                    region["bbox"] = [1 - y1, x0, 1 - y0, x1]
            else:
                dpi = 100 if family == "scan_100dpi_jpeg35" else 200
                with pdfium.PdfDocument(source) as pdf:
                    page = pdf[0]
                    bitmap = page.render(scale=dpi / 72)
                    image = bitmap.to_pil().convert("RGB").copy()
                    bitmap.close()
                    page.close()
                if dpi == 100:
                    jpeg = io.BytesIO()
                    image.convert("L").save(jpeg, format="JPEG", quality=35)
                    image = Image.open(io.BytesIO(jpeg.getvalue())).convert("RGB")
                encoded = io.BytesIO()
                image.save(encoded, format="PDF", resolution=dpi)
                scan = PdfReader(io.BytesIO(encoded.getvalue())).pages[0]
                pages = json.loads(json.dumps(original["pages"]))
                if family == "mixed_two_page":
                    writer.add_page(PdfReader(source).pages[0])
                    writer.add_page(scan)
                    pages.append(json.loads(json.dumps(pages[0])))
                    pages[1]["page"] = 2
                else:
                    writer.add_page(scan)
            with path.open("wb") as handle:
                writer.write(handle)
            manifest["documents"].append(
                {
                    **original,
                    "document_id": path.stem,
                    "path": path.name,
                    "source_hash": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "family": family,
                    "source_document_id": original["document_id"],
                    "pages": pages,
                }
            )
    manifest["stress_sources"] = [doc["document_id"] for doc in selected]
    manifest["coverage"] = dict(Counter(doc["family"] for doc in manifest["documents"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = acquire(args.output)
    add_stress(manifest, args.output)
    path = args.output / "quality-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"documents": len(manifest["documents"]), "families": manifest["coverage"]}))


if __name__ == "__main__":
    main()
