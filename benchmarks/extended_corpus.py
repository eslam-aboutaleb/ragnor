"""Acquire human-transcribed Arabic scans/handwriting without committing data."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import zipfile
import tarfile
from pathlib import Path, PurePosixPath
from time import gmtime
from urllib.request import urlopen

from benchmarks.fetch_public_pdf_corpus import _download

HATHRD_URL = (
    "https://data.mendeley.com/public-files/datasets/xz6f8bw3w8/files/"
    "c6c1f430-3d92-4298-8f4b-b202d4c0e68a/file_downloaded"
)
HATHRD_HASH = "995baf562e9bceb34d90caec747f2189597f8b619078a777d35907254bad68cc"
MISRAJ_REVISION = "7177bf70f77ce259890d6af0bef53f18faf26ec9"


def markdown_text(markdown: str) -> str:
    from markdown_it import MarkdownIt  # noqa: PLC0415
    from benchmarks.fetch_public_pdf_corpus import reference_text  # noqa: PLC0415

    # Nontranscribed image placeholders are not reference text.
    markdown = re.sub(r"<(?:img|image)>.*?</(?:img|image)>", "", markdown, flags=re.S)
    rendered = MarkdownIt().render(markdown)
    return reference_text([{"content": {"html": rendered}}])


def _image_document(  # noqa: PLR0913 -- explicit content and provenance contract
    output, identity, image_bytes, text, *, family, provenance, license_name
):
    from PIL import Image  # noqa: PLC0415

    image = Image.open(io.BytesIO(image_bytes))
    if image.width * image.height > 25_000_000:
        raise ValueError("Image exceeds pixel budget")
    image = image.convert("RGB")
    path = output / f"{identity}.pdf"
    epoch = gmtime(946684800)
    image.save(path, format="PDF", resolution=200, creationDate=epoch, modDate=epoch)
    reference = output / f"{identity}.reference.txt"
    reference.write_text(text, encoding="utf-8")
    return {
        "document_id": identity,
        "path": path.name,
        "source_hash": hashlib.sha256(path.read_bytes()).hexdigest(),
        "annotation_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "family": family,
        "license": license_name,
        "language": "ara",
        "tags": ["arabic_rtl", family],
        "annotation_review": {"status": "local_review_pending", "upstream_evidence": provenance},
        "pages": [
            {
                "page": 1,
                "expected_text": text,
                "regions": [],
                "geometry_ground_truth": False,
                "transcription_available": True,
            }
        ],
        "provenance": provenance,
    }


def acquire_handwriting(output: Path) -> list[dict]:
    from lxml import etree  # noqa: PLC0415

    output.mkdir(parents=True, exist_ok=True)
    raw = _download(HATHRD_URL, output / "hathrd.zip", HATHRD_HASH, 20_000_000)
    documents = []
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        images = {
            str(PurePosixPath(name).with_suffix("")).casefold(): name
            for name in names
            if name.lower().endswith(".png")
        }
        for name in sorted(n for n in names if n.lower().endswith(".docx")):
            stem = str(PurePosixPath(name).with_suffix("")).casefold()
            if stem not in images:
                # Shared A/B transcripts have no proven page partition. Never guess it.
                continue
            if archive.getinfo(name).file_size > 2_000_000:
                raise ValueError("Oversized DOCX reference")
            with zipfile.ZipFile(io.BytesIO(archive.read(name))) as docx:
                xml = docx.read("word/document.xml")
            if len(xml) > 2_000_000:
                raise ValueError("Oversized XML reference")
            parser = etree.XMLParser(resolve_entities=False, no_network=True)
            tree = etree.fromstring(xml, parser)
            namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            paragraphs = [
                "".join(p.xpath(".//w:t/text()", namespaces=namespace))
                for p in tree.xpath("//w:p", namespaces=namespace)
            ]
            text = "\n".join(p for p in paragraphs if p.strip())
            identity = "hathrd-" + PurePosixPath(name).stem.lower()
            documents.append(
                _image_document(
                    output,
                    identity,
                    archive.read(images[stem]),
                    text,
                    family="arabic_handwriting",
                    license_name="CC-BY-4.0",
                    provenance={
                        "dataset": "Historical Arabic Handwritten Text Recognition Dataset",
                        "doi": "10.17632/xz6f8bw3w8.1",
                        "archive_sha256": HATHRD_HASH,
                        "source_image": PurePosixPath(images[stem]).name,
                        "source_annotation": PurePosixPath(name).name,
                        "human_annotation_evidence": "https://data.mendeley.com/datasets/xz6f8bw3w8/1",
                    },
                )
            )
    return documents


def acquire_misraj(output: Path, *, indices: list[int] | None = None) -> list[dict]:
    import pyarrow.parquet as pq  # noqa: PLC0415

    output.mkdir(parents=True, exist_ok=True)
    api = f"https://huggingface.co/api/datasets/Misraj/Misraj-DocOCR/tree/{MISRAJ_REVISION}/data"
    with urlopen(api, timeout=60) as response:  # noqa: S310 -- fixed HTTPS source
        index = json.loads(response.read(100_000))
    selected = set(indices if indices is not None else range(20))
    rows, ordinal = [], 0
    for item in sorted(index, key=lambda entry: entry["path"]):
        if not item["path"].endswith(".parquet"):
            continue
        name = Path(item["path"]).name
        data = _download(
            f"https://huggingface.co/datasets/Misraj/Misraj-DocOCR/resolve/"
            f"{MISRAJ_REVISION}/{item['path']}",
            output / name,
            item["lfs"]["oid"],
            600_000_000,
        )
        del data
        for batch in pq.ParquetFile(output / name).iter_batches(batch_size=8):
            for row in batch.to_pylist():
                if ordinal in selected:
                    image = row["image"]["bytes"]
                    text = markdown_text(row["markdown"])
                    identity = "misraj-" + row["uuid"]
                    provenance = {
                        "dataset": "Misraj/Misraj-DocOCR",
                        "revision": MISRAJ_REVISION,
                        "shard_sha256": item["lfs"]["oid"],
                        "row": ordinal,
                        "annotation_sha256": hashlib.sha256(row["markdown"].encode()).hexdigest(),
                        "human_annotation_evidence": "https://huggingface.co/datasets/Misraj/Misraj-DocOCR",
                    }
                    rows.append(
                        _image_document(
                            output,
                            identity,
                            image,
                            text,
                            family="arabic_mixed_origin",
                            provenance=provenance,
                            license_name="Apache-2.0",
                        )
                    )
                    (output / f"{identity}.reference.md").write_text(
                        row["markdown"], encoding="utf-8"
                    )
                ordinal += 1
    if len(rows) != len(selected):
        raise ValueError("Requested source rows are unavailable")
    return rows


NOD_FILES = {
    "yarmouk_01_col.tar.lzma": "de8d81ab6b856fe5fbf4ab3a360693e530626d537c72c8f44b7fc7a6b800613d",
    "ground_truth.tar.lzma": "6f6c24d47d711a930d121ca3bee43e7f671556a0ca456828a1a737bb793d361f",
}


def acquire_real_scans(output: Path, *, count=10) -> list[dict]:
    for name, digest in NOD_FILES.items():
        _download(
            f"https://zenodo.org/records/5068735/files/{name}?download=1",
            output / name,
            digest,
            60_000_000,
        )
    documents = []
    with (
        tarfile.open(output / "ground_truth.tar.lzma", "r:xz") as annotations,
        tarfile.open(output / "yarmouk_01_col.tar.lzma", "r:xz") as images,
    ):
        names = [
            member
            for member in images.getmembers()
            if member.isfile() and member.name.endswith(".tiff")
        ]
        # Select single-page articles only; never align a partial article to its full transcript.
        grouped = {}
        for member in names:
            identity = PurePosixPath(member.name).stem.split("_")[0]
            grouped.setdefault(identity, []).append(member)
        selected = [
            (identity, group[0]) for identity, group in sorted(grouped.items()) if len(group) == 1
        ][:count]
        for identity, member in selected:
            transcript = annotations.extractfile(f"ground_truth/yarmouk_gt/{identity}.txt").read(
                2_000_001
            )
            if len(transcript) > 2_000_000 or member.size > 100_000_000:
                raise ValueError("Real scan exceeds import limits")
            documents.append(
                _image_document(
                    output,
                    "nod-" + identity,
                    images.extractfile(member).read(),
                    transcript.decode("utf-8-sig"),
                    family="arabic_real_scan",
                    license_name="CC-BY-4.0",
                    provenance={
                        "dataset": "Noisy OCR Dataset, clean Yarmouk colour scans",
                        "doi": "10.5281/zenodo.5068735",
                        "archives_sha256": NOD_FILES,
                        "source_image": PurePosixPath(member.name).name,
                        "human_annotation_evidence": "https://zenodo.org/records/5068735",
                    },
                )
            )
    return documents


def acquire_tessdata(output: Path) -> dict:
    digests = {
        "ara": "e3206d3dc87fd50c24a0fb9f01838615911d25168f4e64415244b67d2bb3e729",
        "eng": "7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2",
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, digest in digests.items():
        _download(
            "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/4.1.0/"
            + f"{name}.traineddata",
            output / f"{name}.traineddata",
            digest,
            10_000_000,
        )
    return digests


def combine_manifests(base_path: Path, extra_path: Path, output: Path, *, common=True) -> dict:
    """Build a fixed comparison cohort without guessing annotation geometry or semantics."""
    import shutil  # noqa: PLC0415
    from benchmarks.run_quality import validate_manifest  # noqa: PLC0415
    from benchmarks.quality_metrics import TableParser  # noqa: PLC0415
    from benchmarks.semantic_metrics import numeric_value  # noqa: PLC0415

    output.mkdir(parents=True, exist_ok=True)
    documents = []
    for path in [base_path, extra_path]:
        manifest = validate_manifest(path)
        for doc in manifest["documents"]:
            if (
                path == base_path
                and common
                and doc["family"] == "public"
                and doc["document_id"] not in manifest["stress_sources"]
            ):
                continue
            destination = output / Path(doc["path"]).name
            if destination.exists():
                if hashlib.sha256(destination.read_bytes()).hexdigest() != doc["source_hash"]:
                    raise ValueError("Corpus filename collision")
            else:
                shutil.copyfile(path.parent / doc["path"], destination)
            doc["path"] = destination.name
            doc["annotation_sha256"] = hashlib.sha256(
                "\n".join(page["expected_text"] for page in doc["pages"]).encode()
            ).hexdigest()
            for page in doc["pages"]:
                page.setdefault("reading_order", [region["id"] for region in page["regions"]])
                for region in page["regions"]:
                    if region["category"] != "Table":
                        continue
                    grid = TableParser(region["html"]).grid()
                    region["cells"] = [
                        {
                            "row": row,
                            "col": col,
                            "text": cell.text,
                            "rowspan": cell.rowspan,
                            "colspan": cell.colspan,
                        }
                        for (row, col), cell in sorted(grid.items())
                    ]
                    for (row, col), cell in grid.items():
                        try:
                            numeric_value(cell.text)
                        except ValueError:
                            continue
                        page.setdefault("critical_fields", []).append(
                            {
                                "id": f"{region['id']}:{row}:{col}",
                                "kind": "number",
                                "value": cell.text,
                                "region_id": region["id"],
                                "table_region_id": region["id"],
                                "row": row,
                                "col": col,
                                "provenance": (
                                    "deterministic projection of upstream table cell; "
                                    "pending independent review"
                                ),
                            }
                        )
            documents.append(doc)
    result = {
        "schema_version": 3,
        "dataset": "fixed public structural and Arabic comparison",
        "coordinate_convention": "normalized top-left MediaBox",
        "documents": documents,
        "stress_sources": [doc["document_id"] for doc in documents],
    }
    (output / "quality-manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--misraj-indices", default=",".join(str(i) for i in range(20)))
    parser.add_argument("--base-manifest", type=Path)
    parser.add_argument("--combined-output", type=Path)
    parser.add_argument("--base-all", action="store_true")
    args = parser.parse_args()
    if bool(args.base_manifest) != bool(args.combined_output):
        parser.error("--base-manifest and --combined-output must be supplied together")
    indices = [int(value) for value in args.misraj_indices.split(",")]
    documents = (
        acquire_handwriting(args.output)
        + acquire_misraj(args.output, indices=indices)
        + acquire_real_scans(args.output)
    )
    acquire_tessdata(args.output / "tessdata")
    manifest = {
        "schema_version": 3,
        "dataset": "Arabic source-transcribed qualification",
        "coordinate_convention": "normalized top-left MediaBox",
        "documents": documents,
        "stress_sources": [d["document_id"] for d in documents],
    }
    (args.output / "quality-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if args.base_manifest:
        combine_manifests(
            args.base_manifest,
            args.output / "quality-manifest.json",
            args.combined_output,
            common=not args.base_all,
        )
    print(json.dumps({"documents": len(documents), "upstream_transcription_provenance": True}))


if __name__ == "__main__":
    main()
