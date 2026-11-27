"""Region detection and foreground-ink coverage, separate from transcription."""

from __future__ import annotations

from benchmarks.quality_metrics import match_boxes

TEXT_CLASSES = {
    "Paragraph",
    "Heading1",
    "Header",
    "Footer",
    "Caption",
    "List",
    "Index",
    "Footnote",
    "text",
}


def region_detection(reference: list[dict], actual: list[dict] | None) -> dict:
    expected = [region for region in reference if region["category"] in TEXT_CLASSES]
    if actual is None:
        return {
            "region_detection_supported": False,
            "region_precision": None,
            "region_recall": None,
            "reference_region_count": len(expected),
            "predicted_region_count": None,
        }
    matched = match_boxes(expected, actual, threshold=0.5)
    observed = {i for i, _ in matched}
    return {
        "region_detection_supported": True,
        "reference_region_count": len(expected),
        "predicted_region_count": len(actual),
        "region_precision": len(matched) / len(actual) if actual else (0.0 if expected else None),
        "region_recall": len(matched) / len(expected) if expected else None,
        "omitted_region_ids": [
            region["id"] for i, region in enumerate(expected) if i not in observed
        ],
        "region_detection_iou": 0.5,
    }


def ink_coverage(image, words: list[dict]) -> dict:
    from PIL import Image, ImageChops, ImageDraw  # noqa: PLC0415

    gray = image.convert("L")
    histogram = gray.histogram()
    total = sum(histogram)
    nonzero = [i for i, count in enumerate(histogram) if count]
    if len(nonzero) < 2:
        return {"word_box_ink_coverage": None, "foreground_pixels": 0, "ink_threshold": None}
    weighted = sum(i * count for i, count in enumerate(histogram))
    weight, partial, best, threshold = 0, 0, -1.0, 0
    for i, count in enumerate(histogram):
        weight += count
        partial += i * count
        if weight in {0, total}:
            continue
        between = (
            weight
            * (total - weight)
            * (partial / weight - (weighted - partial) / (total - weight)) ** 2
        )
        if between > best:
            best, threshold = between, i
    threshold = min(threshold, 245)
    ink = gray.point(lambda value: 255 if value <= threshold else 0)
    foreground = sum(ink.histogram()[1:])
    boxes = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(boxes)
    for word in words:
        x0, y0, x1, y1 = word["bbox"]
        draw.rectangle(
            [
                int(x0 * image.width),
                int(y0 * image.height),
                max(int(x0 * image.width), int(x1 * image.width) - 1),
                max(int(y0 * image.height), int(y1 * image.height) - 1),
            ],
            fill=255,
        )
    covered = sum(ImageChops.multiply(ink, boxes).histogram()[1:])
    return {
        "word_box_ink_coverage": covered / foreground if foreground else None,
        "foreground_pixels": foreground,
        "ink_threshold": threshold,
    }
