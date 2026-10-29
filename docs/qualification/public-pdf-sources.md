# Public PDF benchmark source decisions

The benchmark acquires all 200 original pages from
[Upstage DP-Bench](https://huggingface.co/datasets/upstage/dp-bench), pinned to
revision `24702c61a2fb13325534be664653bc6e60250d13`. Its reference annotations contain
text, polygons, ordered elements and table HTML. The dataset card declares MIT.
All source PDFs are verified against the pinned upstream LFS hashes; the reference
JSON SHA-256 is checked independently. No source documents or annotations are
included in this repository's commits.

[DocLayNet](https://github.com/DS4SD/DocLayNet) remains a candidate for more human
layout annotations; layout labels and PDF text cells do not establish independently
validated transcription or table-cell accuracy.
[KITAB-Bench](https://github.com/mbzuai-oryx/KITAB-Bench) remains a candidate for
Arabic/RTL and handwriting after component-specific data terms are verified.
[OmniDocBench](https://github.com/opendatalab/OmniDocBench) states noncommercial
research restrictions for its dataset, so it is not acquired or vendored as the
default corpus.

The earlier 40-page selection is superseded by the full corpus, structural metrics,
multiple providers and controlled stress variants. The independent full-annotation
review gate is still open. See [reproduction and metric definitions](../extraction-benchmarks.md)
and [actual aggregate results](pdf-quality-results.md).
