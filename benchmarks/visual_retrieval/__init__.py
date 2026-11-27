"""ViDoRe-style offline benchmark for the opt-in visual page fallback (issue #68 / EX-10).

Pages are generated raster drawings with no extractable text. The fake embedder
(`fake_model`) is deterministic and weight-free, so this benchmark verifies the
retrieval plumbing, fusion, metrics and cost reporting. It does NOT measure the
quality of any real ColPali/ColQwen model; run a host provider through
``run_benchmark(provider=...)`` for that.
"""
