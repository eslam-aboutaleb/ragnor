# Token-aware chunking and snapshot invalidation

Word-based `SlidingWindowChunker` remains the default. For model-specific
chunk size enforcement, explicitly inject a verified tokenizer:

```python
from ragnor.chunking import TokenAwareChunker
from ragnor.evidence.tokenizer import ExactTokenizerAdapter

# `encoder` belongs to the host and must match the exact embedding model.
counter = ExactTokenizerAdapter(encoder, name="embedding-model-encoding-revision")
chunker = TokenAwareChunker(counter, max_tokens=512, overlap_tokens=64)
rag = await RagnorBuilder.local(my_embedder).with_chunker(chunker).build()
```

The new strategy does not import a vendor SDK or change existing word-based
outputs. It constructs original-source character spans and checks the injected
tokenizer at each candidate step. This is deliberately exact but slower than
provider-native segmentation for large documents; hosts may inject optimized
`Chunker` implementations under the same port.

**Strong guarantees are conditional:** `ExactTokenizerAdapter` trusts the
host's claim that the supplied counter matches the actual provider model.
A fake or mismatched tokenizer is not a valid basis for production limits.
For local approximate budgets only, pass `exact_required=False`, knowing it
does **not** guarantee any model context limit. This chunker doesn't assert the
full prompt budget (instructions, citations and model output all consume tokens).

The ingestion snapshot includes an SHA-256 fingerprint of the sanitized,
canonical chunker version and configuration. This makes tokenizer-name,
overlap, and token-limit changes trigger re-ingestion. Older persisted
snapshots without this field invalidate safely rather than skipping a needed
reindex. Avoid placing secrets in custom chunker `config`.

The implementation preserves original Unicode text spans and offset metadata;
it does not yet offer sentence/heading-aware segmentation, a tiktoken optional
provider, streaming performance optimizations, or evidence-budget accounting
for complete rendered prompts. Those remain acceptance work under #27.
