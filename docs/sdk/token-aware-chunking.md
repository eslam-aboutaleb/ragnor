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

The strategy does not import a vendor SDK or change existing word-based
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

The implementation preserves original Unicode text spans and offset metadata.
It sizes character spans rather than sentence or heading boundaries. For the
complete model request, use [full prompt budgeting](full-prompt-budget.md).


## Optional tiktoken integration and strict evidence mode

```python
from ragnor.evidence.tiktoken_adapter import TiktokenTokenizer
from ragnor.evidence.assembler import EvidenceAssembler

tokenizer = TiktokenTokenizer(encoding_name="cl100k_base")
assembler = EvidenceAssembler(tokenizer=tokenizer, exact_required=True)
```

Install `ragnor[tokenizers-tiktoken]` only in applications requiring that
encoding. Unknown encodings raise instead of silently selecting a fallback.
Constructing `EvidenceAssembler(exact_required=True)` with the approximate
default raises a configuration error. Approximate-mode evidence assembly
emits one warning per assembler instance, not a claim of hard model limits.

The assembled evidence uses an exact counter for the **rendered evidence
context**, including citation framing; the host must separately count user,
system, tools and answer-budget tokens for the full model request.

## Attesting a tokenizer against the deployed provider

`ExactTokenizerAdapter(..., is_exact=True)` trusts the caller. To check that
claim against the provider's own counts, supply a `ProviderPromptCounter`
backed by the real provider (its token-count endpoint, or the prompt token
usage of a real request) and run:

```python
from ragnor.evidence import attest_tokenizer, attest_prompt_counter

attestation = attest_tokenizer(tokenizer, provider_counter, model_id="my-model-rev")
attestation.require()          # raises TokenizerAttestationError on any mismatch
attestation.fingerprint()      # stable hash of what was attested
```

`attest_tokenizer` compares the tokenizer with the provider's count of each
probe (multilingual text, code, JSON, emoji and long input by default) after
calibrating constant message framing. `attest_prompt_counter` compares a full
serialized-prompt counter on the message shapes you pass. Results hold no probe
text or credentials. A pass covers the probe set at attestation time only;
re-run it when the model, revision or tokenizer changes. Ragnor makes no
network call: the strength of the result is the strength of the oracle you
supply.

## Persisting snapshots

`snapshot_to_dict` / `snapshot_from_dict` serialize `IngestionSnapshot` for a
host-owned version table. Payloads from older releases load with empty values
for newer optional fields and therefore never equal a current snapshot, so the
source is re-ingested. A payload from a newer format, or a malformed one, raises
`SnapshotFormatError`; a version store should treat that as "no current
version" rather than skipping ingestion.
