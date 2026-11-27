# Real-model benchmarks

A manual GitHub Actions workflow (`.github/workflows/real-model-benchmarks.yml`)
runs the EX-02 corpus (30 pages, 8 retrieval probes) against real services:

| Suite | What it measures | Needs |
|---|---|---|
| `docling` | Docling extraction: CER, region recall, probe Recall@5, latency per page | Docling and its models (installed in the job) |
| `embeddings` | Recall@5 and MRR with a real embedding model, plus call latency and cost | embedding model and key |
| `qdrant` | Same retrieval through a real Qdrant server: Recall@5, MRR, upsert time, query p50 and p95 | embedding model and key, Qdrant service container |
| `vlm` | A vision-language model reading each page as an extraction provider: CER, region recall, latency, cost per 1k pages | VLM model and key |

Results are uploaded as the `real-model-benchmark-results` artifact
(`results.json`, `summary.md`) and `summary.md` is added to the job summary.

## Safety

- Trigger is `workflow_dispatch` only. There is no `pull_request`,
  `pull_request_target`, `push` or `schedule` trigger, so forks and PRs cannot
  reach the secrets.
- The job runs only when dispatched from `main`. A branch that edits the
  workflow cannot read the secrets. Merge workflow changes, then dispatch.
- Token permission is `contents: read`. Actions are pinned to commit SHAs.
- Secrets appear only in the `env:` of steps that need them. Nothing echoes
  them. Errors from model calls are reduced to the exception class name.
- Dispatch inputs reach the shell only as environment variables and are
  validated before use.

## Cost caps

- `max_cost_usd` (default 2, hard maximum 10, enforced in the workflow and in
  the code). Before each call the run estimates its cost and refuses to send it
  if the total would pass the cap. Real cost comes from the provider response
  where the library knows the model's price, otherwise a conservative estimate.
- `max_vlm_pages` (default 40) limits pages sent to the VLM.
- The job has a 60 minute timeout. Runs are serialised.
- The cap is a guard in this code. Also set a spend limit with your provider.

## Secrets to add

Same names and convention as the owner's other repositories (an
OpenAI-compatible LiteLLM gateway). Settings, Secrets and variables, Actions,
Repository secrets:

- `LITELLM_API_KEY`: gateway virtual key
- `LITELLM_API_BASE`: gateway URL, for example `https://gateway.example/v1`

Models are not secrets. They are dispatch inputs with gateway-alias defaults:
`embedding_model` = `openai/rag-embedding` and `vlm_model` =
`openai/general-reasoning` (served by the llm-gateway aliases; the `openai/`
prefix selects the OpenAI-compatible protocol). The gateway has no dedicated
vision alias, so `general-reasoning` (GPT-4o upstream) is the default VLM.
Change the inputs for another gateway or provider.

A suite whose settings are missing is reported as skipped. It does not fail the run.

## Limits

The corpus is small and synthetic, and each run is a single pass with no
confidence intervals. Treat the numbers as a comparison between engines on
this corpus, not a production accuracy claim.
