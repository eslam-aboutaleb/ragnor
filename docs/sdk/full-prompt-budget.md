# Full provider prompt context budget

Use separate token caps for individual chunk inputs, rendered evidence,
and the entire provider request. `EvidenceAssembler` controls the
rendered citation/evidence portion only; it cannot know the chat wrapper,
tool-message overhead, images, system messages or target model's reserved
completion tokens.

`ragnor.evidence.prompt_budget.PromptBudgetPlanner` is an opt-in,
zero-dependency port for a host-supplied `ProviderPromptCounter`.

```python
from ragnor.evidence.prompt_budget import PromptBudgetPlanner

# Your provider adapter must implement name, model_id, is_exact=True,
# count_prompt_tokens(messages), and attest to the ACTUAL deployed
# provider/model revision and serialized messages (including tools).
counter = my_provider_prompt_counter
planner = PromptBudgetPlanner(
    counter=counter,
    model_id=counter.model_id,
    total_context_tokens=128_000,
    reserved_completion_tokens=3_000,
)
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": question},
    {"role": "tool", "content": evidence_bundle.format_context()},
]
result = planner.assess(messages)  # raise before sending if oversized
```

The SDK does **not** declare that tiktoken text counts precisely account
for all proprietary chat APIs or vision/tool overhead. Host implementations
must use exact provider token-count endpoints, a verified frame serializer,
or other model-specific audited token-counter logic. A mismatched model,
approximate counter, invalid token value or overflow fails closed. The result
contains only safe numeric diagnostic fields, no prompt data.

For large content, use `EvidenceAssembler` to shrink evidence first; then
validate the **final** message set. Count the complete provider request again
if system instructions, tools, retrieved data or generation reserve changes.
