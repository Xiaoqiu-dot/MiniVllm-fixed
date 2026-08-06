# Summary — Commit ID Range: `798b455`

**Tag:** `release-2`

This release prevents requests from exceeding KV Cache or context capacity and fixes the Decode boundary that decides when a new KV block is required.

## Major Fixes

### 1. Engine and Request Context Limits

**Files**

* `src/myvllm/engine/llm_engine.py`
* `src/myvllm/engine/scheduler.py`
* `tests/test_scheduler.py`

**Bugs**

* The engine accepted a `max_model_length` larger than its physical KV Cache capacity. (Original scheduler initialization)
* A request could set a context limit above the engine limit. (Original request admission)
* A prompt that already filled the entire context could enter the queue despite having no room for a completion token. (Original request admission)
* Oversized requests could remain unschedulable instead of failing early with a useful error. (Original Prefill scheduling)

**Example**

If the engine has 8 KV Blocks with `block_size = 16`, it can store at most 128 Tokens. Configuring `max_model_length = 256` lets a request pass API validation even though no execution can store its full context. Similarly, with a valid 128-Token engine limit, accepting a 128-Token Prompt leaves no slot for the first generated Token. These requests would eventually stall or repeatedly preempt under memory pressure instead of failing at admission.

**Fixes**

* Pass the engine's `max_model_length` into the scheduler. (Engine wiring)
* At engine construction, convert Block capacity to Tokens (`max_cached_blocks * block_size`) and reject an engine context limit that cannot physically fit. (Capacity validation)
* At request admission, resolve the effective limit once: inherit the engine value when unset, accept a smaller positive request limit, and reject a non-positive or larger value before the request enters `waiting`. (Request validation)
* Enforce `num_prompt_tokens < request_max_length`, store the resolved limit on the Sequence, and check total Tokens after every sampled Token so generation finishes and releases its Blocks exactly at the boundary. (Prompt validation, stop condition)

### 2. Decode Block-Boundary Allocation

**Bug**

`postprocess()` appends the newly sampled token before `can_append()` runs. The old check looked for `num_tokens % block_size == 0`, so it tested the end of a block rather than the first token of the next block. This could allow Decode to proceed without allocating required KV Cache space. (Original check)

**Example**

With `block_size = 4`, a Sequence currently has four Tokens in Block 0. `postprocess()` appends Token 5, making `num_tokens = 5`; that Token belongs to a new Block 1. The old condition checks `5 % 4 == 0`, gets false, and claims no new Block is needed. The following Decode then builds a Slot Mapping for Block 1 even though it was never allocated. The correct boundary is `5 % 4 == 1` because the Token has already been appended.

**Fix**

Make `can_append()` follow the same post-append state convention as `append()`: require a free Block exactly when `num_tokens % block_size == 1`; otherwise the Token remains inside the already allocated final Block. This aligns admission with the later Block Table update. (Boundary fix)

## Validation

Scheduler tests cover inherited and request-level context limits, invalid limits, prompts with no generation room, KV Cache capacity overflow, and stopping at the exact context boundary. (Tests)
