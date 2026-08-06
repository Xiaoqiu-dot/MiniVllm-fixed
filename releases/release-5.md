# Summary — Commit ID Range: `ee950af`

**Tag:** `release-5`

This release hardens the complete inference path: multi-GPU worker coordination, Prefix Cache bookkeeping, scheduling fairness, numerical stability, CUDA Graph Decode, and public engine boundary behavior.

## Major Fixes

### 1. Multi-GPU Worker Coordination

**Files**

* `src/myvllm/engine/sequence.py`
* `src/myvllm/engine/model_runner.py`
* `src/myvllm/engine/llm_engine.py`

**Bugs**

* Worker deserialization omitted `Sequence.block_size`, causing methods such as `num_blocks` to fail outside Rank 0. (Original serialization)
* Every engine used the same distributed rendezvous port and shared-memory name, so concurrent engines could collide. (Original worker setup)
* Shared-memory payloads were written without checking their serialized size. (Original shared-memory write)
* Worker initialization and cleanup order could surface failures as NCCL hangs or attempt cleanup more than once. (Original worker lifecycle, original engine cleanup)

**Example**

Two engines started on the same host both tried to bind the same rendezvous port and create the same shared-memory segment. The second engine could fail during setup or attach to resources owned by the first. Separately, Rank 0 serialized a Sequence without `block_size`; after a worker restored it and evaluated `num_blocks`, it raised `AttributeError: 'Sequence' object has no attribute 'block_size'`. Rank 0 could then wait in an NCCL collective, making the real Python exception look like a distributed hang.

**Fixes**

* Serialize and restore `block_size` together with the rest of the Sequence state. (Sequence state)
* Validate `world_size` before spawning processes, ask the OS for an available TCP port, and generate a UUID-based shared-memory name. Pass both values through the worker configuration so every Rank in one engine agrees while independent engines remain isolated. (Engine initialization)
* Rank 0 creates and owns the shared-memory segment; other Ranks attach by name. Serialize each command first, compare its byte length with the configured segment capacity, and fail before writing a truncated payload. (Shared memory)
* Make worker and engine shutdown idempotent and join worker processes during cleanup. (Worker cleanup, engine cleanup)

### 2. Prefix Cache Correctness Under Reuse

**Files**

* `src/myvllm/engine/block_manager.py`
* `src/myvllm/models/qwen3.py`

**Bugs**

* A single Prefix Hash mapped to only one physical Block. Duplicate valid copies could overwrite each other, and releasing one could remove the mapping for another. (Original scalar Hash map, original replacement logic)
* After a Prefix Cache hit, Qwen3 RoPE positions restarted at zero for the uncached suffix rather than continuing after the cached context. (Original positions)

**Example**

Assume physical Blocks 3 and 9 contain the same Token prefix and therefore have the same Hash. A scalar `hash -> block_id` map can remember only Block 9. If Block 9 is recycled, removing that entry also makes still-valid Block 3 unreachable. For RoPE, if 96 Tokens are cached and the suffix contains 32 Tokens, their positions must be 96–127; restarting them at 0–31 rotates Q/K as if the prefix did not exist.

**Fixes**

* Replace the scalar mapping with `hash -> set[block_id]`. Registering adds one physical copy; recycling removes only that Block ID and deletes the Hash key only when its set becomes empty. (Hash index)
* For each logical Prefix Block, iterate all physical candidates with the expected Hash, verify Token IDs to protect against Hash collisions, and choose a valid Block without invalidating its duplicates. (Prefix selection)
* For every Sequence in a variable-length batch, compute `cached_len = full_kv_len - query_len`; repeat that offset for each new Query and add local suffix positions. This produces absolute RoPE positions while retaining the flattened batched representation. (Position calculation, usage)

### 3. Prefill Budget and Scheduling Fairness

**File**

* `src/myvllm/engine/scheduler.py`

**Bugs**

* The Prefill token budget counted cached tokens that would not be recomputed. (Original budget accounting)
* A single uncached Prompt larger than the batch budget could remain blocked forever because chunked Prefill is not implemented. (Original blocked Prefill path)
* Continuous Prefill traffic could starve requests already in Decode. (Original Prefill-first scheduling)

**Examples**

* **Cached-token budget:** with a 128-Token Prompt, a 96-Token cache hit, and `max_num_batched_tokens = 32`, only 32 Tokens need computation. The old scheduler charged all 128 and rejected a request that exactly fits the real workload.
* **Decode starvation:** suppose Decode request D is in `running` and one new Prefill request arrives before every scheduling iteration. Because the old scheduler always tried `waiting` first and returned immediately after scheduling any Prefill, D could wait forever even though Decode work was ready.

**Fixes**

* Ask BlockManager for the prospective Prefix Hit before allocation and charge `len(sequence) - cached_tokens` during admission. After allocation, use the Sequence's recorded `num_cached_tokens` when accumulating the current Batch budget, keeping prediction and actual accounting aligned. (Budget calculation)
* Reject an unsupported oversized Prefill with an actionable error instead of leaving it queued indefinitely. (Oversized Prefill)
* Record whether the previous successful scheduling pass was Prefill. When both queues contain work, skip Prefill after a Prefill pass, schedule Decode, then clear the state so the next pass may Prefill again. This bounds Decode waiting without preventing Prefill when no request is running. (Alternation, state transitions)
* Validate that the sampler returns exactly one Token per scheduled Sequence. (Output validation)

### 4. Numerical Stability and Model Configuration

**Files**

* `src/myvllm/layers/layernorm.py`
* `src/myvllm/layers/sampler.py`
* `src/myvllm/layers/linear.py`
* `src/myvllm/models/qwen3.py`
* `src/myvllm/models/llama.py`

**Bugs**

* RMSNorm ignored the model's configured epsilon and computed variance in BF16/FP16. (Original RMSNorm calculation, original Qwen3 Norm construction)
* Sampling Softmax was calculated directly from low-precision logits. (Original sampler)
* Invalid attention-head divisibility could silently create incorrect tensor-parallel shapes. (Original TP head partitioning, original Qwen3 head partitioning)

**Fixes**

* Propagate `rms_norm_epsilon` to Q/K, decoder, and final norms for Qwen3 and Llama. (Qwen3 norms, Llama norms)
* Calculate RMSNorm variance and scaling in FP32, then cast the result back to the input dtype. (RMSNorm)
* Calculate temperature scaling and Softmax probabilities in FP32. (Sampler)
* Validate that attention and KV head counts are divisible by TP size. (TP validation)
* Support explicit `bfloat16`, `float16`, and `float32` model dtypes with clear errors for unsupported values. (Dtype selection)

### 5. CUDA Graph Decode

**File**

* `src/myvllm/engine/model_runner.py`

**Bugs**

* Graph capture used the wrong configuration key, sized hidden-state buffers with `vocab_size`, accessed the graph pool before capture, and mixed static-buffer dtypes. (Original CUDA Graph setup)
* Unsupported Batch sizes could be captured or replayed. (Original Batch-size selection)

**Example**

A decoder layer returns Hidden States shaped `[batch_size, hidden_size]`, but the old static output Buffer was allocated as `[batch_size, vocab_size]`. Since `vocab_size` is usually much larger than `hidden_size`, graph capture recorded tensors with incompatible semantics and replay could fail during `copy_` or feed a wrongly shaped Tensor to the LM Head. Accessing `graph.pool()` before a successful capture added a second failure path.

**Fixes**

* Allocate every static input with its runtime dtype and Shape, allocate Hidden States as `[max_batch_size, hidden_size]` in the configured model dtype, and use the canonical `max_num_sequences` limit. Replay slices these buffers to the actual Batch while preserving the captured addresses. (Static buffers)
* Build the capture-size list only from Batch sizes supported by the static buffers, warm up the model on the capture stream, capture the graph, and only then retain the successful Graph and its Pool for later captures/replay. (Capture loop)

### 6. Engine API Boundary Behavior

**Fixes**

* Reject empty Prompts and invalid `temperature`, `max_tokens`, or context limits. (Prompt validation, sampling validation)
* Read EOS from the tokenizer and remove special tokens during final decoding. (Tokenizer behavior, Detokenization)
* Return a consistent three-element result from `LLMEngine.step()` even for an empty scheduled batch. (Step result)
* Remove the mutable default `SamplingParams` instance from `Sequence`. (Sequence construction)

## Validation

The Release adds boundary tests for Sequence serialization, duplicate Prefix Hashes, cached-context RoPE positions, invalid sampling parameters, empty scheduling results, empty Prompts, idempotent shutdown, and special-token removal. Scheduler tests also cover Prefix-aware budgets and Prefill/Decode alternation. (Engine boundary tests, Scheduler tests)

> Multi-GPU code paths were fixed, but final runtime validation still requires an environment with at least two CUDA devices.
