# Summary — Commit ID Range: `23d95ae` → `acdac94`

**Tag:** `release-1`

This release establishes correct cross-request Prefix Cache block reuse and makes KV Cache capacity checks aware of reusable blocks.

## Major Fixes

### 1. Preserve KV Blocks Across Requests

**Files**

* `src/myvllm/engine/block_manager.py`
* `src/myvllm/engine/sequence.py`

**Bugs**

* Releasing a request discarded the block's token metadata, preventing later requests from reusing the cached prefix. (Original deallocation)
* Allocation treated cache-hit and newly allocated blocks as the same operation, which reset valid cached metadata before the hit was reused. (Original allocation path, original hit handling)
* Prefix matching could continue after a miss, even though only a contiguous prefix is reusable. (Original per-block matching)
* A fully occupied final block was reported as containing zero tokens. (Original sequence logic)

**Example**

With `block_size = 4`, request A uses `[10, 11, 12, 13, 20]`. Its first Block `[10, 11, 12, 13]` is a reusable Prefix. After A finishes, request B arrives with `[10, 11, 12, 13, 30]`. The old deallocation path cleared the saved Token IDs, so B either missed the cache or treated the remaining Hash mapping as a collision. Even if B found the physical Block, the generic allocation path reset it before reuse.

**Fixes**

* `_allocate_new_block()` first removes a stale Hash mapping only when it still points to the recycled Block, then resets metadata, removes the Block from the free queue, writes the new Hash/Tokens, and initializes `ref_count`. `_allocate_hit_block()` only moves an inactive cached Block back to the used set and increments its reference count. This separation prevents valid cache data from being reset. (Allocation paths)
* `_deallocate_block()` now drops only active ownership: it moves the physical Block to the free queue but deliberately retains its Hash and Token IDs. The metadata remains a cache candidate until that physical Block is recycled for different Tokens. (Deallocation fix)
* `_get_prefix_hit_blocks()` hashes Blocks from left to right and stops on the first missing Hash or Token mismatch. It limits lookup to `(num_tokens - 1) // block_size`, ensuring the final Prompt Token is computed and produces the first completion Logits. (Prefix matching)
* Correct `last_block_num_tokens` for empty, partial, and exactly full final blocks. (Sequence fix)

### 2. Prefix-Aware Capacity Checks

**Bug**

`can_allocate()` required one free block for every logical request block. This rejected requests even when some Prefix Cache blocks were already in use and could be shared by incrementing their reference counts. (Original check)

**Example**

A request needs three logical Blocks and its first Block is already active for another request. If only two physical Blocks are free, the old check compares `2 >= 3` and rejects it. The real allocation needs only two new Blocks: the first is shared by increasing `ref_count`, so the request should be admitted.

**Fixes**

* Centralize cache-hit discovery in `_get_prefix_hit_blocks()` so checking and allocation use the same contiguous-prefix result. (Helper)
* Compute `num_required_free_blocks = seq.num_blocks - num_shared_hit_blocks`. Only hits with `ref_count > 0` are subtracted: an inactive cached Block is still physically present in the free queue and must be claimed from it during allocation. (Capacity calculation)
* During allocation, append the hit Block ID to the request's Block Table, add one Block worth of Tokens to `num_cached_tokens`, and either claim the inactive cached Block or increment an active Block's `ref_count`. Only the unmatched suffix consumes new physical Blocks. (Allocation)

## Result

Two requests with a common full-block prefix can now share the same physical KV blocks. Capacity admission reflects the physical blocks that must actually be acquired rather than the request's total logical block count.
