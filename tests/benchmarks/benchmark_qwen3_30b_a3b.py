"""Qwen3-30B-A3B (MoE) throughput baseline for TP=2 and TP=4.

Unlike the Qwen3-32B stress suite, this model is a Base checkpoint, so prompts
are plain completions rather than chat-templated turns. Each request carries a
unique marker in its first block so no prefix cache sharing inflates the
numbers: this measures raw prefill and decode throughput.

Example:
    python tests/benchmarks/benchmark_qwen3_30b_a3b.py \
        --world-size 2 --concurrency 16 --prompt-tokens 1024 --output-tokens 128
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parents[1]
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark_qwen3_32b_stress import print_result, run_workload
from main_qwen3_30b_a3b import config as moe_config
from myvllm.engine.llm_engine import LLMEngine

FILLER = (
    "Reference segment: distributed inference uses tensor parallel linear "
    "layers, paged KV cache blocks, prefix hashing, continuous scheduling, "
    "Flash Attention during prefill, and paged attention during decode. "
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure Qwen3-30B-A3B throughput at a given TP size."
    )
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--max-batched-tokens", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def build_prompts(tokenizer, concurrency: int, target_tokens: int) -> list[str]:
    """Build unique prompts of approximately ``target_tokens`` tokens each."""
    unit_tokens = len(tokenizer.encode(FILLER))
    repetitions = max(1, (target_tokens - 16) // unit_tokens)
    return [
        f"Document {index:04d} unique identifier {index * 7919:08d}.\n"
        + FILLER * repetitions
        + "\nSummary of the document above:"
        for index in range(concurrency)
    ]


def main() -> None:
    args = parse_args()
    if args.world_size > torch.cuda.device_count():
        raise ValueError(
            f"world_size {args.world_size} exceeds the "
            f"{torch.cuda.device_count()} visible CUDA devices"
        )

    tokenizer = AutoTokenizer.from_pretrained(moe_config["model_name_or_path"])
    prompts = build_prompts(tokenizer, args.concurrency, args.prompt_tokens)
    actual_max_prompt = max(len(tokenizer.encode(prompt)) for prompt in prompts)

    config = dict(moe_config)
    config.update(
        world_size=args.world_size,
        block_size=args.block_size,
        max_num_sequences=args.concurrency,
        max_num_batched_tokens=max(args.max_batched_tokens, actual_max_prompt),
        max_model_length=actual_max_prompt + args.output_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )

    print("=== Qwen3-30B-A3B throughput baseline ===")
    print(f"tensor parallel size:      {args.world_size}")
    print(f"concurrency:               {args.concurrency}")
    print(f"actual max prompt tokens:  {actual_max_prompt:,}")
    print(f"output tokens/request:     {args.output_tokens:,}")
    print(f"engine max model length:   {config['max_model_length']:,}")
    print(f"max batched tokens:        {config['max_num_batched_tokens']:,}")
    for index in range(args.world_size):
        properties = torch.cuda.get_device_properties(index)
        print(
            f"GPU {index}: {properties.name}, "
            f"{properties.total_memory / 2**30:.1f} GiB"
        )

    engine = LLMEngine(config=config)
    try:
        result = run_workload(
            engine,
            f"moe-baseline-tp{args.world_size}",
            prompts,
            args.output_tokens,
        )
    finally:
        engine.exit()

    print_result(result)

    if args.json_output:
        args.json_output.write_text(
            json.dumps(
                {
                    "configuration": {
                        "model": config["model_name_or_path"],
                        "tensor_parallel_size": args.world_size,
                        "concurrency": args.concurrency,
                        "actual_max_prompt_tokens": actual_max_prompt,
                        "output_tokens": args.output_tokens,
                        "max_model_length": config["max_model_length"],
                        "max_num_batched_tokens": (
                            config["max_num_batched_tokens"]
                        ),
                        "block_size_tokens": args.block_size,
                        "gpu_memory_utilization": args.gpu_memory_utilization,
                        "enforce_eager": True,
                        "pytorch_version": torch.__version__,
                        "cuda_version": torch.version.cuda,
                    },
                    "results": [asdict(result)],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
        print(f"\nJSON result written to {args.json_output}")


if __name__ == "__main__":
    main()
