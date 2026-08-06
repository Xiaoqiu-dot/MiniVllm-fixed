import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from myvllm.engine.llm_engine import LLMEngine as LLM
from myvllm.sampling_parameters import SamplingParams

MODEL_PATH = '/root/paddlejob/gpfsspace/mapengtao_exp/models/Qwen3-30B-A3B-Base'

config = {
    'max_num_sequences': 8,
    'max_num_batched_tokens': 2048,
    'max_cached_blocks': 1024,

    'dtype': 'bfloat16',
    'max_model_length': 1024,
    'gpu_memory_utilization': 0.9,

    'block_size': 256,
    # TP=2: num_kv_heads(4) and moe_intermediate_size(768) are both divisible.
    'world_size': 2,

    # MoE routing is data dependent (nonzero/index_select force host syncs), so
    # decoding cannot be captured into a CUDA graph.
    'enforce_eager': True,

    # model params (Qwen3-30B-A3B-Base, MoE)
    'model_name_or_path': MODEL_PATH,
    'vocab_size': 151936,
    'hidden_size': 2048,
    'num_heads': 32,
    'num_kv_heads': 4,
    'head_dim': 128,
    # Attention divides by sqrt(head_dim) internally, so pass 1 here.
    'scale': 1,
    'rms_norm_epsilon': 1e-6,
    'qkv_bias': False,
    'base': 1000000.0,
    'max_position': 32768,
    'moe_intermediate_size': 768,
    'num_experts': 128,
    'num_experts_per_tok': 8,
    'norm_topk_prob': True,
    'num_layers': 48,
    'tie_word_embeddings': False,
    'eos': 151643,
}


def main():
    llm = LLM(config=config)

    sampling_params = SamplingParams(
        temperature=0.6, max_tokens=128, max_model_length=1024
    )
    # Base model (not Instruct): use plain completion prompts, no chat template.
    prompts = [
        "The capital of France is",
        "List the first ten prime numbers:",
        "Mixture-of-Experts models are efficient because",
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, completion in zip(prompts, outputs['text']):
        print("\n")
        print(f"Prompt: {prompt}")
        print(f"Completion: {completion}")


if __name__ == "__main__":
    main()
