import sys, os
from pathlib import Path
import torch.distributed as dist

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from myvllm.engine.llm_engine import LLMEngine as LLM
from myvllm.sampling_parameters import SamplingParams

config = {
    'max_num_sequences': 16,
    'max_num_batched_tokens': 1024,
    'max_cached_blocks': 1024,

    'dtype': 'bfloat16',
    'max_model_length': 128,
    'gpu_memory_utilization': 0.9,

    'block_size': 256,
    'world_size': 1,

    'enforce_eager': True,

    # model params (Llama 3.1 8B Instruct)
    'model_name_or_path': '/root/paddlejob/gpfsspace/Llama-3.1-8B-Instruct',
    'vocab_size': 128256,
    'hidden_size': 4096,
    'head_dim': 128,
    'num_qo_heads': 32,
    'num_kv_heads': 8,
    'has_attn_bias': False,
    'rms_norm_epsilon': 1e-5,
    'rope_base': 500000,
    'max_position_embeddings': 131072,
    'intermediate_size': 14336,
    'ffn_bias': False,
    'num_layers': 32,
    'tie_word_embeddings': False,  # 8B does NOT tie embeddings (unlike 1B)
    'eos': 128009,
}


def main():
    model_name = config.get('model_name_or_path')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(config=config)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256, max_model_length=128)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
        "give me your opinion on the impact of artificial intelligence on society",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    generated_texts = outputs['text']

    for prompt, output in zip(prompts, generated_texts):
        print("\n")
        print(f"Prompt: {prompt}")
        print(f"Completion: {output}")


if __name__ == "__main__":
    main()
