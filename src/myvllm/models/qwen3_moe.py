import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from myvllm.layers import *
from myvllm.models.qwen3 import Qwen3Attention, get_qwen_positions


class RowParallelLinearNoReduce(RowParallelLinear):
    """Row-parallel linear that leaves its partial sum un-reduced.

    A sparse MoE layer runs many of these back to back. Reducing inside every
    expert would cost one all-reduce per active expert per layer, so the
    surrounding block sums the partial outputs locally and reduces once.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class Qwen3MoeExpert(nn.Module):
    """One expert FFN: SwiGLU over ``moe_intermediate_size``, sharded over TP."""

    def __init__(self, hidden_size: int, moe_intermediate_size: int):
        super().__init__()
        self.gate_up = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[moe_intermediate_size] * 2,
            bias=False,
        )
        self.activation = SiluAndMul()
        self.down_proj = RowParallelLinearNoReduce(
            input_size=moe_intermediate_size,
            output_size=hidden_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.activation(self.gate_up(x)))


class Qwen3MoeSparseMoeBlock(nn.Module):
    """Router + 128 experts, top-k selection, weighted sum, single all-reduce."""

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        norm_topk_prob: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.hidden_size = hidden_size
        self.tp_size = dist.get_world_size()

        # Router is replicated: same input on every rank -> same routing.
        self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([
            Qwen3MoeExpert(hidden_size, moe_intermediate_size)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x can be 2D (varlen prefill) or 3D (batched). Flatten to 2D for
        # per-token routing, restore shape at the end.
        orig_shape = x.shape
        x_flat = x.view(-1, self.hidden_size)
        num_tokens = x_flat.size(0)

        # Route in fp32 for numerical stability of softmax/topk.
        router_logits = self.gate(x_flat).float()
        routing_weights = F.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = routing_weights.topk(self.top_k, dim=-1)
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(x_flat.dtype)

        # Accumulator holds partial (un-reduced) sums.
        final = torch.zeros_like(x_flat)

        # For each expert, gather the tokens routed to it, run the FFN, scatter
        # the weighted output back. Skips experts with no assigned tokens so
        # empty experts cost only a boolean mask reduction.
        # expert_mask[e, k, t] = 1 iff token t assigned expert e at slot k.
        # We iterate over experts rather than tokens to batch the matmul.
        flat_expert_ids = topk_indices.view(-1)              # [T * top_k]
        flat_token_ids = (
            torch.arange(num_tokens, device=x_flat.device)
            .unsqueeze(-1)
            .expand(-1, self.top_k)
            .reshape(-1)
        )                                                     # [T * top_k]
        flat_weights = topk_weights.view(-1)                  # [T * top_k]

        for expert_id in range(self.num_experts):
            selection = (flat_expert_ids == expert_id).nonzero(as_tuple=True)[0]
            if selection.numel() == 0:
                continue
            token_ids = flat_token_ids[selection]
            weights = flat_weights[selection].unsqueeze(-1)
            expert_in = x_flat.index_select(0, token_ids)
            expert_out = self.experts[expert_id](expert_in) * weights
            final.index_add_(0, token_ids, expert_out)

        # One all-reduce for all experts + all top-k slots combined.
        if self.tp_size > 1:
            dist.all_reduce(final, op=dist.ReduceOp.SUM)

        return final.view(orig_shape)


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
        rms_norm_epsilon: float,
        qkv_bias: bool,
        base: float,
        max_position: int,
        moe_intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        norm_topk_prob: bool,
        block_size: int,
    ):
        super().__init__()
        gamma = torch.ones(hidden_size)
        self.input_layernorm = LayerNorm(gamma, eps=rms_norm_epsilon)
        self.self_attn = Qwen3Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            rms_norm_epsilon=rms_norm_epsilon,
            qkv_bias=qkv_bias,
            base=base,
            max_position=max_position,
            block_size=block_size,
            use_large_scale_attention=False,
        )
        self.post_attention_layernorm = LayerNorm(gamma, eps=rms_norm_epsilon)
        self.mlp = Qwen3MoeSparseMoeBlock(
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            norm_topk_prob=norm_topk_prob,
        )

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            x, residual = self.input_layernorm(x, residual)
        else:
            residual = x
            x = self.input_layernorm(x)

        from myvllm.utils import get_context
        context = get_context()
        positions = get_qwen_positions(context, x.device, x.size(0))

        x = self.self_attn(x, positions=positions)
        x, residual = self.post_attention_layernorm(x, residual)
        x = self.mlp(x)
        return x, residual


class Qwen3MoeModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
        rms_norm_epsilon: float,
        qkv_bias: bool,
        base: float,
        max_position: int,
        moe_intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        norm_topk_prob: bool,
        num_layers: int,
        block_size: int,
    ):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
        )
        self.layers = nn.ModuleList([
            Qwen3MoeDecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                scale=scale,
                num_kv_heads=num_kv_heads,
                rms_norm_epsilon=rms_norm_epsilon,
                qkv_bias=qkv_bias,
                base=base,
                max_position=max_position,
                moe_intermediate_size=moe_intermediate_size,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                norm_topk_prob=norm_topk_prob,
                block_size=block_size,
            )
            for _ in range(num_layers)
        ])
        gamma = torch.ones(hidden_size)
        self.norm = LayerNorm(gamma, eps=rms_norm_epsilon)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            x, residual = layer(x, residual)
        x, _ = self.norm(x, residual)
        return x


class Qwen3MoeForCausalLM(nn.Module):
    # HF experts.{E}.gate_proj / up_proj  ->  experts.{E}.gate_up (fused)
    # HF q_proj / k_proj / v_proj         ->  qkv_projection
    packed_modules_mapping = {
        "qkv_projection": [
            ("q_proj", "q"),
            ("k_proj", "k"),
            ("v_proj", "v"),
        ],
        "gate_up": [
            ("gate_proj", 0),
            ("up_proj", 1),
        ],
    }

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
        rms_norm_epsilon: float,
        qkv_bias: bool,
        base: float,
        max_position: int,
        moe_intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        norm_topk_prob: bool,
        num_layers: int,
        tie_word_embeddings: bool,
        block_size: int,
    ):
        super().__init__()
        self.model = Qwen3MoeModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            rms_norm_epsilon=rms_norm_epsilon,
            qkv_bias=qkv_bias,
            base=base,
            max_position=max_position,
            moe_intermediate_size=moe_intermediate_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            norm_topk_prob=norm_topk_prob,
            num_layers=num_layers,
            block_size=block_size,
        )
        self.lm_head = ParallelLMHead(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
        )
        if tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
