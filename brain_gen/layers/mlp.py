import torch
import math
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from typing import Optional

from .activations import swiglu


class MLPMoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_experts: int = 4,
        experts_per_token: int = 4,
        swiglu_limit: float = 7.0,
        intermediate_size: Optional[int] = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = d_model

        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.swiglu_limit = swiglu_limit
        self.world_size = 1
        self.norm = nn.RMSNorm(d_model, device=device)
        self.gate = nn.Linear(
            d_model,
            num_experts,
            device=device,
        )

        assert intermediate_size % self.world_size == 0
        self.mlp1_weight = nn.Parameter(
            torch.empty(
                (
                    num_experts,
                    intermediate_size * 2 // self.world_size,
                    d_model,
                ),
                device=device,
            )
        )
        self.mlp1_bias = nn.Parameter(
            torch.empty(
                (num_experts, intermediate_size * 2 // self.world_size),
                device=device,
            )
        )
        self.mlp2_weight = nn.Parameter(
            torch.empty(
                (
                    num_experts,
                    d_model,
                    intermediate_size // self.world_size,
                ),
                device=device,
            )
        )
        self.mlp2_bias = nn.Parameter(
            torch.empty(
                (num_experts, d_model),
                device=device,
            )
        )

        with torch.no_grad():
            self.gate.weight.normal_(std=1.0 / math.sqrt(d_model))
            nn.init.zeros_(self.gate.bias)

            self.mlp1_weight.normal_(std=1.0 / math.sqrt(d_model))
            nn.init.zeros_(self.mlp1_bias)

            self.mlp2_weight.normal_(std=1.0 / math.sqrt(intermediate_size))
            nn.init.zeros_(self.mlp2_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(x)
        original_shape = hidden.shape
        d_model = original_shape[-1]
        hidden_flat = hidden.reshape(-1, d_model)

        gate_logits = self.gate(hidden_flat)
        experts = torch.topk(gate_logits, k=self.experts_per_token, dim=-1, sorted=True)
        expert_weights = torch.nn.functional.softmax(experts.values, dim=-1)
        expert_indices = experts.indices

        mlp1_weight = self.mlp1_weight[expert_indices]
        mlp1_bias = self.mlp1_bias[expert_indices]
        mlp_input = hidden_flat.unsqueeze(1).unsqueeze(-1)
        t = torch.matmul(mlp1_weight, mlp_input).squeeze(-1)
        t = t + mlp1_bias
        t = swiglu(t, limit=self.swiglu_limit)

        mlp2_weight = self.mlp2_weight[expert_indices]
        mlp2_bias = self.mlp2_bias[expert_indices]
        t = torch.matmul(mlp2_weight, t.unsqueeze(-1)).squeeze(-1)
        if self.world_size > 1:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t = t + mlp2_bias

        moe_output = torch.sum(t * expert_weights.unsqueeze(-1), dim=1)
        moe_output = moe_output.view(*original_shape[:-1], d_model)

        return x + moe_output


class MLP(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        dropout: float = 0.0,
    ):
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_model

        self.norm = nn.RMSNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = act_layer()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, residual: bool = True) -> Tensor:
        t = self.norm(x)
        t = self.fc1(t)
        t = self.act(t)
        t = self.drop(t)
        t = self.fc2(t)

        if residual:
            return x + self.drop(t)

        return self.drop(t)
