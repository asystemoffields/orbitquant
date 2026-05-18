from __future__ import annotations

import copy

import torch
import torch.nn as nn

from gemma4_pmra_orbit_stack_eval import (
    build_rotation_spec,
    make_folded_mlp_forward,
    make_mlp_forward,
)


class TinyMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=True)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(z)


def compare_rotation(rotation_name: str) -> None:
    torch.manual_seed(1234 + len(rotation_name))
    hidden_size = 7
    intermediate_size = 16
    block_size = 8
    bits = 2
    alpha = 0.375

    base = TinyMLP(hidden_size, intermediate_size)
    current = copy.deepcopy(base)
    folded = copy.deepcopy(base)

    calibration_z = torch.randn(31, intermediate_size)
    down_weight = base.down_proj.weight.detach().float()
    spec = build_rotation_spec(rotation_name, calibration_z, down_weight, intermediate_size, block_size)

    current.forward = make_mlp_forward(current, bits, spec, alpha)
    folded.forward = make_folded_mlp_forward(folded, bits, spec, alpha)

    x = torch.randn(3, 5, hidden_size)
    with torch.inference_mode():
        current_out = current(x)
        folded_out = folded(x)

    torch.testing.assert_close(current_out, folded_out, rtol=1e-5, atol=1e-5)
    print(f"folded MLP equivalence passed: {rotation_name}")


def main() -> None:
    for rotation_name in [
        "identity",
        "block_hadamard",
        "preperm_activation_max_hadamard",
        "preperm_boundary_rms_hadamard",
    ]:
        compare_rotation(rotation_name)


if __name__ == "__main__":
    main()
