from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class PerturbationToyData:
    x_ctrl: torch.Tensor
    perturb_id: torch.Tensor
    x_pert: torch.Tensor


def make_mixture_2d(n: int, device: str = "cpu") -> torch.Tensor:
    """生成两个高斯团的 2D 数据，作为基础生成任务。"""
    means = torch.tensor([[2.0, 0.0], [-2.0, 0.0]], device=device)
    comp = torch.randint(0, 2, (n,), device=device)
    noise = 0.45 * torch.randn(n, 2, device=device)
    return means[comp] + noise


def build_toy_cell_perturbation_dataset(
    n: int = 6000,
    gene_dim: int = 16,
    num_perturbations: int = 4,
    seed: int = 42,
    device: str = "cpu",
) -> PerturbationToyData:
    """
    构建一个可学习的“细胞扰动”玩具数据集。

    x_ctrl: 未扰动表达
    perturb_id: 扰动编号
    x_pert: 扰动后表达 = 非线性基因网络变换 + 扰动特异偏移 + 噪声
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    x_ctrl = torch.randn(n, gene_dim, generator=g, device=device)
    perturb_id = torch.randint(0, num_perturbations, (n,), generator=g, device=device)

    # 共享的“基因调控”映射
    w = torch.randn(gene_dim, gene_dim, generator=g, device=device) / math.sqrt(gene_dim)
    hidden = torch.tanh(x_ctrl @ w)

    # 每个扰动有不同作用方向
    pert_embed = torch.randn(num_perturbations, gene_dim, generator=g, device=device)
    p = pert_embed[perturb_id]

    # 构造扰动后表达（有系统规律 + 噪声）
    x_pert = x_ctrl + 0.8 * hidden + 0.6 * p + 0.05 * torch.randn_like(x_ctrl, generator=g)

    return PerturbationToyData(x_ctrl=x_ctrl, perturb_id=perturb_id, x_pert=x_pert)


def one_hot(ids: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(ids, num_classes=num_classes).float()
