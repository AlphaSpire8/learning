from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from toy_data import build_toy_cell_perturbation_dataset, one_hot


class CondMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def split_dataset(x_ctrl, p_id, x_pert, ratio=0.8):
    n = x_ctrl.size(0)
    cut = int(n * ratio)
    return (
        (x_ctrl[:cut], p_id[:cut], x_pert[:cut]),
        (x_ctrl[cut:], p_id[cut:], x_pert[cut:]),
    )


def train_diffusion(args, train_data, val_data, gene_dim, num_pert):
    device = train_data[0].device
    timesteps = 64

    beta = torch.linspace(1e-4, 2e-2, timesteps, device=device)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    model = CondMLP(in_dim=gene_dim + gene_dim + num_pert + 1, out_dim=gene_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    x_ctrl, p_id, x_pert = train_data
    p_oh = one_hot(p_id, num_pert)

    for step in range(1, args.steps + 1):
        idx = torch.randint(0, x_ctrl.size(0), (args.batch_size,), device=device)

        ctrl_b = x_ctrl[idx]
        pert_b = x_pert[idx]
        p_b = p_oh[idx]

        t_idx = torch.randint(0, timesteps, (args.batch_size,), device=device)
        t = t_idx.float().unsqueeze(-1) / (timesteps - 1)

        eps = torch.randn_like(pert_b)
        a_bar = alpha_bar[t_idx].unsqueeze(-1)
        x_t = torch.sqrt(a_bar) * pert_b + torch.sqrt(1 - a_bar) * eps

        inp = torch.cat([x_t, ctrl_b, p_b, t], dim=-1)
        pred_eps = model(inp)
        loss = F.mse_loss(pred_eps, eps)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 200 == 0 or step == 1:
            print(f"[cell-diffusion] step={step:04d} loss={loss.item():.4f}")

    val_mse = evaluate_diffusion(model, val_data, num_pert, timesteps, beta, alpha, alpha_bar)
    return val_mse


@torch.no_grad()
def evaluate_diffusion(model, val_data, num_pert, timesteps, beta, alpha, alpha_bar):
    x_ctrl, p_id, x_pert = val_data
    p_oh = one_hot(p_id, num_pert)

    x = torch.randn_like(x_pert)
    n = x.size(0)
    for i in reversed(range(timesteps)):
        t = torch.full((n, 1), i / (timesteps - 1), device=x.device)
        inp = torch.cat([x, x_ctrl, p_oh, t], dim=-1)
        pred_eps = model(inp)

        a = alpha[i]
        a_bar = alpha_bar[i]
        b = beta[i]
        x0_hat = (x - torch.sqrt(1 - a_bar) * pred_eps) / torch.sqrt(a_bar)
        mean = torch.sqrt(a) * x0_hat + torch.sqrt(1 - a) * pred_eps

        if i > 0:
            x = mean + torch.sqrt(b) * torch.randn_like(x)
        else:
            x = mean

    return F.mse_loss(x, x_pert).item()


def train_flow(args, train_data, val_data, gene_dim, num_pert):
    device = train_data[0].device
    model = CondMLP(in_dim=gene_dim + gene_dim + num_pert + 1, out_dim=gene_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    x_ctrl, p_id, x_pert = train_data
    p_oh = one_hot(p_id, num_pert)

    for step in range(1, args.steps + 1):
        idx = torch.randint(0, x_ctrl.size(0), (args.batch_size,), device=device)

        ctrl_b = x_ctrl[idx]
        target_b = x_pert[idx]
        p_b = p_oh[idx]

        x0 = torch.randn_like(target_b)
        t = torch.rand(args.batch_size, 1, device=device)
        x_t = (1 - t) * x0 + t * target_b
        u_t = target_b - x0

        inp = torch.cat([x_t, ctrl_b, p_b, t], dim=-1)
        pred_u = model(inp)
        loss = F.mse_loss(pred_u, u_t)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 200 == 0 or step == 1:
            print(f"[cell-flow] step={step:04d} loss={loss.item():.4f}")

    val_mse = evaluate_flow(model, val_data, num_pert, ode_steps=80)
    return val_mse


@torch.no_grad()
def evaluate_flow(model, val_data, num_pert, ode_steps=80):
    x_ctrl, p_id, x_pert = val_data
    p_oh = one_hot(p_id, num_pert)

    x = torch.randn_like(x_pert)
    dt = 1.0 / ode_steps
    n = x.size(0)

    for i in range(ode_steps):
        t = torch.full((n, 1), i / ode_steps, device=x.device)
        inp = torch.cat([x, x_ctrl, p_oh, t], dim=-1)
        v = model(inp)
        x = x + dt * v

    return F.mse_loss(x, x_pert).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["diffusion", "flow"], required=True)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n", type=int, default=6000)
    p.add_argument("--gene-dim", type=int, default=16)
    p.add_argument("--num-perturbations", type=int, default=4)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = build_toy_cell_perturbation_dataset(
        n=args.n,
        gene_dim=args.gene_dim,
        num_perturbations=args.num_perturbations,
        device=device,
    )

    train_data, val_data = split_dataset(ds.x_ctrl, ds.perturb_id, ds.x_pert)

    if args.method == "diffusion":
        mse = train_diffusion(args, train_data, val_data, args.gene_dim, args.num_perturbations)
    else:
        mse = train_flow(args, train_data, val_data, args.gene_dim, args.num_perturbations)

    print(f"[result] method={args.method} val_mse={mse:.4f}")


if __name__ == "__main__":
    main()
