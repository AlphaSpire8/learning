from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from toy_data import make_mixture_2d


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t.unsqueeze(-1))


class NoisePredictor(nn.Module):
    def __init__(self, data_dim: int = 2, hidden: int = 128, tdim: int = 32):
        super().__init__()
        self.t_emb = TimeEmbedding(tdim)
        self.net = nn.Sequential(
            nn.Linear(data_dim + tdim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, data_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.t_emb(t)
        return self.net(torch.cat([x_t, te], dim=-1))


def build_schedule(timesteps: int, device: str):
    beta = torch.linspace(1e-4, 2e-2, timesteps, device=device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return beta, alpha, alpha_bar


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x0 = make_mixture_2d(args.n_train, device=device)

    model = NoisePredictor().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    beta, alpha, alpha_bar = build_schedule(args.timesteps, device)

    for step in range(1, args.steps + 1):
        idx = torch.randint(0, x0.size(0), (args.batch_size,), device=device)
        batch = x0[idx]

        t_idx = torch.randint(0, args.timesteps, (args.batch_size,), device=device)
        t = t_idx.float() / (args.timesteps - 1)

        eps = torch.randn_like(batch)
        a_bar = alpha_bar[t_idx].unsqueeze(-1)
        x_t = torch.sqrt(a_bar) * batch + torch.sqrt(1.0 - a_bar) * eps

        pred = model(x_t, t)
        loss = F.mse_loss(pred, eps)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 100 == 0 or step == 1:
            print(f"[diffusion] step={step:04d} loss={loss.item():.4f}")

    return model, (beta, alpha, alpha_bar), device


@torch.no_grad()
def sample(model, schedule, n: int, device: str):
    beta, alpha, alpha_bar = schedule
    timesteps = beta.numel()

    x = torch.randn(n, 2, device=device)
    for i in reversed(range(timesteps)):
        t = torch.full((n,), i / (timesteps - 1), device=device)
        pred_eps = model(x, t)

        a = alpha[i]
        a_bar = alpha_bar[i]
        b = beta[i]

        x0_hat = (x - torch.sqrt(1 - a_bar) * pred_eps) / torch.sqrt(a_bar)
        mean = torch.sqrt(a) * x0_hat + torch.sqrt(1 - a) * pred_eps

        if i > 0:
            x = mean + torch.sqrt(b) * torch.randn_like(x)
        else:
            x = mean
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=5000)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--timesteps", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--sample-size", type=int, default=1500)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    model, schedule, device = train(args)

    data = make_mixture_2d(args.sample_size, device=device).cpu()
    gen = sample(model, schedule, args.sample_size, device=device).cpu()

    if args.plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.scatter(data[:, 0], data[:, 1], s=5)
        plt.title("Target data")
        plt.axis("equal")
        plt.subplot(1, 2, 2)
        plt.scatter(gen[:, 0], gen[:, 1], s=5)
        plt.title("Diffusion samples")
        plt.axis("equal")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
