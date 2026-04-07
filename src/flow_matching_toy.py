from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from toy_data import make_mixture_2d


class VelocityField(nn.Module):
    def __init__(self, data_dim: int = 2, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, data_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_t, t.unsqueeze(-1)], dim=-1))


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x_data = make_mixture_2d(args.n_train, device=device)
    model = VelocityField().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for step in range(1, args.steps + 1):
        idx = torch.randint(0, x_data.size(0), (args.batch_size,), device=device)
        x1 = x_data[idx]  # data endpoint
        x0 = torch.randn_like(x1)  # noise endpoint

        t = torch.rand(args.batch_size, device=device)
        x_t = (1 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * x1
        target_u = x1 - x0

        pred_u = model(x_t, t)
        loss = F.mse_loss(pred_u, target_u)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 100 == 0 or step == 1:
            print(f"[flow] step={step:04d} loss={loss.item():.4f}")

    return model, device


@torch.no_grad()
def sample(model, n: int, device: str, ode_steps: int = 80):
    x = torch.randn(n, 2, device=device)
    dt = 1.0 / ode_steps

    for i in range(ode_steps):
        t_scalar = i / ode_steps
        t = torch.full((n,), t_scalar, device=device)
        v = model(x, t)
        x = x + dt * v
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=5000)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--sample-size", type=int, default=1500)
    p.add_argument("--ode-steps", type=int, default=80)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    model, device = train(args)

    data = make_mixture_2d(args.sample_size, device=device).cpu()
    gen = sample(model, args.sample_size, device=device, ode_steps=args.ode_steps).cpu()

    if args.plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.scatter(data[:, 0], data[:, 1], s=5)
        plt.title("Target data")
        plt.axis("equal")
        plt.subplot(1, 2, 2)
        plt.scatter(gen[:, 0], gen[:, 1], s=5)
        plt.title("Flow samples")
        plt.axis("equal")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
