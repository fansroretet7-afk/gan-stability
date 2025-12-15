import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt

DEVICE = "cpu"
BATCH_SIZE = 64
EPOCHS = 15
LATENT_DIM = 16
LR = 1e-3
R1_LAMBDA = 10.0
NOISE_STD = 0.1

METHODS = [
    "vanilla",
    "r1",
    "instance_noise",
    "ttur",
    "optimistic"
]

# ======================
# DATA
# ======================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)
dataset = Subset(dataset, range(1000))
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

criterion = nn.BCEWithLogitsLoss()


results = {}

for METHOD in METHODS:
    print(f"\n=== {METHOD.upper()} ===")

    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)

    if METHOD == "ttur":
        opt_G = optim.Adam(G.parameters(), lr=LR)
        opt_D = optim.Adam(D.parameters(), lr=4 * LR)
    else:
        opt_G = optim.Adam(G.parameters(), lr=LR)
        opt_D = optim.Adam(D.parameters(), lr=LR)

    G_losses, D_losses = [], []

    for epoch in range(EPOCHS):
        for real, _ in loader:
            real = real.to(DEVICE)
            batch = real.size(0)

            if METHOD == "instance_noise":
                real = real + NOISE_STD * torch.randn_like(real)

            # ---- Discriminator ----
            z = torch.randn(batch, LATENT_DIM).to(DEVICE)
            fake = G(z).detach()

            if METHOD == "instance_noise":
                fake = fake + NOISE_STD * torch.randn_like(fake)

            real_logits = D(real)
            fake_logits = D(fake)

            d_loss = criterion(real_logits, torch.ones_like(real_logits)) + \
                     criterion(fake_logits, torch.zeros_like(fake_logits))

            if METHOD == "r1":
                real.requires_grad_(True)
                grad = torch.autograd.grad(
                    D(real).sum(), real, create_graph=True
                )[0]
                r1 = grad.view(batch, -1).norm(2, dim=1).mean()
                d_loss = d_loss + R1_LAMBDA * r1

            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            # ---- Generator ----
            z = torch.randn(batch, LATENT_DIM).to(DEVICE)
            fake = G(z)
            g_loss = criterion(D(fake), torch.ones_like(fake_logits))

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            if METHOD == "optimistic":
                opt_G.step()
                opt_D.step()

            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())

    results[METHOD] = (G_losses, D_losses)


fig, axes = plt.subplots(len(METHODS), 2, figsize=(12, 2.5 * len(METHODS)))

for i, method in enumerate(METHODS):
    G_l, D_l = results[method]

    axes[i, 0].plot(G_l)
    axes[i, 0].set_title(f"{method} — G loss")

    axes[i, 1].plot(D_l)
    axes[i, 1].set_title(f"{method} — D loss")

plt.tight_layout()
plt.show()