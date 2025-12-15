import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

methods = [
    "Vanilla",
    "R1",
    "Instance noise",
    "TTUR",
    "Extragradient",
    "Optimistic"
]

t = np.linspace(0, 12, 300)

fig, axes = plt.subplots(2, 6, figsize=(18, 6))

TRAJ_LIM = (-1.5, 1.5)
SPEC_XLIM = (-0.7, 0.2)
SPEC_YLIM = (-0.1, 1.3)

def add_arrows(ax, x, y, n_arrows=5, color='black'):
    """Добавляет стрелки на траекторию для направления времени"""
    idx = np.linspace(0, len(x)-2, n_arrows, dtype=int)
    for i in idx:
        ax.annotate('', xy=(x[i+1], y[i+1]), xytext=(x[i], y[i]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

for i, method in enumerate(methods):

    # -------- ДАННЫЕ --------
    if method == "Vanilla":
        x = np.cos(t)
        y = np.sin(t)
        eig_real = np.zeros(8)
        eig_imag = np.linspace(0.6, 1.2, 8)
    elif method == "R1":
        x = 0.85 * np.cos(t)
        y = 0.85 * np.sin(t)
        eig_real = -0.1 * np.ones(8)
        eig_imag = np.linspace(0.5, 1.0, 8)
    elif method == "Instance noise":
        x = 0.9 * np.cos(t) + 0.03 * np.random.randn(len(t))
        y = 0.9 * np.sin(t) + 0.03 * np.random.randn(len(t))
        eig_real = -0.15 * np.ones(8)
        eig_imag = np.linspace(0.4, 0.9, 8)
    elif method == "TTUR":
        x = 0.9 * np.cos(t)
        y = 0.7 * np.sin(t)
        eig_real = -0.2 * np.ones(8)
        eig_imag = np.linspace(0.3, 0.8, 8)
    elif method == "Extragradient":
        x = np.exp(-0.15 * t) * np.cos(t)
        y = np.exp(-0.15 * t) * np.sin(t)
        eig_real = -0.5 * np.ones(8)
        eig_imag = np.linspace(0.2, 0.5, 8)
    elif method == "Optimistic":
        x = np.exp(-0.12 * t) * np.cos(t)
        y = np.exp(-0.12 * t) * np.sin(t)
        eig_real = -0.4 * np.ones(8)
        eig_imag = np.linspace(0.15, 0.45, 8)

    # ================== ТРАЕКТОРИИ ==================
    ax_traj = axes[0, i]
    ax_traj.plot(x, y, color='black', linewidth=2)
    add_arrows(ax_traj, x, y, n_arrows=6)
    ax_traj.scatter(x[0], y[0], marker='o', s=35, color='black')   # старт
    ax_traj.scatter(x[-1], y[-1], marker='x', s=50, color='black') # конец

    ax_traj.axhline(0, linewidth=0.7)
    ax_traj.axvline(0, linewidth=0.7)
    ax_traj.set_xlim(TRAJ_LIM)
    ax_traj.set_ylim(TRAJ_LIM)
    ax_traj.set_aspect('equal')
    ax_traj.set_title(method)
    ax_traj.set_xlabel(r'$\theta_G$')
    ax_traj.set_ylabel(r'$\theta_D$')
    ax_traj.grid(True)

    # ================== СПЕКТР ==================
    ax_spec = axes[1, i]
    ax_spec.scatter(eig_real, eig_imag, color='green', marker='^', s=45)

    ax_spec.axhline(0, linewidth=0.7)
    ax_spec.axvline(0, linewidth=0.7)
    ax_spec.set_xlim(SPEC_XLIM)
    ax_spec.set_ylim(SPEC_YLIM)
    ax_spec.set_aspect('equal', adjustable='box')
    ax_spec.set_xlabel(r'Re$(\lambda)$')
    ax_spec.set_ylabel(r'Im$(\lambda)$')
    ax_spec.grid(True)

# Горизонтальные подписи над верхним и нижним рядом подграфиков
fig.text(0.5, 0.97, "Parameter trajectories (θ_G, θ_D) with direction", ha='center', fontsize=12)
fig.text(0.5, 0.48, "Jacobian spectrum (eigenvalues λ)", ha='center', fontsize=12)

plt.tight_layout()
plt.savefig("toy_gan_trajectories_with_arrows.png", dpi=300, bbox_inches='tight')
plt.show()