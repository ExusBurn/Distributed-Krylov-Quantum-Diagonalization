import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("./TEST", exist_ok=True)

plt.rcParams.update({
    "figure.facecolor":  "#ffffff",
    "axes.facecolor":    "#ffffff",
    "axes.edgecolor":    "#000000",
    "axes.labelcolor":   "#000000",
    "axes.titlecolor":   "#000000",
    "axes.grid":         True,
    "grid.color":        "#dddddd",
    "grid.linewidth":    0.8,
    "xtick.color":       "#000000",
    "ytick.color":       "#000000",
    "font.family":       "sans-serif",
    "font.size":         11,
    "lines.linewidth":   2.2,
})

BLUE   = "#378ADD"
GREEN  = "#1D9E75"
AMBER  = "#EF9F27"

# Data
K = [1, 2, 3, 4]

cuda_E0  = [5.00000000, 3.62340579, 3.5021, 3.45849098]
cudaq_E0 = [4.99999467, 3.62348811, 3.5038, 3.46321394]
omp_E0   = [5.00000000, 3.62340579, 3.5021, 3.45849098]

E0_exact = 3.04

fig, ax = plt.subplots(figsize=(4.6, 4.4))

# Plot lines
ax.plot(K, cuda_E0,  marker="o", color=BLUE,  label="CUDA (MPI, RTX A5000)")
ax.plot(K, cudaq_E0, marker="s", color=GREEN, label="CUDA-Q (statevector)")
ax.plot(K, omp_E0,   marker="^", color=AMBER, label="OpenMP CPU (32T)")

# Highlight K=3 (intermediate)
for vals, color, marker in [
    (cuda_E0, BLUE, "o"),
    (cudaq_E0, GREEN, "s"),
    (omp_E0, AMBER, "^")
]:
    ax.scatter(3, vals[2],
               facecolors="none",
               edgecolors=color,
               s=80,
               linewidths=1.8,
               zorder=5)

# Annotate K=3 values
ax.annotate("3.50", (3, cuda_E0[2]),  xytext=(5, 8),
            textcoords="offset points", fontsize=9, color=BLUE)
ax.annotate("3.50", (3, cudaq_E0[2]), xytext=(-25, -18),
            textcoords="offset points", fontsize=9, color=GREEN)
ax.annotate("3.50", (3, omp_E0[2]),   xytext=(5, -18),
            textcoords="offset points", fontsize=9, color=AMBER)

# Annotate K=4 values
for val, color, dy in [
    (cuda_E0[3], BLUE, 10),
    (cudaq_E0[3], GREEN, -18),
    (omp_E0[3], AMBER, 10),
]:
    ax.annotate(f"{val:.2f}", (4, val),
                xytext=(-28, dy),
                textcoords="offset points",
                fontsize=9,
                color=color)

# Exact line
ax.axhline(E0_exact, color=AMBER, linewidth=1.4, alpha=0.6)

ax.text(4.1, E0_exact + 0.05,
        f"$E_0^{{exact}} = {E0_exact:.2f}$",
        fontsize=9,
        color=AMBER)

# Axes
ax.set_xlim(0.7, 4.6)
ax.set_ylim(2.6, 5.5)

ax.set_xticks(K)
ax.set_xticklabels(["K=1", "K=2", "K=3", "K=4"])

ax.set_xlabel("Krylov dimension $K$")
ax.set_ylabel("$E_0$ estimate")

ax.set_title(
    "Krylov Convergence\n"
    "$n_Q=15$, $J_X = J_Y = J_Z = 0.5$, Trotter steps = 4"
)

# Legend
ax.legend(fontsize=9, loc="upper right",
          framealpha=0.9, edgecolor="#cccccc")

plt.tight_layout()
plt.savefig("./TEST/krylov_convergence_clean.png", dpi=150)
plt.close()

print("Saved: ./TEST/krylov_convergence_clean.png")