import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os, shutil

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
    "text.color":        "#000000",
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "#cccccc",
    "legend.labelcolor": "#000000",
    "font.family":       "sans-serif",
    "font.size":         10,
    "lines.linewidth":   2.0,
})

BLUE   = "#378ADD"
GREEN  = "#1D9E75"
AMBER  = "#EF9F27"
GRAY   = "#8b949e"
RED    = "#E24B4A"
PINK   = "#D4537E"

# ═══════════════════════════════════════════════════════════════════════
# PLOT 1 — Krylov convergence
# ═══════════════════════════════════════════════════════════════════════

E0_exact = 3.04

cuda_E0  = [5.00000000, 3.62340579, 3.5021, 3.45849098]
cudaq_E0 = [4.99999467, 3.62348811, 3.5038, 3.46321394]
omp_E0   = [5.00000000, 3.62340579, 3.5021, 3.45849098]

fig, ax = plt.subplots(figsize=(4.5,4.5))

configs = [
    (cuda_E0,  BLUE,  "o", "CUDA  (MPI 3-rank, RTX A5000)"),
    (cudaq_E0, GREEN, "s", "CUDA-Q  (statevector sim)"),
    (omp_E0,   AMBER, "^", "OpenMP CPU  (32T static)"),
]

for E0_vals, color, marker, label in configs:
    ax.plot([1, 2, 3, 4], E0_vals, color=color, linewidth=2.0, zorder=3)
    ax.scatter([1, 2, 4], [E0_vals[0], E0_vals[1], E0_vals[3]],
               color=color, marker=marker, s=60, zorder=5)

    k3_val = E0_vals[2]
    ax.scatter([3], [k3_val], facecolors="none", edgecolors=color,
               marker=marker, s=70, linewidths=1.8, zorder=5)

    ax.annotate(f"{k3_val:.2f}", xy=(3, k3_val),
                xytext=(7, 4), textcoords="offset points",
                fontsize=7.5, color=color)

    ax.plot([], [], color=color, marker=marker,
            linestyle="-", label=label)

ax.axhline(E0_exact, color=AMBER, linestyle="-",
           linewidth=1.4, alpha=0.55,
           label=f"Exact  E$_0$ = {E0_exact:.2f}")

ax.text(4.35, E0_exact + 0.06,
        f"E$_0^{{\\rm exact}}$ = {E0_exact:.2f}",
        fontsize=8, color=AMBER, alpha=0.85)

# annotate K=4 values
for col, val, (dx, dy) in [
    (BLUE,  cuda_E0[3],  (-38, +12)),
    (GREEN, cudaq_E0[3], (-38, -18)),
    (AMBER, omp_E0[3],   (-38, +12))
]:
    ax.annotate(f"{val:.2f}", xy=(4, val),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8, color=col)

ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(["K=1", "K=2", "K=3", "K=4"])
ax.set_xlim(0.6, 4.9)
ax.set_ylim(2.6, 5.5)

ax.set_ylabel("E$_0$ estimate")
ax.set_xlabel("Krylov dimension K")

ax.set_title(
    "Krylov convergence\n"
    "nQ=15, $J_X = J_Y = J_Z = 0.5$, Trotter steps=4"
)

ax.legend(fontsize=9, loc="upper right")

fig.tight_layout()
fig.savefig("./TEST/plot1_krylov_convergence.png",
            dpi=150, bbox_inches="tight")
plt.close()

print("Saved plot1_krylov_convergence.png")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 2 — Timing + Speedup
# ═══════════════════════════════════════════════════════════════════════

nQ_s  = [18, 19, 20, 21, 22, 23]
t_s   = [170, 405, 2241, 5641, 12920, 29261]

nQ_d  = [18, 19, 20, 21, 22, 23]
t_d   = [12734, 27105, 57975, 133983, 244372, 519225]

nQ_c  = [18, 19, 20, 21, 22, 23, 24, 25, 26]
t_c   = [58, 201, 253, 506, 956, 1692, 3234, 6096, 13328]

nQ_q  = [18, 19, 20, 21, 22, 23]
t_q   = [165, 655, 1372, 2478, 5039, 10471]

def ms2s(lst):
    return [v/1000.0 for v in lst]

# compute speedup CUDA-Q / CUDAMPI
speedup = {}
for nq, tq in zip(nQ_q, t_q):
    if nq in nQ_c:
        tc = t_c[nQ_c.index(nq)]
        speedup[nq] = tq / tc

fig, ax = plt.subplots(figsize=(4.5,4.5))

ax.semilogy(nQ_s, ms2s(t_s), color=BLUE,  marker="s",
            linewidth=2.2, label="OMP Static")

ax.semilogy(nQ_d, ms2s(t_d), color=RED,   marker="o",
            linewidth=2.2, label="OMP Dynamic")

ax.semilogy(nQ_c, ms2s(t_c), color=GREEN, marker="^",
            linewidth=2.2, label="CUDAMPI")

ax.semilogy(nQ_q, ms2s(t_q), color=PINK,  marker="D",
            linewidth=2.2, label="CUDA-Q")

# annotate speedups
for nq in speedup:
    tc_sec = ms2s([t_c[nQ_c.index(nq)]])[0]
    sp = speedup[nq]

    offset = -12 if nq % 2 == 0 else -18
    
    ax.annotate(f"{sp:.1f}×",
                xy=(nq, tc_sec),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                fontweight="bold",
                color="black")  

ax.set_xticks(list(range(18, 27)))
ax.set_xlim(17.5, 26.5)

ax.set_xlabel("Number of Qubits (nQ)")
ax.set_ylabel("Wall Time (log scale)")

ax.set_title(
    "nQ Sweep — Wall Time Comparison\n"
    "$J_X = J_Y = J_Z = 0.5$, Krylov dim=4, Trotter steps=4"
)

ax.yaxis.set_major_formatter(
    ticker.FuncFormatter(
        lambda v, _:
        f"{v*1000:.0f} ms" if v < 1 else
        f"{v:.0f} s" if v < 60 else
        f"{v/60:.1f} min"
    )
)

ax.legend(fontsize=7, loc="lower right",
          framealpha=0.2, edgecolor="#cccccc",
          borderpad=0.3, labelspacing=0.3,
          handlelength=1.5)

fig.tight_layout()
fig.savefig("./TEST/plot2_timing_comparison.png",
            dpi=150, bbox_inches="tight")
plt.close()

print("Saved plot2_timing_comparison.png")

shutil.copy(__file__, "./TEST/kqd_plots_final.py")
print("Saved kqd_plots_final.py")