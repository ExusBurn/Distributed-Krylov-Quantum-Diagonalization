import numpy as np
import matplotlib.pyplot as plt
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
    "font.size":         11,
    "lines.linewidth":   2.2,
})

BLUE   = "#378ADD"
GREEN  = "#1D9E75"
AMBER  = "#EF9F27"
GRAY   = "#8b949e"

# ─────────────────────────────
# DATA
# ─────────────────────────────
gpu_data = {
    "COARSEN=1": ([1, 2, 3], [5832.2, 4020.1, 2096.2]),
    "COARSEN=2": ([1, 2, 3], [5949.8, 3791.2, 1958.4]),
    "COARSEN=4": ([1, 2, 3], [6758.0, 4499.9, 2341.2]),
}

cpu_data = {
    "16 threads/rank": ([1, 2, 3], [88133, 58954, 31498]),
    "32 threads/rank": ([1, 2, 3], [86825, 58461, 31403]),
    "64 threads/rank": ([1, 2, 3], [88824, 59488, 33427]),
}

colors  = [BLUE, GREEN, AMBER]
markers = ["o", "s", "^"]

# ─────────────────────────────
# GPU STRONG SCALING
# ─────────────────────────────
fig, ax = plt.subplots(figsize=(5.2, 4.6))

for (label, (x, t_ms)), c, m in zip(gpu_data.items(), colors, markers):
    t_s = [v/1000 for v in t_ms]

    ax.plot(x, t_s, marker=m, color=c, label=label)

    # annotate ONLY COARSEN=1 and COARSEN=4
    if label in ["COARSEN=1", "COARSEN=4"]:
        ax.annotate(f"{t_s[-1]:.2f}s",
                    xy=(x[-1], t_s[-1]),
                    xytext=(8, 0),
                    textcoords="offset points",
                    fontsize=10,
                    fontweight="bold",
                    color="black",
                    va="center")

# ideal
t1 = gpu_data["COARSEN=1"][1][0] / 1000
ax.plot([1,2,3], t1/np.array([1,2,3]),
        linestyle="--", color=GRAY, label="Ideal")

ax.set_xticks([1,2,3])
ax.set_xticklabels(["1 GPU","2 GPUs","3 GPUs"])
ax.set_xlabel("Number of GPUs")
ax.set_ylabel("Wall time (s)")
ax.set_title("GPU Strong Scaling\nnQ=23, J=0.5")

ax.set_xlim(0.9,3.3)
ax.set_ylim(1.5,7)

ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("./TEST/gpu_strong_scaling.png", dpi=150)
plt.close()

print("Saved gpu_strong_scaling.png")

# ─────────────────────────────
# CPU STRONG SCALING
# ─────────────────────────────
fig, ax = plt.subplots(figsize=(5.2, 4.6))

for (label, (x, t_ms)), c, m in zip(cpu_data.items(), colors, markers):
    t_s = [v/1000 for v in t_ms]

    ax.plot(x, t_s, marker=m, color=c, label=label)

    # annotate ONLY 32 threads
    if label == "32 threads/rank":
        ax.annotate(f"{t_s[-1]:.2f}s",
                    xy=(x[-1], t_s[-1]),
                    xytext=(8, 0),
                    textcoords="offset points",
                    fontsize=10,
                    fontweight="bold",
                    color="black",
                    va="center")

# ideal
t1 = cpu_data["32 threads/rank"][1][0] / 1000
ax.plot([1,2,3], t1/np.array([1,2,3]),
        linestyle="--", color=GRAY, label="Ideal")

ax.set_xticks([1,2,3])
ax.set_xticklabels(["1 rank","2 ranks","3 ranks"])
ax.set_xlabel("MPI ranks")
ax.set_ylabel("Wall time (s)")
ax.set_title("CPU Strong Scaling\nnQ=23, J=0.5")

ax.set_xlim(0.9,3.3)
ax.set_ylim(25,95)

ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("./TEST/cpu_strong_scaling.png", dpi=150)
plt.close()

print("Saved cpu_strong_scaling.png")

# save script
shutil.copy(__file__, "./TEST/strong_scaling_final_filtered.py")
print("Saved strong_scaling_final_filtered.py")