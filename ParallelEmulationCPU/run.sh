#!/bin/bash
# =============================================================================
#  run.sh  —  Strong-scaling experiment for kqd_openmp (dynamic scheduling)
#
#  Layout
#  ------
#    3 MPI ranks, one per node (3 nodes total).
#    Each rank parallelises its statevector sweeps with OpenMP.
#    We sweep OMP_NUM_THREADS ∈ {2, 4, 8, 16, 32, 64} while holding the
#    problem size fixed (nQ from input.txt), so this is a pure strong-scaling
#    study.
#
#  What gets measured
#  ------------------
#    Wall time reported by each rank from MPI_Barrier → compute → wall-clock.
#    The bottleneck rank is always rank 0 (k=3, most Trotter steps).
#    You'll see how that time shrinks as you add threads.
#
#  Comparing to GPU
#  ----------------
#    The GPU version (linebyline.cu) used 1 H100 per rank.
#    Each CUDA "thread" does ~1 amplitude pair.
#    Each OpenMP thread does ns/2 / OMP_NUM_THREADS amplitude pairs
#    in a scalar loop — useful for understanding memory-bandwidth vs compute.
#
#  Output files
#  ------------
#    Dynamic runs  → run_dynamic_Xt_nQY.txt
#    Static runs   → run_Xt_nQY.txt          (untouched by this script)
#
#  Usage
#  -----
#    sbatch run.sh
# =============================================================================

#SBATCH --job-name=kqd_omp_dynamic
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64          # reserve max CPUs; OMP_NUM_THREADS controls actual use
#SBATCH --time=04:00:00
#SBATCH --partition=debug
#SBATCH --output=kqd_omp_dynamic_%j.out
#SBATCH --error=kqd_omp_dynamic_%j.err

# ── Environment ───────────────────────────────────────────────────────────────
export PATH=/usr/lib64/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=/usr/include/openmpi-x86_64

SRCDIR=/scratch/aadithyaiyer/FinalProject/ParallelEmulationCPU
SOURCE=$SRCDIR/kqd_openmp.cpp
BINARY=$SRCDIR/kqd_openmp

# ── Scheduler label (keeps output files separate from static runs) ────────────
SCHED=dynamic

echo "============================================"
echo " Job ID  : ${SLURM_JOB_ID:-local}"
echo " Nodes   : ${SLURM_NODELIST:-localhost}"
echo " Sched   : $SCHED"
echo " Start   : $(date)"
echo "============================================"

cd "$SRCDIR"

# ── Build ─────────────────────────────────────────────────────────────────────
# -fopenmp          : enable OpenMP
# -O3 -march=native : full vectorisation (AVX-512 on modern Xeons / AVX2 elsewhere)
# -std=c++17        : structured bindings used in jacobi_eigh

MPI_COMPILE=$(mpicxx --showme:compile 2>/dev/null \
              | tr ' ' '\n' | grep -v '^-pthread$' | xargs)
MPI_LINK=$(mpicxx --showme:link 2>/dev/null \
           | tr ' ' '\n' | grep -v '^-pthread$' \
           | sed 's/^-Wl,\(.*\)/-Wl,\1/' | xargs)

echo ""
echo "[build] Compiling kqd_openmp.cpp ..."
mpicxx -std=c++17 -O3 -march=native -fopenmp \
       $MPI_COMPILE $MPI_LINK \
       "$SOURCE" -o "$BINARY" \
    && echo "[build] OK" \
    || { echo "[build] FAILED"; exit 1; }
echo ""

# ── Input ─────────────────────────────────────────────────────────────────────
# Only write input.txt if it does not already exist — preserves any existing
# static-run input so the two experiments stay comparable.
NQ=15
if [ ! -f "$SRCDIR/input.txt" ]; then
    echo "0.5 0.5 0.5 ${NQ}" > "$SRCDIR/input.txt"
    echo "[input] Written: JX=0.5 JY=0.5 JZ=0.5 nQ=${NQ}"
else
    echo "[input] Using existing input.txt: $(cat $SRCDIR/input.txt)"
fi
echo ""

# ── Strong-scaling sweep ──────────────────────────────────────────────────────
for NTHREADS in 2 4 8 16 32 64; do
    echo "============================================"
    echo " OMP_NUM_THREADS = $NTHREADS  (nQ = ${NQ})  sched = $SCHED"
    echo "============================================"

    export OMP_NUM_THREADS=$NTHREADS
    export OMP_PROC_BIND=close   # keep threads on the same socket / NUMA domain
    export OMP_PLACES=cores      # bind to physical cores (not hardware threads)

    srun --mpi=pmix -n 3 \
         --cpus-per-task=$NTHREADS \
         "$BINARY" 2>&1 | tee "$SRCDIR/run_${SCHED}_${NTHREADS}t_nQ${NQ}.txt"

    echo ""
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo "============================================"
echo " Scaling logs: run_${SCHED}_*t_nQ${NQ}.txt"
echo ""
echo " INTERPRETATION GUIDE"
echo " ---------------------"
echo " Ideal strong scaling: 2x threads → 0.5x wall time."
echo " In practice, you will see:"
echo "   32→64 threads:  near-ideal if nQ is large (DRAM BW split between cores)"
echo "   64→128 threads: diminishing returns (NUMA boundary, BW saturation)"
echo ""
echo " dynamic vs static scheduling:"
echo "   For uniform gate sweeps (identical work per iteration), static is"
echo "   usually faster — no work-stealing queue overhead."
echo "   dynamic may win only if thread launch times vary significantly"
echo "   (e.g. NUMA imbalance, OS jitter at high thread counts)."
echo ""
echo " Bottleneck rank is always rank 0 (k=3, 3x Trotter steps)."
echo " Ranks 1 and 2 will show shorter times — same as GPU MPI_Barrier imbalance."
echo ""
echo " CPU vs GPU quick comparison:"
echo "   GPU H100 HBM2e BW  : ~3.35 TB/s across ~67 billion threads/s throughput"
echo "   CPU dual-socket BW  : ~300-500 GB/s  (DRAM), ~2-4 TB/s (L3 aggregate)"
echo "   For nQ=10 (fits in L3): CPU may be competitive."
echo "   For nQ>=20 (spills to DRAM): GPU wins by ~5-10x on BW alone."
echo ""
echo " End: $(date)"
echo "============================================"