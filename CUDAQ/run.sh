#!/bin/bash
#SBATCH --job-name=kqd_mgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=kqd_mgpu_%j.out
#SBATCH --error=kqd_mgpu_%j.err

# ─────────────────────────────────────────────────────────────────────
# Quantum Krylov Diagonalisation — CUDA-Q Single-GPU Run + Nsight Systems
# ─────────────────────────────────────────────────────────────────────

export PATH=/scratch/aadithyaiyer/cudaq_install/cudaq/bin:$PATH
export LD_LIBRARY_PATH=/scratch/aadithyaiyer/cudaq_install/cudaq/lib:$LD_LIBRARY_PATH

export PATH=/usr/lib64/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:$LD_LIBRARY_PATH

SRCDIR=/scratch/aadithyaiyer/FinalProject/CUDAQ
BINARY=$SRCDIR/kqd_single
INPUT=$SRCDIR/input.txt

# ── Nsight Systems output config ─────────────────────────────────────
PROFILE_DIR=$SRCDIR/profiles
mkdir -p "$PROFILE_DIR"
PROFILE_OUT=$PROFILE_DIR/kqd_nsys_${SLURM_JOB_ID}

echo "============================================"
echo "Job ID      : $SLURM_JOB_ID"
echo "Nodes       : $SLURM_NODELIST"
echo "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "nvq++ path  : $(which nvq++)"
echo "nsys path   : $(which nsys 2>/dev/null || echo 'NOT FOUND — check module load')"
echo "Start time  : $(date)"
echo "============================================"

# ── Compile targeting a single nvidia GPU ────────────────────────────
echo "Compiling with nvq++ --target nvidia ..."
nvq++ --target nvidia \
      -o "$BINARY" \
      "$SRCDIR/CUDAQ.cpp"

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed. Exiting."
    exit 1
fi
echo "Compilation successful."
echo "============================================"

# ── Run with Nsight Systems profiling ────────────────────────────────
cd "$SRCDIR"

echo "Running with nsys profile ..."
nsys profile \
    --trace=cuda,nvtx,osrt \
    --gpu-metrics-device=all \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    --stats=true \
    --output="$PROFILE_OUT" \
    "$BINARY"

EXIT_CODE=$?

echo ""
echo "============================================"
echo "Binary exit code : $EXIT_CODE"
echo "Profile report   : ${PROFILE_OUT}.nsys-rep"
echo "End time         : $(date)"
echo "============================================"
echo ""
echo "To view the profile, copy the .nsys-rep file to your local"
echo "machine and open with Nsight Systems GUI:"
echo "  scp <cluster>:${PROFILE_OUT}.nsys-rep ."

exit $EXIT_CODE