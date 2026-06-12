#!/bin/bash
#SBATCH --job-name=kqd_sweep
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --partition=debug
#SBATCH --output=kqd_sweep_%j.out
#SBATCH --error=kqd_sweep_%j.err

export PATH=/usr/lib64/openmpi/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=/usr/include/openmpi-x86_64

SRCDIR=/scratch/aadithyaiyer/FinalProject/ParallelEmulation
SOURCE=$SRCDIR/linebyline.cu
BINARY=$SRCDIR/kqd_sweep

echo "============================================"
echo " Job ID : $SLURM_JOB_ID"
echo " Nodes  : $SLURM_NODELIST"
echo " GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo " Start  : $(date)"
echo "============================================"

cd $SRCDIR

#  Build
MPI_COMPILE=$(mpicxx --showme:compile | tr ' ' '\n' | grep -v '^-pthread$' | xargs)
MPI_LINK=$(mpicxx --showme:link \
    | tr ' ' '\n' \
    | grep -v '^-pthread$' \
    | sed 's/^-Wl,\(.*\)/-Xlinker \1/' \
    | xargs)

echo "[build] Compiling linebyline.cu..."
nvcc -std=c++17 -O3 -arch=sm_86 \
     -Xcompiler "-fopenmp,-pthread" \
     $MPI_COMPILE $MPI_LINK \
     ${SOURCE} -o ${BINARY} \
  && echo "[build] OK" \
  || { echo "[build] FAILED"; exit 1; }

#  Results file
RESULTS=$SRCDIR/sweep_results.csv
echo "nQ,JX,JY,JZ,E0_exact,E0_kqd,abs_err,wall_ms" > $RESULTS

#  Sweep
# for nQ in 10; do
#     for J in 1; do

#         echo ""
#         echo "─────────────────────────────────────────"
#         echo " nQ=$nQ  JX=JY=JZ=$J"
#         echo "─────────────────────────────────────────"

#         echo "$J $J*3 $J $nQ" > $SRCDIR/input.txt

#         START_MS=$(($(date +%s%N) / 1000000))

#         srun --mpi=pmix -n 3 ${BINARY}

#         END_MS=$(($(date +%s%N) / 1000000))
#         WALL_MS=$((END_MS - START_MS))

#         # Pull E0_exact and E0_kqd (K=4 line) from qkd_results.txt
#         E0_exact=$(grep "Classical exact E0" $SRCDIR/qkd_results.txt \
#                    | awk '{print $NF}')
#         E0_kqd=$(grep "K=4" $SRCDIR/qkd_results.txt \
#                  | tail -1 | awk '{print $2}' | sed 's/E0=//')
#         ABS_ERR=$(grep "K=4" $SRCDIR/qkd_results.txt \
#                   | tail -1 | awk '{print $3}' | sed 's/|err|=//')

#         echo "$nQ,$J,$J,$J,$E0_exact,$E0_kqd,$ABS_ERR,$WALL_MS" >> $RESULTS

#         echo " wall time: ${WALL_MS} ms"
#         echo " E0_exact=$E0_exact  E0_kqd=$E0_kqd  |err|=$ABS_ERR"

#     done
# done

#  Single Run Configuration
nQ=16
JX=0.5
JY=0.5
JZ=0.5

echo ""
echo "─────────────────────────────────────────"
echo " nQ=$nQ  JX=$JX  JY=$JY  JZ=$JZ"
echo "─────────────────────────────────────────"

echo "$JX $JY $JZ $nQ" > $SRCDIR/input.txt

START_MS=$(($(date +%s%N) / 1000000))

mkdir -p $SRCDIR/nsys_nQ16

srun --mpi=pmix -n 3 nsys profile \
    --output=$SRCDIR/nsys_nQ16/kqd_rank_%q{SLURM_PROCID} \
    --trace=cuda,mpi \
    ${BINARY}

END_MS=$(($(date +%s%N) / 1000000))
WALL_MS=$((END_MS - START_MS))

# Extract results
E0_exact=$(grep "Classical exact E0" $SRCDIR/qkd_results.txt | awk '{print $NF}')
E0_kqd=$(grep "K=4" $SRCDIR/qkd_results.txt | tail -1 | awk '{print $2}' | sed 's/E0=//')
ABS_ERR=$(grep "K=4" $SRCDIR/qkd_results.txt | tail -1 | awk '{print $3}' | sed 's/|err|=//')

echo "nQ,JX,JY,JZ,E0_exact,E0_kqd,abs_err,wall_ms" > $RESULTS
echo "$nQ,$JX,$JY,$JZ,$E0_exact,$E0_kqd,$ABS_ERR,$WALL_MS" >> $RESULTS

echo " wall time: ${WALL_MS} ms"
echo " E0_exact=$E0_exact  E0_kqd=$E0_kqd  |err|=$ABS_ERR"

# Summary
echo ""
echo "============================================"
echo " SWEEP COMPLETE"
echo " Results written to: $RESULTS"
echo ""
echo " CSV contents:"
cat $RESULTS
echo "============================================"
echo " End: $(date)"
echo "============================================"
