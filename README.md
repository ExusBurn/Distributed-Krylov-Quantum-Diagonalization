# Distributed Krylov Quantum Diagonalization

A distributed, GPU-accelerated statevector simulator for **Krylov Quantum Diagonalization (KQD)** applied to the Heisenberg XXX spin chain. This project implements and benchmarks three backends — CUDA/MPI, CUDA-Q, and OpenMP/MPI — achieving a **6.2× speedup** over the CUDA-Q baseline at 23 qubits.

*Final project for Parallel Programming, Department of Computational and Data Sciences, IISc Bangalore.*

---

## Background

The time-independent Schrödinger equation Ĥψ = Eψ governs many-body quantum systems. For a system of just 20 atoms, the Hamiltonian is a 2²⁰ × 2²⁰ matrix — making exact diagonalization classically infeasible.

**Krylov Quantum Diagonalization** sidesteps this by building a compact Krylov subspace via time-evolved quantum states |e^{−iHkt}ψ₀⟩, extracting matrix elements through the Efficient Hadamard Test, and solving the resulting Generalized Eigenvalue Problem (GEVP) classically:

$$\tilde{H}\vec{c} = E\tilde{S}\vec{c}$$

Because the Krylov matrices are Toeplitz (due to Trotterized time evolution), only the first row of the projected Hamiltonian H̃ and overlap matrix S̃ need to be computed:

$$S(k) = \langle\psi_0|U^k|\psi_0\rangle, \quad H(k) = \langle\psi_0|HU^k|\psi_0\rangle$$

---

## Repository Structure

```
.
├── CUDAQ.cpp          # CUDA-Q implementation (baseline)
├── linebyline.cu      # Hand-written CUDA/MPI implementation
├── kqd_openmp.cpp     # Structurally identical OpenMP/MPI CPU port
```

---

## Implementations

### 1. `CUDAQ.cpp` — CUDA-Q Baseline

Uses NVIDIA's [CUDA-Q](https://github.com/NVIDIA/cuda-quantum) C++ framework (`cudaq::observe`) for high-level quantum circuit simulation. This serves as the correctness reference and performance baseline.

**Key features:**
- High-level gate application via CUDA-Q primitives
- Efficient Hadamard Test for Re/Im extraction of S(k) and H(k)
- Jacobi iterative diagonalization for the GEVP solve

### 2. `linebyline.cu` — CUDA/MPI (Primary Implementation)

Hand-written CUDA kernels operating directly on the statevector array, bypassing CUDA-Q abstractions. Three-way MPI task parallelism distributes Krylov vectors k=1,2,3 across 3 GPU ranks (emulating IBM Kingston, Fez, Marrakesh).

**Kernel suite:**
- **Single-qubit gates** (`apply_x_gate`, `_y`, `_z`, `_hadamard`, `_s`, `_sdg`, `_cx`, `_rz`): bit-arithmetic paired-amplitude indexing from qHiPSTER; 256 threads/block with compile-time coarsening factor C ∈ {1, 2, 4}
- **Fused 4×4 two-qubit gates** (`apply_rxx`, `apply_ryy`, `apply_rzz`): single kernel per gate instead of ~10 decomposed gates; three-segment index decomposition (inner/mid/outer) selects the four coupled amplitudes
- **Shared-memory tree reduction** (`reduce_x_expectation`): two-stage reduction with 256-element `__shared__` array and single `atomicAdd` per block, computing ⟨X_anc⟩ = P(anc=0) − P(anc=1)
- **Ancilla-sector helpers** (`extract_odd_sector`, `insert_odd_sector`): compact the full 2^{nQ+1} statevector to the system-only buffer during Trotter, halving DRAM traffic

**Key optimisations:**
- **Statevector reuse**: Trotter evolution is run once per rank; `cudaMemcpyDeviceToDevice` branches the result into four measurement buffers (sv_s, sv_s_im, sv_re, sv_im), eliminating 3 of every 4 Trotter sweeps required by the IBM circuit-based reference
- **Second-order Suzuki-Trotter**: even steps apply RXX→RYY→RZZ over even-then-odd bonds; odd steps reverse both bond order and gate sequence (RZZ→RYY→RXX), cancelling O(δt²) errors at no extra kernel cost
- **Compile-time coarsening**: C=2 gives optimal throughput-per-watt by raising arithmetic intensity without over-serialising threads

### 3. `kqd_openmp.cpp` — OpenMP/MPI CPU Port

Mirrors every CUDA kernel structurally with identical bit-arithmetic index formulas. Uses `#pragma omp parallel for` with benchmarked `static` vs `dynamic` scheduling.

**Key findings:**
- Static scheduling gives sequential DRAM access and hardware prefetching — dynamic causes cache-line thrashing across the 2^{nQ} statevector
- OpenMP `reduction(+:val)` is the CPU equivalent of shared-memory tree + `atomicAdd`, expressed in one directive
- C++ move semantics on `std::vector<double2>` give zero-copy statevector branching, the CPU equivalent of `cudaMemcpyDeviceToDevice`

---

## Algorithm: Efficient Hadamard Test

For each k ∈ {1, 2, 3}, the pipeline is:

```
PREP  →  U(k·dt) via Trotter  →  UNPREP(Pauli)  →  ⟨X_anc⟩
```

1. **PREP**: Hadamard on ancilla q₀, then CX(q₀ → q_mid) to entangle ancilla with system
2. **U(t)**: Second-order Suzuki-Trotter with alternating forward/reverse bond sweeps (RXX, RYY, RZZ)
3. **UNPREP**: CX reverse, then basis change (H or S†+H on ancilla) to extract Re or Im
4. **Measure ⟨X_anc⟩**: Global reduction over the statevector

Since each Krylov vector only depends on |ψ₀⟩, the k=1,2,3 computations are fully independent — ideal for MPI task parallelism with a single `MPI_Gather` of 12 doubles at the end.

The time step is set as dt = π / ‖H‖, where ‖H‖ = max(|λ_min|, |λ_max|) from the spectrum of the Heisenberg chain.

---

## Results

**Setup**: JX = JY = JZ = 0.5, Krylov dim K = 4, Trotter steps = 4. GPU: NVIDIA RTX A5000 ×3 with MPI. CPU: 32 OpenMP threads/rank.

| Backend | nQ = 18 | nQ = 23 |
|---|---|---|
| OMP Dynamic | baseline (slowest) | baseline |
| CUDA-Q | ~5× faster than OMP Dynamic | — |
| OMP Static | ~5× faster than OMP Dynamic | similar to CUDA-Q |
| **CUDA/MPI** | **2.8×** over CUDA-Q | **6.2×** over CUDA-Q |

**Strong scaling (nQ = 23):**
- GPU: ~6s (1 GPU) → **2.10s (3 GPUs, COARSEN=2)** — near-ideal scaling
- CPU: ~88s (1 rank) → **31.40s (3 ranks)** — near-ideal scaling

**Per-kernel profiling (Nsight Systems)**: `reduce_x_expect` is the single most expensive kernel (~1.13ms/rank), justifying the shared-memory tree design. Load is balanced near-perfectly across all three ranks.

**Convergence**: All three backends produce identical E₀ estimates at every K, converging from 5.0 (K=1) toward E₀^exact ≈ 3.04, validating correctness of the hand-written kernels.

---

## Build & Run

### CUDA/MPI

```bash
nvcc -O3 -arch=sm_80 linebyline.cu -o kqd_cuda \
     -Xcompiler "-fopenmp" -lmpi -DCOARSEN=2

mpirun -np 3 ./kqd_cuda <nqubits> <jx> <jy> <jz> <krylov_dim> <trotter_steps>
# Example:
mpirun -np 3 ./kqd_cuda 20 0.5 0.5 0.5 4 4
```

Profile with Nsight Systems:
```bash
mpirun -np 3 nsys profile --trace=cuda,mpi -o rank%q{OMPI_COMM_WORLD_RANK} ./kqd_cuda 16 0.5 0.5 0.5 4 4
```

### CUDA-Q

```bash
nvq++ -O3 CUDAQ.cpp -o kqd_cudaq
./kqd_cudaq <nqubits> <jx> <jy> <jz> <krylov_dim> <trotter_steps>
```

### OpenMP/MPI

```bash
mpicxx -O3 -fopenmp kqd_openmp.cpp -o kqd_omp
OMP_NUM_THREADS=32 mpirun -np 3 ./kqd_omp <nqubits> <jx> <jy> <jz> <krylov_dim> <trotter_steps>
```

---

## Dependencies

- CUDA ≥ 11.0 (sm_80 for A5000/RTX 3000+)
- MPI (OpenMPI or MPICH)
- NVIDIA CUDA-Q (for `CUDAQ.cpp` only)
- C++17-compatible compiler

---

## Future Directions

- **Coalesced statevector layout**: restructure amplitudes so that those coupled by two-qubit gates are contiguous in memory, improving warp-level access for RXX/RYY/RZZ
- **Stream-parallel expectation values**: assign one CUDA stream per Pauli term in H for fully overlapped execution within a single pass
- **Real QPU deployment**: treat IBM Fez, Kingston, and Marrakesh each as an MPI-equivalent rank (pending public inter-QPU API availability)
- **Scale beyond 30+ qubits**: NVLink-enabled statevector sharding, extending Krylov dimension K > 4 for strongly correlated systems

---

## References

1. E. N. Epperly, L. Lin, Y. Nakatsukasa, "A theory of quantum subspace diagonalization," *SIAM J. Matrix Anal. Appl.*, 2022.
2. IBM Quantum, "Krylov Quantum Diagonalization," Quantum Learning Course, 2024.
3. R. Babbush et al., "Diagonalization of large many-body Hamiltonians on a quantum processor," *Nature*, 2024.
4. M. Smelyanskiy, N. Sawaya, A. Aspuru-Guzik, "qHiPSTER: The Quantum High Performance Software Testing Environment," 2016.
5. NVIDIA, "CUDA-Q: The platform for hybrid quantum-classical computing," 2024.
6. M. Suzuki, "General theory of fractal path integrals," *J. Math. Phys.*, 1991.
