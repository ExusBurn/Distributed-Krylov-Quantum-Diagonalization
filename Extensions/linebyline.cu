


#include <mpi.h>            // MPI: we split the Krylov indices k across ranks/GPUs.
#include <cuda_runtime.h>   // CUDA runtime API (cudaMalloc, launches, etc.).
#include <cuComplex.h>      // Not strictly needed now, but harmless; double2 is our amplitude type.

#include <iostream>         // std::cout for progress printing on rank 0.
#include <vector>           // std::vector everywhere for host-side matrices.
#include <complex>          // std::complex<double> for the small classical matrices.
#include <cmath>            // std::sqrt, std::cos, std::abs, std::log, ...
#include <algorithm>        // std::sort, std::max, std::min.
#include <fstream>          // results file output.
#include <iomanip>          // std::setprecision for readable energies.
#include <chrono>           // wall-clock timing of the quantum part.
#include <cstdlib>          // std::getenv for the optional RESULTS_FILE path.
#include <limits>           // infinity / NaN sentinels.
#include <string>           // std::string for paths and notes.
#include <lapacke.h>        // LAPACK C interface for zhegv (generalized Hermitian eigenproblem).
using Complex = std::complex<double>;                 // Short alias: one complex amplitude on the host.
using CMat    = std::vector<std::vector<Complex>>;    // Short alias: a dense complex matrix (small, host-side).
static const double PI = 3.14159265358979323846;      // pi, used for the anti-aliasing dt and Rz angles.
__device__ static constexpr double INV_SQRT2 = 0.7071067811865475; // 1/sqrt(2), used inside the Hadamard kernel.

// ----------------------------------------------------------------------------



// Thread Coursening Factor: For a specific quantum gate, something I was playing around with
#ifndef COARSEN
#define COARSEN 1            // Default: one logical item per thread.
#endif

static const int BLOCK = 256; // Threads per CUDA block. 256 is a safe, occupancy-friendly default.


// ----------------------------------------------------------------------------
//  reduce_x_expectation: computes <X> on the ancilla (qubit 0) by summing
//  (+prob) for even global indices and (-prob) for odd ones. Because qubit 0
//  is the least-significant bit, "index even" == "ancilla measured 0", so this
//  reduction returns  P(anc=0) - P(anc=1) = <Z_anc>  in the rotated basis used
//  by the Hadamard test -- i.e. the real/imag part of the overlap we want.
// ----------------------------------------------------------------------------
__global__ void reduce_x_expectation(const double2 *sv, long long ns, double *out) {
    __shared__ double sdata[BLOCK];                                   // Per-block scratch for the tree reduction.
    long long base = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * COARSEN; // First logical index for this thread.
    double val = 0.0;                                                 // Thread-local partial sum.

    #pragma unroll
    for (int c = 0; c < COARSEN; c++) {                               // Loop over this thread's COARSEN items.
        long long idx = base + c;                                     // Global amplitude index.
        if (idx < ns) {                                               // Guard against the tail past the array end.
            double re = sv[idx].x, im = sv[idx].y;                    // Real/imag parts of the amplitude.
            double prob = re*re + im*im;                              // |amplitude|^2 = probability of this basis state.
            val += (idx % 2 == 0) ? prob : -prob;                    // + if ancilla bit 0 == 0, - if == 1.
        }
    }

    sdata[threadIdx.x] = val;                                         // Stage each thread's partial into shared memory.
    __syncthreads();                                                  // Make all partials visible before reducing.
    for (int s = BLOCK/2; s > 0; s >>= 1) {                           // Standard binary tree reduction.
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();                                              // Sync between reduction levels.
    }
    if (threadIdx.x == 0) atomicAdd(out, sdata[0]);                   // One atomic per block into the global accumulator.
}


// ----------------------------------------------------------------------------
//  solve_gen_eig: solve the generalized Hermitian eigenproblem  H c = E S c
//  using LAPACK's zhegv (via the LAPACKE C interface).
//  Returns the smallest eigenvalue, or NaN if LAPACK reports failure.
// ----------------------------------------------------------------------------
static double solve_gen_eig(const CMat &H, const CMat &S) {
    int n = (int)H.size();
    if (n == 1) return H[0][0].real();

    // Flatten to row-major arrays; std::complex<double> and lapack_complex_double
    // are layout-compatible (both are two consecutive doubles).
    std::vector<lapack_complex_double> a(n * n), b(n * n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            a[i*n+j] = {H[i][j].real(), H[i][j].imag()};
            b[i*n+j] = {S[i][j].real(), S[i][j].imag()};
        }

    // Eigendecompose S; after the call b holds eigenvectors as columns, sigma ascending.
    std::vector<double> sigma(n);
    if (LAPACKE_zheev(LAPACK_ROW_MAJOR, 'V', 'U', n, b.data(), n, sigma.data()) != 0)
        return std::numeric_limits<double>::quiet_NaN();

    // good_vecs: rows are eigenvectors of S with eigenvalue > 1e-2.
    // Reinterpret b as Complex* — both are two consecutive doubles, layout-compatible.
    const Complex *V = reinterpret_cast<const Complex*>(b.data());
    CMat good_vecs;
    for (int j = 0; j < n; j++)
        if (sigma[j] > 1e-2) {
            std::vector<Complex> v(n);
            for (int i = 0; i < n; i++) v[i] = V[i*n + j];
            good_vecs.push_back(v);
        }
    int m = (int)good_vecs.size();
    if (m == 0) return std::numeric_limits<double>::quiet_NaN();

    // h_tilde = good_vecs.conj() @ H @ good_vecs.T  (m x m)
    // s_tilde = good_vecs.conj() @ S @ good_vecs.T  (m x m)
    CMat h_tilde(m, std::vector<Complex>(m, 0));
    CMat s_tilde(m, std::vector<Complex>(m, 0));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++) {
                    h_tilde[i][j] += std::conj(good_vecs[i][r]) * H[r][c] * good_vecs[j][c];
                    s_tilde[i][j] += std::conj(good_vecs[i][r]) * S[r][c] * good_vecs[j][c];
                }

    // Solve generalized eigenproblem h_tilde c = E s_tilde c.
    std::vector<lapack_complex_double> hf(m*m), sf(m*m);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++) {
            hf[i*m+j] = {h_tilde[i][j].real(), h_tilde[i][j].imag()};
            sf[i*m+j] = {s_tilde[i][j].real(), s_tilde[i][j].imag()};
        }
    std::vector<double> w(m);
    if (LAPACKE_zhegv(LAPACK_ROW_MAJOR, 1, 'N', 'U', m, hf.data(), m, sf.data(), m, w.data()) != 0)
        return std::numeric_limits<double>::quiet_NaN();
    return w[0];
}


// ----------------------------------------------------------------------------
//  Single-qubit gate kernels (coarsened). All use the standard bit-surgery:
//  split the global index into (hi, lo) around qubit q, form the paired
//  indices i0 (q=0) and i1 (q=1), and apply the 2x2 gate to (sv[i0], sv[i1]).
//  These were verified against textbook matrices and are correct.
// ----------------------------------------------------------------------------

// X gate: swap the q=0 and q=1 amplitudes.
__global__ void apply_x_gate(double2 *sv, long long ns, int q) {
    long long base = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * COARSEN; // First pair index for this thread.
    long long half = ns >> 1;                                       // There are ns/2 amplitude pairs.
    #pragma unroll
    for (int c = 0; c < COARSEN; c++) {
        long long tid = base + c;                                   // Logical pair index.
        if (tid >= half) return;                                    // Tail guard.
        long long lo = tid & ((1LL << q) - 1);                      // Bits below q.
        long long hi = tid >> q;                                    // Bits at/above q (excluding q itself).
        long long i0 = (hi << (q + 1)) | lo;                        // Index with qubit q = 0.
        long long i1 = i0 | (1LL << q);                             // Index with qubit q = 1.
        double2 tmp = sv[i0]; sv[i0] = sv[i1]; sv[i1] = tmp;        // Swap the two amplitudes (X action).
    }
}

// Hadamard gate on qubit q.
__global__ void apply_hadamard(double2 *sv, long long ns, int q) {
    long long base = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * COARSEN;
    long long half = ns >> 1;
    #pragma unroll
    for (int c = 0; c < COARSEN; c++) {
        long long tid = base + c;
        if (tid >= half) return;
        long long lo = tid & ((1LL << q) - 1);
        long long hi = tid >> q;
        long long i0 = (hi << (q + 1)) | lo;                        // |...0...>
        long long i1 = i0 | (1LL << q);                             // |...1...>
        double2 v0 = sv[i0], v1 = sv[i1];                           // Read both amplitudes.
        sv[i0] = make_double2((v0.x + v1.x) * INV_SQRT2, (v0.y + v1.y) * INV_SQRT2); // (a+b)/sqrt2
        sv[i1] = make_double2((v0.x - v1.x) * INV_SQRT2, (v0.y - v1.y) * INV_SQRT2); // (a-b)/sqrt2
    }
}

// CNOT with control ctrl, target tgt.
__global__ void apply_cx(double2 *sv, long long ns, int ctrl, int tgt) {
    long long base = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * COARSEN;
    long long half = ns >> 1;
    #pragma unroll
    for (int c = 0; c < COARSEN; c++) {
        long long tid = base + c;
        if (tid >= half) return;
        long long lo = tid & ((1LL << tgt) - 1);                    // Bit surgery is around the TARGET qubit.
        long long hi = tid >> tgt;
        long long i0 = (hi << (tgt + 1)) | lo;                      // target = 0
        long long i1 = i0 | (1LL << tgt);                           // target = 1
        if (!(i0 & (1LL << ctrl))) continue;                        // Only flip when the control bit is 1.
        double2 tmp = sv[i0]; sv[i0] = sv[i1]; sv[i1] = tmp;        // Swap target amplitudes (controlled X).
    }
}

// S gate (phase +i on |1>).
__global__ void apply_s_gate(double2 *sv, long long ns, int q) {
    long long base = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * COARSEN;
    long long half = ns >> 1;
    #pragma unroll
    for (int c = 0; c < COARSEN; c++) {
        long long tid = base + c;
        if (tid >= half) return;
        long long lo = tid & ((1LL << q) - 1);
        long long hi = tid >> q;
        long long i1 = ((hi << (q + 1)) | lo) | (1LL << q);         // Only the |q=1> amplitude is touched.
        double2 v1 = sv[i1];
        sv[i1] = make_double2(-v1.y, v1.x);                         // multiply by i: (x+iy) -> (-y + i x).
    }
}

// S-dagger gate (phase -i on |1>); used to read the IMAGINARY part in the Hadamard test.
__global__ void apply_sdg_gate(double2 *sv, long long ns, int q) {
    long long base = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * COARSEN;
    long long half = ns >> 1;
    #pragma unroll
    for (int c = 0; c < COARSEN; c++) {
        long long tid = base + c;
        if (tid >= half) return;
        long long lo = tid & ((1LL << q) - 1);
        long long hi = tid >> q;
        long long i1 = ((hi << (q + 1)) | lo) | (1LL << q);
        double2 v1 = sv[i1];
        sv[i1] = make_double2(v1.y, -v1.x);                        // multiply by -i: (x+iy) -> (y - i x).
    }
}

// Y gate.
__global__ void apply_y_gate(double2 *sv, long long ns, int q) {
    long long base = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * COARSEN;
    long long half = ns >> 1;
    #pragma unroll
    for (int c = 0; c < COARSEN; c++) {
        long long tid = base + c;
        if (tid >= half) return;
        long long lo = tid & ((1LL << q) - 1);
        long long hi = tid >> q;
        long long i0 = (hi << (q + 1)) | lo;
        long long i1 = i0 | (1LL << q);
        double2 v0 = sv[i0], v1 = sv[i1];
        sv[i0] = make_double2(v1.y, -v1.x);                        // Y|0..> component: -i * (q=1 amplitude).
        sv[i1] = make_double2(-v0.y, v0.x);                        // Y|1..> component: +i * (q=0 amplitude).
    }
}

// Z gate (sign flip on |1>).
__global__ void apply_z_gate(double2 *sv, long long ns, int q) {
    long long base = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * COARSEN;
    long long half = ns >> 1;
    #pragma unroll
    for (int c = 0; c < COARSEN; c++) {
        long long tid = base + c;
        if (tid >= half) return;
        long long lo = tid & ((1LL << q) - 1);
        long long hi = tid >> q;
        long long i1 = ((hi << (q + 1)) | lo) | (1LL << q);
        sv[i1].x = -sv[i1].x;                                      // Negate real part of |q=1>.
        sv[i1].y = -sv[i1].y;                                      // Negate imag part of |q=1>.
    }
}

// Rz(theta) = diag(e^{-i theta/2}, e^{+i theta/2}). Kept for completeness.
__global__ void apply_rz(double2 *sv, long long ns, int q, double theta) {
    long long base = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * COARSEN;
    long long half = ns >> 1;
    double c_ = cos(theta / 2.0), s_ = sin(theta / 2.0);            // Half-angle as usual for rotations.
    #pragma unroll
    for (int c = 0; c < COARSEN; c++) {
        long long tid = base + c;
        if (tid >= half) return;
        long long lo = tid & ((1LL << q) - 1);
        long long hi = tid >> q;
        long long i0 = (hi << (q + 1)) | lo;
        long long i1 = i0 | (1LL << q);
        double2 v0 = sv[i0], v1 = sv[i1];
        sv[i0] = make_double2(c_*v0.x + s_*v0.y, c_*v0.y - s_*v0.x);// multiply |0> by e^{-i theta/2}.
        sv[i1] = make_double2(c_*v1.x - s_*v1.y, c_*v1.y + s_*v1.x);// multiply |1> by e^{+i theta/2}.
    }
}


// ----------------------------------------------------------------------------
//  Two-qubit fused exponentials  exp(-i theta/2 * P x P)  for P in {X,Y,Z}.
//  Index surgery splits into (outer, mid, inner) around the two qubits and
//  forms the four basis indices i00,i01,i10,i11. Verified against
//  exp(-i theta/2 PP) = cos(theta/2) I - i sin(theta/2) PP. Correct.
//  IMPORTANT physics note: on a SINGLE bond, XX, YY and ZZ mutually commute,
//  so RXX * RYY * RZZ on one bond is EXACT (no Trotter error within a bond).
//  All Trotter error comes only from neighbouring bonds sharing a qubit.
// ----------------------------------------------------------------------------

__global__ void apply_rxx_fused(double2 *sv, long long ns, int q1, int q2, double theta) {
    long long base = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * COARSEN;
    long long quarter = ns >> 2;                                    // ns/4 quartets of amplitudes.
    int lo_bit = min(q1, q2), hi_bit = max(q1, q2);                 // Order the two qubit indices.
    double c = cos(theta / 2.0), s = sin(theta / 2.0);             // Rotation half-angle.
    #pragma unroll
    for (int ci = 0; ci < COARSEN; ci++) {
        long long tid = base + ci;
        if (tid >= quarter) return;
        long long inner = tid & ((1LL << lo_bit) - 1);             // Bits below lo_bit.
        long long mid   = (tid >> lo_bit) & ((1LL << (hi_bit - lo_bit - 1)) - 1); // Bits between the two qubits.
        long long outer = tid >> (hi_bit - 1);                    // Bits above hi_bit.
        long long i00 = (outer << (hi_bit+1)) | (mid << (lo_bit+1)) | inner; // both qubits 0
        long long i01 = i00 | (1LL << lo_bit);                    // lo qubit 1
        long long i10 = i00 | (1LL << hi_bit);                    // hi qubit 1
        long long i11 = i00 | (1LL << lo_bit) | (1LL << hi_bit);  // both qubits 1
        double2 v00=sv[i00], v01=sv[i01], v10=sv[i10], v11=sv[i11];// Read the quartet.
        // XX couples 00<->11 and 01<->10 with a -i sin factor:
        sv[i00] = make_double2(c*v00.x + s*v11.y,  c*v00.y - s*v11.x);
        sv[i01] = make_double2(c*v01.x + s*v10.y,  c*v01.y - s*v10.x);
        sv[i10] = make_double2(c*v10.x + s*v01.y,  c*v10.y - s*v01.x);
        sv[i11] = make_double2(c*v11.x + s*v00.y,  c*v11.y - s*v00.x);
    }
}

__global__ void apply_ryy_fused(double2 *sv, long long ns, int q1, int q2, double theta) {
    long long base = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * COARSEN;
    long long quarter = ns >> 2;
    int lo_bit = min(q1, q2), hi_bit = max(q1, q2);
    double c = cos(theta / 2.0), s = sin(theta / 2.0);
    #pragma unroll
    for (int ci = 0; ci < COARSEN; ci++) {
        long long tid = base + ci;
        if (tid >= quarter) return;
        long long inner = tid & ((1LL << lo_bit) - 1);
        long long mid   = (tid >> lo_bit) & ((1LL << (hi_bit - lo_bit - 1)) - 1);
        long long outer = tid >> (hi_bit - 1);
        long long i00 = (outer << (hi_bit+1)) | (mid << (lo_bit+1)) | inner;
        long long i01 = i00 | (1LL << lo_bit);
        long long i10 = i00 | (1LL << hi_bit);
        long long i11 = i00 | (1LL << lo_bit) | (1LL << hi_bit);
        double2 v00=sv[i00], v01=sv[i01], v10=sv[i10], v11=sv[i11];
        // YY: signs differ from XX because Y|0>=i|1>, Y|1>=-i|0> (00<->11 picks up -1 inside YY):
        sv[i00] = make_double2(c*v00.x - s*v11.y,  c*v00.y + s*v11.x);
        sv[i01] = make_double2(c*v01.x + s*v10.y,  c*v01.y - s*v10.x);
        sv[i10] = make_double2(c*v10.x + s*v01.y,  c*v10.y - s*v01.x);
        sv[i11] = make_double2(c*v11.x - s*v00.y,  c*v11.y + s*v00.x);
    }
}

__global__ void apply_rzz_fused(double2 *sv, long long ns, int q1, int q2, double theta) {
    long long base = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * COARSEN;
    long long quarter = ns >> 2;
    int lo_bit = min(q1, q2), hi_bit = max(q1, q2);
    double c = cos(theta / 2.0), s = sin(theta / 2.0);
    #pragma unroll
    for (int ci = 0; ci < COARSEN; ci++) {
        long long tid = base + ci;
        if (tid >= quarter) return;
        long long inner = tid & ((1LL << lo_bit) - 1);
        long long mid   = (tid >> lo_bit) & ((1LL << (hi_bit - lo_bit - 1)) - 1);
        long long outer = tid >> (hi_bit - 1);
        long long i00 = (outer << (hi_bit+1)) | (mid << (lo_bit+1)) | inner;
        long long i01 = i00 | (1LL << lo_bit);
        long long i10 = i00 | (1LL << hi_bit);
        long long i11 = i00 | (1LL << lo_bit) | (1LL << hi_bit);
        // ZZ is diagonal: parity-even (00,11) get e^{-i theta/2}, parity-odd (01,10) get e^{+i theta/2}.
        double2 v00=sv[i00]; sv[i00] = make_double2(c*v00.x + s*v00.y, c*v00.y - s*v00.x);
        double2 v11=sv[i11]; sv[i11] = make_double2(c*v11.x + s*v11.y, c*v11.y - s*v11.x);
        double2 v01=sv[i01]; sv[i01] = make_double2(c*v01.x - s*v01.y, c*v01.y + s*v01.x);
        double2 v10=sv[i10]; sv[i10] = make_double2(c*v10.x - s*v10.y, c*v10.y + s*v10.x);
    }
}


// ----------------------------------------------------------------------------
//  Sector extract/insert: the ancilla (qubit 0) is the least-significant bit.
//  The "odd sector" (ancilla == 1) holds the system register on which we apply
//  the Heisenberg dynamics; extract pulls sv[2*i+1] into a compact system-only
//  array, insert writes it back. (Unchanged.)
// ----------------------------------------------------------------------------
__global__ void extract_odd_sector(const double2 *sv, double2 *sv_sys, long long ns_sys) {
    long long base = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * COARSEN;
    #pragma unroll
    for (int c = 0; c < COARSEN; c++) {
        long long i = base + c;
        if (i >= ns_sys) return;
        sv_sys[i] = sv[2*i + 1];                                    // 2*i+1 == ancilla bit set to 1.
    }
}

__global__ void insert_odd_sector(const double2 *sv_sys, double2 *sv, long long ns_sys) {
    long long base = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * COARSEN;
    #pragma unroll
    for (int c = 0; c < COARSEN; c++) {
        long long i = base + c;
        if (i >= ns_sys) return;
        sv[2*i + 1] = sv_sys[i];                                    // Write the evolved system amplitudes back.
    }
}

// (Kept for debugging; not on the hot path.)
__global__ void extract_amplitude(const double2* sv, long long target_idx, double2* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = sv[target_idx];
}


// ----------------------------------------------------------------------------
//  Block-count helpers: convert a number of logical work-items into a grid
//  size, accounting for COARSEN.
// ----------------------------------------------------------------------------
static inline int blocks_half(long long ns) {                       // For single-qubit gates: ns/2 pairs.
    long long half = ns >> 1;
    return (int)((half + (long long)BLOCK * COARSEN - 1) / ((long long)BLOCK * COARSEN));
}
static inline int blocks_quarter(long long ns) {                    // For two-qubit gates: ns/4 quartets.
    long long quarter = ns >> 2;
    return (int)((quarter + (long long)BLOCK * COARSEN - 1) / ((long long)BLOCK * COARSEN));
}
static inline int blocks_full(long long ns) {                       // For full-array passes (e.g. reduction).
    return (int)((ns + (long long)BLOCK * COARSEN - 1) / ((long long)BLOCK * COARSEN));
}
static inline int blocks_n(long long n) {                           // For system-sized passes (extract/insert).
    return (int)((n + (long long)BLOCK * COARSEN - 1) / ((long long)BLOCK * COARSEN));
}


// ----------------------------------------------------------------------------
//  apply_trotter_bond: one full bond exponential exp(-i dt (JX XX + JY YY + JZ ZZ))
//  on qubits (q1,q2). Because XX,YY,ZZ commute on a single bond this product is
//  EXACT for that bond (the RXX/RYY/RZZ order does not matter on one bond).
//  Angle is 2*J*dt because RPP(theta)=exp(-i theta/2 PP).
// ----------------------------------------------------------------------------
void apply_trotter_bond(double2 *sv, long long ns, int q1, int q2,
                        double JX, double JY, double JZ, double dt, cudaStream_t stream = 0) {
    int b4 = blocks_quarter(ns);                                    // Grid for two-qubit kernels.
    apply_rxx_fused<<<b4, BLOCK, 0, stream>>>(sv, ns, q1, q2, 2.0 * JX * dt); // exp(-i JX dt XX)
    apply_ryy_fused<<<b4, BLOCK, 0, stream>>>(sv, ns, q1, q2, 2.0 * JY * dt); // exp(-i JY dt YY)
    apply_rzz_fused<<<b4, BLOCK, 0, stream>>>(sv, ns, q1, q2, 2.0 * JZ * dt); // exp(-i JZ dt ZZ)
}

// ----------------------------------------------------------------------------
//  [FIX 4] trotter_step_symmetric: ONE second-order (Strang) Trotter step of
//  size dt_step, applied IDENTICALLY each time it is called:
//      even bonds at dt_step/2  ->  odd bonds at dt_step  ->  even bonds at dt_step/2
//  Symmetric splitting gives O(dt_step^3) local / O(dt_step^2) global error.
//  Crucially, every call is the SAME unitary U(dt_step), so applying it N times
//  is exactly U(dt_step)^N -- the power structure the Toeplitz fill relies on.
//  (The old code alternated forward/reverse orderings between steps, which made
//  consecutive steps DIFFERENT unitaries and broke the power property.)
// ----------------------------------------------------------------------------
void trotter_step_symmetric(double2 *sv_sys, long long ns_sys, int nQ,
                            double JX, double JY, double JZ, double dt_step) {
    double half = 0.5 * dt_step;                                    // Half step for the outer (even) layers.
    // First half-step on EVEN bonds (0-1, 2-3, ...); these bonds are disjoint.
    for (int q = 0; q < nQ - 1; q += 2)
        apply_trotter_bond(sv_sys, ns_sys, q, q+1, JX, JY, JZ, half, 0);
    // Full step on ODD bonds (1-2, 3-4, ...); also disjoint among themselves.
    for (int q = 1; q < nQ - 1; q += 2)
        apply_trotter_bond(sv_sys, ns_sys, q, q+1, JX, JY, JZ, dt_step, 0);
    // Second half-step on EVEN bonds -> makes the whole step time-symmetric.
    for (int q = 0; q < nQ - 1; q += 2)
        apply_trotter_bond(sv_sys, ns_sys, q, q+1, JX, JY, JZ, half, 0);
    // Kernels on the default stream serialize in issue order, so layers are
    // applied in the correct sequence; we sync once at the call site after the
    // full evolution, before measuring.
}


// ----------------------------------------------------------------------------
//  compute_sk_hk_rank: the heart of the method. For Krylov index k it returns
//      outS = <psi_0 | U^k | psi_0>                       (overlap S_k)
//      outH = <psi_0 | H U^k | psi_0>                     (Hamiltonian moment H_k)
//  via a Hadamard test on an ancilla (qubit 0), where U = U(dt/steps_per_unit)
//  is ONE fixed symmetric Trotter step and U^k is k*steps_per_unit copies of it.
//
//  [FIX 1] Signature now takes the INTEGER k and the BASE dt separately (not a
//          pre-multiplied dt_total), so we can hold the step SIZE fixed and let
//          the step COUNT grow with k -- guaranteeing |psi_k> = U^k|psi_0>.
// ----------------------------------------------------------------------------
void compute_sk_hk_rank(int nQ, int /*excBit -- vestigial, kept for ABI*/,
                        int steps_per_unit,             // Trotter steps PER Krylov unit (fixed step size = dt/this).
                        double JX, double JY, double JZ,
                        int k, double dt,               // [FIX 1] integer power k and base dt.
                        Complex &outS, Complex &outH,
                        int /*rank*/) {

    long long ns     = 1LL << (nQ + 1);                             // Full Hilbert space incl. ancilla: 2^(nQ+1).
    size_t sv_bytes  = ns * sizeof(double2);                       // Bytes for one full statevector.
    long long ns_sys = 1LL << nQ;                                  // System-only Hilbert space: 2^nQ.

    int bh      = blocks_half(ns);                                 // Grids for various passes (see helpers).
    int bh_sys  = blocks_half(ns_sys);
    int bf      = blocks_full(ns);
    int bn_sys  = blocks_n(ns_sys);

    // measure_x: launch the reduction kernel and copy the scalar <X_anc> back.
    auto measure_x = [&](double2 *sv) -> double {
        double *d_val;                                             // Device accumulator.
        cudaMalloc(&d_val, sizeof(double));
        cudaMemset(d_val, 0, sizeof(double));                      // Start at zero (kernel atomicAdds into it).
        reduce_x_expectation<<<bf, BLOCK>>>(sv, ns, d_val);
        cudaDeviceSynchronize();                                   // Wait for the reduction to finish.
        double h_val;
        cudaMemcpy(&h_val, d_val, sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_val);
        return h_val;                                              // = Re or Im of the requested overlap.
    };

    // run_trotter: prepare the Hadamard-test state with the reference |psi_0>
    // (a Neel-like state in the ancilla=1 sector) and apply U^k to the system.
    auto run_trotter = [&]() -> double2* {
        double2 *sv;
        cudaMalloc(&sv, sv_bytes);
        cudaMemset(sv, 0, sv_bytes);                               // Start from |0...0>.
        double2 one = {1.0, 0.0};
        cudaMemcpy(sv, &one, sizeof(double2), cudaMemcpyHostToDevice); // Amplitude 1 on |0...0>.
        apply_hadamard<<<bh, BLOCK>>>(sv, ns, 0);                  // Put the ancilla (qubit 0) into |+>.
        for (int _q = 0; _q < nQ; _q += 2)                        // Controlled-prep: entangle ancilla with the
            apply_cx<<<bh, BLOCK>>>(sv, ns, 0, _q + 1);           //   system to build the reference |psi_0>.
        cudaDeviceSynchronize();

        // [FIX 1 + FIX 4] Fixed step SIZE; step COUNT scales with k.
        double dt_step = dt / steps_per_unit;                     // Constant elementary step size (independent of k).
        long long total_steps = (long long)k * steps_per_unit;    // U^k == this many identical symmetric steps.

        size_t sv_sys_bytes = ns_sys * sizeof(double2);
        double2 *sv_sys;
        cudaMalloc(&sv_sys, sv_sys_bytes);
        extract_odd_sector<<<bn_sys, BLOCK>>>(sv, sv_sys, ns_sys); // Pull the ancilla=1 branch (the system register).
        cudaDeviceSynchronize();

        for (long long j = 0; j < total_steps; j++)                // Apply the SAME symmetric step total_steps times.
            trotter_step_symmetric(sv_sys, ns_sys, nQ, JX, JY, JZ, dt_step);
        cudaDeviceSynchronize();                                   // Finish all evolution before reinserting.

        insert_odd_sector<<<bn_sys, BLOCK>>>(sv_sys, sv, ns_sys);  // Write the evolved system back into the ancilla=1 branch.
        cudaFree(sv_sys);
        return sv;                                                 // Caller owns sv and must cudaFree it.
    };

    // ---- S(k) = <psi_0 | U^k | psi_0> -----------------------------------
    double2 *sv_s = run_trotter();                                 // |Phi> = (controlled U^k) acting in the test register.
    double2 *sv_s_im;
    cudaMalloc(&sv_s_im, sv_bytes);
    cudaMemcpy(sv_s_im, sv_s, sv_bytes, cudaMemcpyDeviceToDevice); // Clone for the imaginary-part branch.

    // Real part: undo the controlled prep, Hadamard the ancilla, measure <X>.
    apply_x_gate   <<<bh, BLOCK>>>(sv_s, ns, 0);
    for (int _q = 0; _q < nQ; _q += 2) apply_cx<<<bh, BLOCK>>>(sv_s, ns, 0, _q + 1);
    apply_x_gate   <<<bh, BLOCK>>>(sv_s, ns, 0);
    apply_hadamard <<<bh, BLOCK>>>(sv_s, ns, 0);
    cudaDeviceSynchronize();
    double re_s = measure_x(sv_s);                                 // Re <psi_0|U^k|psi_0>.
    cudaFree(sv_s);

    // Imag part: same, but an S-dagger on the ancilla rotates the measured axis.
    apply_x_gate   <<<bh, BLOCK>>>(sv_s_im, ns, 0);
    for (int _q = 0; _q < nQ; _q += 2) apply_cx<<<bh, BLOCK>>>(sv_s_im, ns, 0, _q + 1);
    apply_x_gate   <<<bh, BLOCK>>>(sv_s_im, ns, 0);
    apply_sdg_gate <<<bh, BLOCK>>>(sv_s_im, ns, 0);
    apply_hadamard <<<bh, BLOCK>>>(sv_s_im, ns, 0);
    cudaDeviceSynchronize();
    double im_s = measure_x(sv_s_im);                              // Im <psi_0|U^k|psi_0>.
    cudaFree(sv_s_im);
    outS = Complex(re_s, im_s);                                    // Assemble the complex overlap S_k.

    // ---- H(k) = <psi_0 | H U^k | psi_0> ---------------------------------
    // H = sum over bonds of (JX XX + JY YY + JZ ZZ). We measure each Pauli pair
    // on each bond via the same Hadamard test and accumulate with its coupling.
    outH = Complex(0.0, 0.0);
    double coeffs[3] = {JX, JY, JZ};                               // Index 0->XX, 1->YY, 2->ZZ couplings.
    double2 *sv_base = run_trotter();                              // Re-prepare U^k|psi_0> once; reuse for all bonds.

    for (int b = 0; b < nQ - 1; b++) {                             // Loop over open-chain bonds (b, b+1).
        int q1_sys = b, q2_sys = b + 1;                            // System-register qubit indices for this bond.
        for (int pauli = 0; pauli < 3; pauli++) {                  // pauli = 0:XX, 1:YY, 2:ZZ.

            // --- Real part of <psi_0| P_bond U^k |psi_0> ---
            double2 *sv_re;
            cudaMalloc(&sv_re, sv_bytes);
            cudaMemcpy(sv_re, sv_base, sv_bytes, cudaMemcpyDeviceToDevice); // Fresh copy of U^k|psi_0>.
            double2 *sv_sys_p;
            cudaMalloc(&sv_sys_p, ns_sys * sizeof(double2));
            extract_odd_sector<<<bn_sys, BLOCK>>>(sv_re, sv_sys_p, ns_sys); // Operate on the system sector.
            cudaDeviceSynchronize();
            if (pauli == 0) {                                      // Apply the chosen Pauli pair on this bond.
                apply_x_gate<<<bh_sys, BLOCK>>>(sv_sys_p, ns_sys, q1_sys);
                apply_x_gate<<<bh_sys, BLOCK>>>(sv_sys_p, ns_sys, q2_sys);
            } else if (pauli == 1) {
                apply_y_gate<<<bh_sys, BLOCK>>>(sv_sys_p, ns_sys, q1_sys);
                apply_y_gate<<<bh_sys, BLOCK>>>(sv_sys_p, ns_sys, q2_sys);
            } else {
                apply_z_gate<<<bh_sys, BLOCK>>>(sv_sys_p, ns_sys, q1_sys);
                apply_z_gate<<<bh_sys, BLOCK>>>(sv_sys_p, ns_sys, q2_sys);
            }
            insert_odd_sector<<<bn_sys, BLOCK>>>(sv_sys_p, sv_re, ns_sys);
            cudaFree(sv_sys_p);
            cudaDeviceSynchronize();
            apply_x_gate   <<<bh, BLOCK>>>(sv_re, ns, 0);          // Undo controlled prep + Hadamard test read-out.
            for (int _q = 0; _q < nQ; _q += 2) apply_cx<<<bh, BLOCK>>>(sv_re, ns, 0, _q + 1);
            apply_x_gate   <<<bh, BLOCK>>>(sv_re, ns, 0);
            apply_hadamard <<<bh, BLOCK>>>(sv_re, ns, 0);
            cudaDeviceSynchronize();
            double re_h = measure_x(sv_re);
            cudaFree(sv_re);

            // --- Imag part (S-dagger rotates the measurement axis) ---
            double2 *sv_im;
            cudaMalloc(&sv_im, sv_bytes);
            cudaMemcpy(sv_im, sv_base, sv_bytes, cudaMemcpyDeviceToDevice);
            double2 *sv_sys_p2;
            cudaMalloc(&sv_sys_p2, ns_sys * sizeof(double2));
            extract_odd_sector<<<bn_sys, BLOCK>>>(sv_im, sv_sys_p2, ns_sys);
            cudaDeviceSynchronize();
            if (pauli == 0) {
                apply_x_gate<<<bh_sys, BLOCK>>>(sv_sys_p2, ns_sys, q1_sys);
                apply_x_gate<<<bh_sys, BLOCK>>>(sv_sys_p2, ns_sys, q2_sys);
            } else if (pauli == 1) {
                apply_y_gate<<<bh_sys, BLOCK>>>(sv_sys_p2, ns_sys, q1_sys);
                apply_y_gate<<<bh_sys, BLOCK>>>(sv_sys_p2, ns_sys, q2_sys);
            } else {
                apply_z_gate<<<bh_sys, BLOCK>>>(sv_sys_p2, ns_sys, q1_sys);
                apply_z_gate<<<bh_sys, BLOCK>>>(sv_sys_p2, ns_sys, q2_sys);
            }
            insert_odd_sector<<<bn_sys, BLOCK>>>(sv_sys_p2, sv_im, ns_sys);
            cudaFree(sv_sys_p2);
            cudaDeviceSynchronize();
            apply_x_gate   <<<bh, BLOCK>>>(sv_im, ns, 0);
            for (int _q = 0; _q < nQ; _q += 2) apply_cx<<<bh, BLOCK>>>(sv_im, ns, 0, _q + 1);
            apply_x_gate   <<<bh, BLOCK>>>(sv_im, ns, 0);
            apply_sdg_gate <<<bh, BLOCK>>>(sv_im, ns, 0);
            apply_hadamard <<<bh, BLOCK>>>(sv_im, ns, 0);
            cudaDeviceSynchronize();
            double im_h = measure_x(sv_im);
            cudaFree(sv_im);

            outH += coeffs[pauli] * Complex(re_h, im_h);           // Accumulate this bond/Pauli contribution into H_k.
        }
    }
    cudaFree(sv_base);
}


// ----------------------------------------------------------------------------
//  ks_for_rank: round-robin assignment of Krylov indices 1..krylov_dim-1 to MPI
//  ranks. (k=0 is trivial: S_0=1, H_0 known analytically, handled on rank 0.)
// ----------------------------------------------------------------------------
std::vector<int> ks_for_rank(int rank, int nranks, int krylov_dim) {
    std::vector<int> all_ks;
    for (int k = 1; k < krylov_dim; k++) all_ks.push_back(k);       // Indices 1..K-1.
    std::vector<int> my_ks;
    for (int i = 0; i < (int)all_ks.size(); i++)
        if (i % nranks == rank) my_ks.push_back(all_ks[i]);         // This rank takes every nranks-th index.
    return my_ks;
}


// ----------------------------------------------------------------------------
//  heisenberg_matvec: y = H x in the FULL 2^nQ space, used only by the
//  classical reference solver below. H = sum_i (JX X_iX_{i+1}+JY Y_iY_{i+1}+
//  JZ Z_iZ_{i+1}), open chain. (Unchanged.)
// ----------------------------------------------------------------------------
static void heisenberg_matvec(const std::vector<double> &in, std::vector<double> &out,
                              int nQ, double JX, double JY, double JZ) {
    long long dim = 1LL << nQ;
    std::fill(out.begin(), out.end(), 0.0);
    for (long long s = 0; s < dim; s++) {
        double xs = in[s], diag = 0.0;
        for (int i = 0; i < nQ - 1; i++) {
            int bi = (int)((s >> i) & 1);                          // Spin at site i.
            int bj = (int)((s >> (i + 1)) & 1);                    // Spin at site i+1.
            diag += JZ * (bi == bj ? 1.0 : -1.0);                  // Z_iZ_{i+1}: +1 aligned, -1 anti.
            long long sf = s ^ (1LL << i) ^ (1LL << (i + 1));      // Flip both spins (XX/YY off-diagonal).
            double coeff = (bi == bj) ? (JX - JY) : (JX + JY);     // Combined XX+YY matrix element.
            out[sf] += coeff * xs;
        }
        out[s] += diag * xs;                                       // Diagonal (ZZ) contribution.
    }
}

// Sturm sequence: number of eigenvalues of the tridiagonal (a,b) strictly below x.
static int sturm_count(const std::vector<double> &a, const std::vector<double> &b, double x) {
    int n = (int)a.size(), count = 0;
    double d = a[0] - x;
    if (d < 0.0) count++;
    for (int i = 1; i < n; i++) {
        if (d == 0.0) d = 1e-300;                                  // Avoid division by exactly zero.
        d = (a[i] - x) - (b[i - 1] * b[i - 1]) / d;
        if (d < 0.0) count++;
    }
    return count;
}

// Classical reference extrema (E_min, E_max) via matrix-free Lanczos + bisection.
std::pair<double, double> heisenberg_exact_extrema(int nQ, double JX, double JY, double JZ) {
    long long dim = 1LL << nQ;
    int m = (int)std::min(dim, (long long)200);                    // Lanczos depth (ample for dim<=1024).

    std::vector<double> v_prev(dim, 0.0), v(dim), w(dim);
    std::vector<std::vector<double>> Q; Q.reserve(m);

    double nrm = 0.0;                                              // Generic start vector touching all sectors.
    for (long long i = 0; i < dim; i++) { v[i] = std::sin(0.1*(i+1)) + 0.123; nrm += v[i]*v[i]; }
    nrm = std::sqrt(nrm);
    for (long long i = 0; i < dim; i++) v[i] /= nrm;

    std::vector<double> alpha, beta;
    double beta_prev = 0.0;
    for (int it = 0; it < m; it++) {
        Q.push_back(v);
        heisenberg_matvec(v, w, nQ, JX, JY, JZ);                   // w = H v.
        double a = 0.0;
        for (long long i = 0; i < dim; i++) a += w[i] * v[i];      // alpha = <v|H|v>.
        for (long long i = 0; i < dim; i++) w[i] -= a * v[i] + beta_prev * v_prev[i];
        for (auto &q : Q) {                                        // Full reorthogonalization (numerical hygiene).
            double dot = 0.0;
            for (long long i = 0; i < dim; i++) dot += w[i] * q[i];
            for (long long i = 0; i < dim; i++) w[i] -= dot * q[i];
        }
        double bnrm = 0.0;
        for (long long i = 0; i < dim; i++) bnrm += w[i] * w[i];
        bnrm = std::sqrt(bnrm);
        alpha.push_back(a);
        if (bnrm < 1e-12) break;                                   // Invariant subspace reached.
        beta.push_back(bnrm);
        v_prev = v;
        for (long long i = 0; i < dim; i++) v[i] = w[i] / bnrm;
        beta_prev = bnrm;
    }

    int n = (int)alpha.size();
    double lo = std::numeric_limits<double>::infinity(), hi = -lo;
    for (int i = 0; i < n; i++) {                                  // Gershgorin bracket for bisection.
        double r = (i>0 ? std::abs(beta[i-1]) : 0.0) + (i<n-1 ? std::abs(beta[i]) : 0.0);
        lo = std::min(lo, alpha[i] - r);
        hi = std::max(hi, alpha[i] + r);
    }
    double min_lo = lo, min_hi = hi;                               // Bisect for the smallest eigenvalue.
    for (int iter = 0; iter < 200; iter++) {
        double mid = 0.5 * (min_lo + min_hi);
        if (sturm_count(alpha, beta, mid) >= 1) min_hi = mid; else min_lo = mid;
    }
    double E_min = 0.5 * (min_lo + min_hi);

    double max_lo = lo, max_hi = hi;                               // Bisect for the largest eigenvalue.
    for (int iter = 0; iter < 200; iter++) {
        double mid = 0.5 * (max_lo + max_hi);
        if (sturm_count(alpha, beta, mid) >= n) max_hi = mid; else max_lo = mid;
    }
    double E_max = 0.5 * (max_lo + max_hi);

    return {E_min, E_max};
}


// ============================================================================
//  main
// ============================================================================
int main(int argc, char **argv) {

    // ---- Read parameters from input.txt: JX JY JZ [nQ] [krylov_dim] [trotter_steps] ----
    double JX, JY, JZ;
    int nQ;
    int krylov_dim = 8;
    int trotter_steps = 4;                                         // NOTE: now interpreted as STEPS PER KRYLOV UNIT.
    {
        std::ifstream f("input.txt");
        if (!f.is_open()) { std::cerr << "Cannot open input.txt\n"; return 1; }
        f >> JX >> JY >> JZ;
        if (!(f >> nQ)) nQ = 20;
        if (!(f >> krylov_dim)) krylov_dim = 8;
        if (!(f >> trotter_steps)) trotter_steps = 4;
    }
    if (krylov_dim < 1) krylov_dim = 1;
    if (krylov_dim > 64) krylov_dim = 64;
    if (trotter_steps < 1) trotter_steps = 1;

    // ---- MPI / GPU setup ----
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    if (nranks < 1 || nranks > 8) {
        if (rank == 0) std::cerr << "ERROR: Supported GPU counts are 1..8.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    cudaSetDevice(rank % num_gpus);                                // Pin each rank to a GPU.
    cudaFree(0);                                                   // Force context creation now.

    int excBit = nQ / 2;                                           // Vestigial; passed through but unused downstream.

    // ---- Classical reference extrema (also used to pick dt) ----
    auto [E0_exact, E_max] = heisenberg_exact_extrema(nQ, JX, JY, JZ);

    double E_scale = std::max(std::abs(E0_exact), std::abs(E_max));
    double dt      = PI / std::max(E_scale, 1e-12);                // Guard against E_scale == 0.

    // ---- Analytic k=0 values for the Toeplitz seed ----
    Complex S0 = Complex(1.0, 0.0);                                // <psi_0|psi_0> = 1.
    Complex H0 = Complex(-(double)(nQ - 1) * JZ, 0.0);            // <Neel|H|Neel>: every bond antiparallel -> -JZ each.

    // ---- Distribute Krylov indices and run the quantum part ----
    std::vector<int> my_ks = ks_for_rank(rank, nranks, krylov_dim);
    std::vector<double> local_results(krylov_dim * 4, 0.0);        // [k*4 + {Sr,Si,Hr,Hi}] packed per index.

    MPI_Barrier(MPI_COMM_WORLD);
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int k : my_ks) {                                          // Each rank computes its assigned S_k, H_k.
        Complex my_S(0,0), my_H(0,0);
        // [FIX 1] pass integer k and base dt; fixed step size, k*steps_per_unit steps.
        compute_sk_hk_rank(nQ, excBit, trotter_steps,
                           JX, JY, JZ,
                           k, dt,
                           my_S, my_H, rank);
        cudaDeviceSynchronize();
        local_results[k*4+0] = my_S.real();
        local_results[k*4+1] = my_S.imag();
        local_results[k*4+2] = my_H.real();
        local_results[k*4+3] = my_H.imag();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    // ---- Gather everyone's results onto all ranks ----
    std::vector<double> all_results(krylov_dim * 4, 0.0);
    MPI_Allreduce(local_results.data(), all_results.data(),
                  krylov_dim * 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // ---- Rank 0 builds the Toeplitz matrices, runs the GEVP sweep, reports ----
    if (rank == 0) {
        const char *results_env = std::getenv("RESULTS_FILE");
        const std::string results_path = (results_env && results_env[0]) ? results_env : "results.txt";
        std::ofstream out(results_path, std::ios::app);
        out << "\n==================================================\n";
        out << "COARSEN=" << COARSEN << "  nGPUs=" << nranks << "  krylov_dim=" << krylov_dim
            << "  nQ=" << nQ << "  trotter_steps(per unit)=" << trotter_steps << "\n";

        // Repack the flat MPI buffer into per-k complex arrays SR[k], HR[k].
        std::vector<Complex> SR(krylov_dim), HR(krylov_dim);
        SR[0] = S0; HR[0] = H0;                                    // Seed the analytic k=0 entries.
        for (int k = 1; k < krylov_dim; k++) {
            SR[k] = Complex(all_results[k*4+0], all_results[k*4+1]);
            HR[k] = Complex(all_results[k*4+2], all_results[k*4+3]);
        }

        out << "JX=" << JX << " JY=" << JY << " JZ=" << JZ << "\n";
        out << "dt(anti-alias)=" << std::fixed << std::setprecision(6) << dt << "\n";
        out << "Classical exact E0: " << std::setprecision(8) << E0_exact << "\n\n";
        out << "Wall time: " << std::setprecision(1) << elapsed_ms << " ms\n";
        std::cout << "\n=== nQ=" << nQ << "  trotter_steps(per unit)=" << trotter_steps
                  << "  dt=" << std::setprecision(6) << dt
                  << "  exact_E0=" << std::setprecision(8) << E0_exact << " ===\n";
        std::cout << "  Krylov convergence:\n";
        out << "  Krylov convergence:\n";

        const double CONV_TOL = 1e-7;                              // Declare convergence when E0 stops moving.
        double best_e0   = std::numeric_limits<double>::infinity();// Best (lowest physical) estimate so far.
        int    best_k    = -1;
        double prev_e0   = std::numeric_limits<double>::quiet_NaN();// Previous K's estimate (for delta).

        for (int K = 1; K <= krylov_dim; K++) {                    // Grow the Krylov subspace one vector at a time.
            // Build the K x K Toeplitz overlap (Sk) and Hamiltonian (Hk) blocks.
            // [FIX 1 makes this fill exact]: with U^k a true power, S is Toeplitz
            // and Hermitian PSD; H is Toeplitz up to Trotter error.
            CMat Sk(K, std::vector<Complex>(K));
            CMat Hk(K, std::vector<Complex>(K));
            for (int i = 0; i < K; i++) {
                for (int j = 0; j < K; j++) {
                    int d = std::abs(i - j);                       // Toeplitz: entry depends only on |i-j|.
                    Sk[i][j] = (i <= j) ? SR[d] : std::conj(SR[d]);// Lower triangle is the conjugate (Hermitian).
                    Hk[i][j] = (i <= j) ? HR[d] : std::conj(HR[d]);
                }
            }
            double e0 = solve_gen_eig(Hk, Sk);
            if (!std::isfinite(e0)) {
                std::cout << "  K=" << K << "  LAPACK solve failed; stopping.\n";
                out         << "  K=" << K << "  LAPACK solve failed; stopping.\n";
                break;
            }

            double err = std::abs(e0 - E0_exact);                  // Error vs the classical reference.
            double delta = std::isfinite(prev_e0) ? std::abs(e0 - prev_e0) : 1e30; // Change since last K.

            if (e0 < best_e0) { best_e0 = e0; best_k = K; }        // Track the best physical estimate.

            std::cout << "  K=" << K << "  E0=" << std::fixed << std::setprecision(8)
                      << e0 << "  |err|=" << std::setprecision(8) << err << "\n";
            out << "  K=" << K << "  E0=" << e0 << "  |err|=" << err << "\n";

            // Convergence test: E0 has stopped changing to within CONV_TOL.
            if (delta < CONV_TOL) {
                std::cout << "  Converged at K=" << K << "  E0=" << e0
                          << "  |err|=" << err << "  (E0 stationary)\n";
                out << "  Converged at K=" << K << "  E0=" << e0 << "  |err|=" << err << "\n";
                break;
            }
            prev_e0 = e0;                                          // Carry forward for the next delta.
        }

        // Final summary line: the best physical estimate and its residual error.
        if (best_k > 0) {
            double best_err = std::abs(best_e0 - E0_exact);
            std::cout << "  BEST  K=" << best_k << "  E0=" << std::setprecision(8) << best_e0
                      << "  |err|=" << best_err << "\n";
            out << "  BEST  K=" << best_k << "  E0=" << best_e0 << "  |err|=" << best_err << "\n";
        }
        out.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
