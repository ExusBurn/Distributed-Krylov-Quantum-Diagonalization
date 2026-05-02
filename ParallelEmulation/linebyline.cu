#include <mpi.h>           
#include <cuda_runtime.h>  
#include <cuComplex.h>     

#include <iostream>        
#include <vector>          
#include <complex>         
#include <cmath>           
#include <algorithm>       
#include <numeric>         
#include <fstream>         
#include <sstream>         
#include <iomanip>         
#include <chrono>          



using Complex = std::complex<double>;
using CMat    = std::vector<std::vector<Complex>>;
static const double PI        = 3.14159265358979323846;
__host__ __device__ static constexpr double INV_SQRT2 = 0.7071067811865475;

// Jacobi iterative diagonalization for a Hermitian matrix.
// Returns eigenvalues sorted in ascending order along with their eigenvectors.

// Reduction kernel that computes the ancilla X expectation as P(anc=0) - P(anc=1).
__global__ void reduce_x_expectation(const double2 *sv, long long ns, double *out) {
    __shared__ double sdata[256];
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    double val = 0.0;

    // stride loop over all amplitude pairs
    for (long long i = tid; i < ns; i += (long long)blockDim.x * gridDim.x) {
        double re = sv[i].x, im = sv[i].y;
        double prob = re*re + im*im;
        // even index = anc=0 → +prob, odd index = anc=1 → -prob
        val += (i % 2 == 0) ? prob : -prob;
    }

    sdata[threadIdx.x] = val;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, sdata[0]);
}

// For the Y expectation, we apply Sdg on the ancilla first to rotate into
// the Y basis, then use the same reduction approach.

std::pair<std::vector<double>, CMat> jacobi_eigh(CMat A) {
    int n = (int)A.size();
    CMat V(n, std::vector<Complex>(n, 0));
    for (int i = 0; i < n; i++) V[i][i] = 1;  

    for (int iter = 0; iter < 3000; iter++) {
        double maxOff = 0; int p = 0, q = 1;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (std::abs(A[i][j]) > maxOff) { maxOff = std::abs(A[i][j]); p = i; q = j; }
        if (maxOff < 1e-14) break;  

        double app = A[p][p].real(), aqq = A[q][q].real();
        Complex apq = A[p][q];
        double tau = (aqq - app) / (2.0 * std::abs(apq));
        double t   = (tau >= 0 ? 1.0 : -1.0) / (std::abs(tau) + std::sqrt(1 + tau*tau));
        double c   = 1.0 / std::sqrt(1 + t*t), s = t * c;
        Complex ph = apq / std::abs(apq);
        Complex cs_c(c), cs_s = Complex(s) * std::conj(ph);

        for (int r = 0; r < n; r++) {
            if (r == p || r == q) continue;
            Complex arp = A[r][p], arq = A[r][q];
            A[r][p] = cs_c * arp + cs_s * arq;               A[p][r] = std::conj(A[r][p]);
            A[r][q] = -std::conj(cs_s) * arp + cs_c * arq;   A[q][r] = std::conj(A[r][q]);
        }
        A[p][p] = app - t * std::abs(apq);
        A[q][q] = aqq + t * std::abs(apq);
        A[p][q] = A[q][p] = 0;

        for (int r = 0; r < n; r++) {
            Complex vrp = V[r][p], vrq = V[r][q];
            V[r][p] = cs_c * vrp + cs_s * vrq;
            V[r][q] = -std::conj(cs_s) * vrp + cs_c * vrq;
        }
    }

    std::vector<double> ev(n);
    for (int i = 0; i < n; i++) ev[i] = A[i][i].real();
    std::vector<int> idx(n); std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return ev[a] < ev[b]; });
    std::vector<double> sev(n); CMat sV(n, std::vector<Complex>(n));
    for (int i = 0; i < n; i++) {
        sev[i] = ev[idx[i]];
        for (int r = 0; r < n; r++) sV[r][i] = V[r][idx[i]];
    }
    return {sev, sV};
}

// Solves the generalized eigenvalue problem H c = E S c and returns the lowest eigenvalue.
double solve_gen_eig(const CMat &H, const CMat &S, double threshold) {
    int d = (int)H.size();
    if (d == 1) return H[0][0].real();

    auto [sv, svecs] = jacobi_eigh(S);
    std::vector<std::vector<Complex>> good;
    for (int i = 0; i < d; i++) {
        if (sv[i] > threshold) {
            std::vector<Complex> v(d);
            for (int r = 0; r < d; r++) v[r] = svecs[r][i];
            good.push_back(v);
        }
    }
    int m = (int)good.size();
    if (m == 0) return 1e10;

    CMat Hr(m, std::vector<Complex>(m, 0)), Sr(m, std::vector<Complex>(m, 0));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            for (int r = 0; r < d; r++)
                for (int c = 0; c < d; c++) {
                    Hr[i][j] += std::conj(good[i][r]) * H[r][c] * good[j][c];
                    Sr[i][j] += std::conj(good[i][r]) * S[r][c] * good[j][c];
                }

    auto [srv, srvecs] = jacobi_eigh(Sr);
    CMat Sinv(m, std::vector<Complex>(m, 0));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            for (int k = 0; k < m; k++)
                Sinv[i][j] += srvecs[i][k] * (1.0 / std::sqrt(std::max(srv[k], 1e-20)))
                              * std::conj(srvecs[j][k]);

    CMat Ht(m, std::vector<Complex>(m, 0));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            for (int k = 0; k < m; k++)
                for (int l = 0; l < m; l++)
                    Ht[i][j] += Sinv[i][k] * Hr[k][l] * Sinv[l][j];

    auto [evals, _] = jacobi_eigh(Ht);
    return evals[0]; 
}

double Compute_SpectralNorm(const CMat &H_proj) {
    auto [evals, evecs] = jacobi_eigh(H_proj);
    return std::max(std::abs(evals.front()), std::abs(evals.back()));
}

static const int BLOCK = 256;

// Individual quantum gate kernels, each operating on the full statevector in parallel.
__global__ void apply_x_gate(double2 *sv, long long ns, int q) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ns >> 1) return;
    long long lo = tid & ((1LL << q) - 1);
    long long hi = tid >> q;
    long long i0 = (hi << (q + 1)) | lo;   
    long long i1 = i0 | (1LL << q);         
    double2 tmp = sv[i0]; sv[i0] = sv[i1]; sv[i1] = tmp;
}

__global__ void apply_hadamard(double2 *sv, long long ns, int q) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ns >> 1) return;
    long long lo = tid & ((1LL << q) - 1);
    long long hi = tid >> q;
    long long i0 = (hi << (q + 1)) | lo;
    long long i1 = i0 | (1LL << q);
    double2 v0 = sv[i0], v1 = sv[i1];
    sv[i0] = make_double2((v0.x + v1.x) * INV_SQRT2, (v0.y + v1.y) * INV_SQRT2);
    sv[i1] = make_double2((v0.x - v1.x) * INV_SQRT2, (v0.y - v1.y) * INV_SQRT2);
}

__global__ void apply_cx(double2 *sv, long long ns, int ctrl, int tgt) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ns >> 1) return;
    long long lo = tid & ((1LL << tgt) - 1);
    long long hi = tid >> tgt;
    long long i0 = (hi << (tgt + 1)) | lo;
    long long i1 = i0 | (1LL << tgt);
    if (!(i0 & (1LL << ctrl))) return;  
    double2 tmp = sv[i0]; sv[i0] = sv[i1]; sv[i1] = tmp;
}

__global__ void apply_s_gate(double2 *sv, long long ns, int q) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ns >> 1) return;
    long long lo = tid & ((1LL << q) - 1);
    long long hi = tid >> q;
    long long i1 = ((hi << (q + 1)) | lo) | (1LL << q);
    double2 v1 = sv[i1];
    sv[i1] = make_double2(-v1.y, v1.x);
}

__global__ void apply_sdg_gate(double2 *sv, long long ns, int q) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ns >> 1) return;
    long long lo = tid & ((1LL << q) - 1);
    long long hi = tid >> q;
    long long i1 = ((hi << (q + 1)) | lo) | (1LL << q);
    double2 v1 = sv[i1];
    sv[i1] = make_double2(v1.y, -v1.x);
}

__global__ void apply_rz(double2 *sv, long long ns, int q, double theta) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ns >> 1) return;
    long long lo = tid & ((1LL << q) - 1);
    long long hi = tid >> q;
    long long i0 = (hi << (q + 1)) | lo;
    long long i1 = i0 | (1LL << q);
    double c = cos(theta / 2.0), s = sin(theta / 2.0);
    double2 v0 = sv[i0], v1 = sv[i1];
    sv[i0] = make_double2(c * v0.x + s * v0.y, c * v0.y - s * v0.x);  
    sv[i1] = make_double2(c * v1.x - s * v1.y, c * v1.y + s * v1.x);  
}

__global__ void apply_y_gate(double2 *sv, long long ns, int q) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ns >> 1) return;
    long long lo = tid & ((1LL << q) - 1);
    long long hi = tid >> q;
    long long i0 = (hi << (q + 1)) | lo;
    long long i1 = i0 | (1LL << q);
    double2 v0 = sv[i0], v1 = sv[i1];
    sv[i0] = make_double2(v1.y, -v1.x);   
    sv[i1] = make_double2(-v0.y, v0.x);   
}

__global__ void apply_z_gate(double2 *sv, long long ns, int q) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ns >> 1) return;
    long long lo = tid & ((1LL << q) - 1);
    long long hi = tid >> q;
    long long i1 = ((hi << (q + 1)) | lo) | (1LL << q);
    sv[i1].x = -sv[i1].x;
    sv[i1].y = -sv[i1].y;
}

// Extracts a single amplitude from the statevector on the GPU side.
// This avoids having to copy the entire statevector back to the CPU
// just to read one number.
__global__ void extract_amplitude(const double2* sv, long long target_idx, double2* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out = sv[target_idx];
    }
}

__global__ void apply_rxx_fused(double2 *sv, long long ns, int q1, int q2, double theta) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ns >> 2) return;

    int lo_bit = min(q1, q2), hi_bit = max(q1, q2);
    long long inner = tid & ((1LL << lo_bit) - 1);
    long long mid   = (tid >> lo_bit) & ((1LL << (hi_bit - lo_bit - 1)) - 1);
    long long outer = tid >> (hi_bit - 1);

    long long i00 = (outer << (hi_bit+1)) | (mid << (lo_bit+1)) | inner;
    long long i01 = i00 | (1LL << lo_bit);
    long long i10 = i00 | (1LL << hi_bit);
    long long i11 = i00 | (1LL << lo_bit) | (1LL << hi_bit);

    double c = cos(theta / 2.0);
    double s = sin(theta / 2.0);

    double2 v00 = sv[i00], v01 = sv[i01], v10 = sv[i10], v11 = sv[i11];

    // 00 -> c*v00 - is*v11
    sv[i00] = make_double2(c*v00.x + s*v11.y,  c*v00.y - s*v11.x);
    // 01 -> c*v01 - is*v10
    sv[i01] = make_double2(c*v01.x + s*v10.y,  c*v01.y - s*v10.x);
    // 10 -> c*v10 - is*v01
    sv[i10] = make_double2(c*v10.x + s*v01.y,  c*v10.y - s*v01.x);
    // 11 -> c*v11 - is*v00
    sv[i11] = make_double2(c*v11.x + s*v00.y,  c*v11.y - s*v00.x);
}

__global__ void apply_ryy_fused(double2 *sv, long long ns, int q1, int q2, double theta) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ns >> 2) return;

    int lo_bit = min(q1, q2), hi_bit = max(q1, q2);
    long long inner = tid & ((1LL << lo_bit) - 1);
    long long mid   = (tid >> lo_bit) & ((1LL << (hi_bit - lo_bit - 1)) - 1);
    long long outer = tid >> (hi_bit - 1);

    long long i00 = (outer << (hi_bit+1)) | (mid << (lo_bit+1)) | inner;
    long long i01 = i00 | (1LL << lo_bit);
    long long i10 = i00 | (1LL << hi_bit);
    long long i11 = i00 | (1LL << lo_bit) | (1LL << hi_bit);

    double c = cos(theta / 2.0);
    double s = sin(theta / 2.0);

    double2 v00 = sv[i00], v01 = sv[i01], v10 = sv[i10], v11 = sv[i11];

    // 00 -> c*v00 + is*v11  (sign flip vs RXX)
    sv[i00] = make_double2(c*v00.x - s*v11.y,  c*v00.y + s*v11.x);
    // 01 -> c*v01 - is*v10
    sv[i01] = make_double2(c*v01.x + s*v10.y,  c*v01.y - s*v10.x);
    // 10 -> c*v10 - is*v01
    sv[i10] = make_double2(c*v10.x + s*v01.y,  c*v10.y - s*v01.x);
    // 11 -> c*v11 + is*v00  (sign flip vs RXX)
    sv[i11] = make_double2(c*v11.x - s*v00.y,  c*v11.y + s*v00.x);
}

__global__ void apply_rzz_fused(double2 *sv, long long ns, int q1, int q2, double theta) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ns >> 2) return;

    // Extract the 4 indices for this (q1, q2) group
    int lo_bit = min(q1, q2);
    int hi_bit = max(q1, q2);
    long long inner = tid & ((1LL << lo_bit) - 1);
    long long mid   = (tid >> lo_bit) & ((1LL << (hi_bit - lo_bit - 1)) - 1);
    long long outer = tid >> (hi_bit - 1);

    long long i00 = (outer << (hi_bit+1)) | (mid << (lo_bit+1)) | inner;
    long long i01 = i00 | (1LL << lo_bit);
    long long i10 = i00 | (1LL << hi_bit);
    long long i11 = i00 | (1LL << lo_bit) | (1LL << hi_bit);

    double c = cos(theta / 2.0);
    double s = sin(theta / 2.0);

    // same parity (00, 11): multiply by e^{-iθ/2} = c - is
    double2 v00 = sv[i00];
    sv[i00] = make_double2(c*v00.x + s*v00.y,  c*v00.y - s*v00.x);

    double2 v11 = sv[i11];
    sv[i11] = make_double2(c*v11.x + s*v11.y,  c*v11.y - s*v11.x);

    // different parity (01, 10): multiply by e^{+iθ/2} = c + is
    double2 v01 = sv[i01];
    sv[i01] = make_double2(c*v01.x - s*v01.y,  c*v01.y + s*v01.x);

    double2 v10 = sv[i10];
    sv[i10] = make_double2(c*v10.x - s*v10.y,  c*v10.y + s*v10.x);
}

// Applies the full Trotter bond (RXX, RYY, RZZ) for one nearest-neighbor pair.
// The stream parameter lets multiple bonds run concurrently on the GPU.
void apply_trotter_bond(double2 *sv, long long ns, int q1, int q2,
                        double JX, double JY, double JZ, double dt, cudaStream_t stream = 0) {
    int blocks4 = (int)((ns / 4 + BLOCK - 1) / BLOCK);

    // RXX: direct 4x4 matrix on (q1,q2) amplitude group — no full state vector sweep
    apply_rxx_fused<<<blocks4, BLOCK, 0, stream>>>(sv, ns, q1, q2, 2.0 * JX * dt);

    // RYY: same structure as RXX but with opposite phase on corner elements
    apply_ryy_fused<<<blocks4, BLOCK, 0, stream>>>(sv, ns, q1, q2, 2.0 * JY * dt);

    // RZZ: diagonal matrix — same parity gets e^{-iθ/2}, different parity gets e^{+iθ/2}
    apply_rzz_fused<<<blocks4, BLOCK, 0, stream>>>(sv, ns, q1, q2, 2.0 * JZ * dt);
}

void apply_trotter_bond_reverse(double2 *sv, long long ns, int q1, int q2,
                        double JX, double JY, double JZ, double dt, cudaStream_t stream = 0) {
    int blocks4 = (int)((ns / 4 + BLOCK - 1) / BLOCK);

    // Reversed order: RZZ first, then RYY, then RXX
    apply_rzz_fused<<<blocks4, BLOCK, 0, stream>>>(sv, ns, q1, q2, 2.0 * JZ * dt);
    apply_ryy_fused<<<blocks4, BLOCK, 0, stream>>>(sv, ns, q1, q2, 2.0 * JY * dt);
    apply_rxx_fused<<<blocks4, BLOCK, 0, stream>>>(sv, ns, q1, q2, 2.0 * JX * dt);
}

// Extracts the ancilla=1 sector from the full statevector into a smaller buffer.
__global__ void extract_odd_sector(const double2 *sv, double2 *sv_sys, long long ns_sys) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ns_sys) return;
    sv_sys[i] = sv[2*i + 1];
}

// Inserts the system statevector back into the ancilla=1 sector of the full statevector.
__global__ void insert_odd_sector(const double2 *sv_sys, double2 *sv, long long ns_sys) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ns_sys) return;
    sv[2*i + 1] = sv_sys[i];
}


void compute_sk_hk_rank(int nQ, int excBit, int nSteps,
                        double JX, double JY, double JZ,
                        double dt_total,
                        Complex &outS, Complex &outH,
                        int rank) {

    long long ns     = 1LL << (nQ + 1);
    size_t sv_bytes  = ns * sizeof(double2);
    long long ns_sys = 1LL << nQ;
    int blocks_sys   = (int)((ns_sys / 2 + BLOCK - 1) / BLOCK);
    int blocks       = (int)((ns / 2 + BLOCK - 1) / BLOCK);
    int rblocks      = (int)((ns + BLOCK - 1) / BLOCK);

    // Helper: measures the ancilla X expectation value using a GPU reduction.
    auto measure_x = [&](double2 *sv) -> double {
        double *d_val;
        cudaMalloc(&d_val, sizeof(double));
        cudaMemset(d_val, 0, sizeof(double));
        reduce_x_expectation<<<rblocks, BLOCK>>>(sv, ns, d_val);
        cudaDeviceSynchronize();
        double h_val;
        cudaMemcpy(&h_val, d_val, sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_val);
        return h_val;
    };

    // Helper: prepares the Hadamard test state and runs Trotter evolution.
    auto run_trotter = [&]() -> double2* {
        double2 *sv;
        cudaMalloc(&sv, sv_bytes);
        cudaMemset(sv, 0, sv_bytes);

        // Step 0: |0⟩|0⟩^N
        double2 one = {1.0, 0.0};
        cudaMemcpy(sv, &one, sizeof(double2), cudaMemcpyHostToDevice);

        // Step 1: H on ancilla
        apply_hadamard<<<blocks, BLOCK>>>(sv, ns, 0);

        // Step 2: CX anc→excBit+1  (controlled state prep)
        apply_cx<<<blocks, BLOCK>>>(sv, ns, 0, excBit + 1);
        cudaDeviceSynchronize();

        // Step 3: Trotter on ancilla=1 sector only
        double dt_step = dt_total / nSteps;
        size_t sv_sys_bytes = ns_sys * sizeof(double2);
        double2 *sv_sys;
        cudaMalloc(&sv_sys, sv_sys_bytes);

        extract_odd_sector<<<(ns_sys/BLOCK+1), BLOCK>>>(sv, sv_sys, ns_sys);
        cudaDeviceSynchronize();

        for (int j = 0; j < nSteps; j++) {
            if (j % 2 == 0) {
                for (int q = 0; q < nQ - 1; q += 2)
                    apply_trotter_bond(sv_sys, ns_sys, q, q+1, JX, JY, JZ, dt_step, 0);
                cudaDeviceSynchronize();
                for (int q = 1; q < nQ - 1; q += 2)
                    apply_trotter_bond(sv_sys, ns_sys, q, q+1, JX, JY, JZ, dt_step, 0);
                cudaDeviceSynchronize();
            } else {
                for (int q = 1; q < nQ - 1; q += 2)
                    apply_trotter_bond_reverse(sv_sys, ns_sys, q, q+1, JX, JY, JZ, dt_step, 0);
                cudaDeviceSynchronize();
                for (int q = 0; q < nQ - 1; q += 2)
                    apply_trotter_bond_reverse(sv_sys, ns_sys, q, q+1, JX, JY, JZ, dt_step, 0);
                cudaDeviceSynchronize();
            }
        }

        insert_odd_sector<<<(ns_sys/BLOCK+1), BLOCK>>>(sv_sys, sv, ns_sys);
        cudaFree(sv_sys);

        // sv now holds: 1/√2 ( e^{iϕ}|0⟩|0⟩^N + |1⟩U|ψ₀⟩ )
        return sv;
    };

    // Measure the overlap matrix element S(k).
    double2 *sv_s = run_trotter();

    // clone for Im branch before Re branch modifies it
    double2 *sv_s_im;
    cudaMalloc(&sv_s_im, sv_bytes);
    cudaMemcpy(sv_s_im, sv_s, sv_bytes, cudaMemcpyDeviceToDevice);

    // Real part of S: apply anti-controlled un-preparation, then Hadamard, then measure.
    apply_x_gate   <<<blocks, BLOCK>>>(sv_s, ns, 0);
    apply_cx       <<<blocks, BLOCK>>>(sv_s, ns, 0, excBit + 1);
    apply_x_gate   <<<blocks, BLOCK>>>(sv_s, ns, 0);
    apply_hadamard <<<blocks, BLOCK>>>(sv_s, ns, 0);
    cudaDeviceSynchronize();
    double re_s = measure_x(sv_s);
    cudaFree(sv_s);

    // Imaginary part of S: same un-preparation, but rotate into Y basis before measuring.
    apply_x_gate   <<<blocks, BLOCK>>>(sv_s_im, ns, 0);
    apply_cx       <<<blocks, BLOCK>>>(sv_s_im, ns, 0, excBit + 1);
    apply_x_gate   <<<blocks, BLOCK>>>(sv_s_im, ns, 0);
    apply_sdg_gate <<<blocks, BLOCK>>>(sv_s_im, ns, 0);
    apply_hadamard <<<blocks, BLOCK>>>(sv_s_im, ns, 0);
    cudaDeviceSynchronize();
    double im_s = measure_x(sv_s_im);
    cudaFree(sv_s_im);

    outS = Complex(re_s, im_s);

    // Measure the Hamiltonian matrix element H(k) by summing over all bonds
    // and Pauli terms (XX, YY, ZZ).
    outH = Complex(0.0, 0.0);
    double coeffs[3] = {JX, JY, JZ};

    double2 *sv_base = run_trotter();

    for (int b = 0; b < nQ - 1; b++) {
        int q1_sys = b;
        int q2_sys = b + 1;

        for (int pauli = 0; pauli < 3; pauli++) {

            // Real part of H: clone state, apply Pauli to the system sector,
            // then un-prepare and measure.
            double2 *sv_re;
            cudaMalloc(&sv_re, sv_bytes);
            cudaMemcpy(sv_re, sv_base, sv_bytes, cudaMemcpyDeviceToDevice);

            double2 *sv_sys_p;
            cudaMalloc(&sv_sys_p, ns_sys * sizeof(double2));
            extract_odd_sector<<<(ns_sys/BLOCK+1), BLOCK>>>(sv_re, sv_sys_p, ns_sys);
            cudaDeviceSynchronize();

            if (pauli == 0) {
                apply_x_gate<<<blocks_sys, BLOCK>>>(sv_sys_p, ns_sys, q1_sys);
                apply_x_gate<<<blocks_sys, BLOCK>>>(sv_sys_p, ns_sys, q2_sys);
            } else if (pauli == 1) {
                apply_y_gate<<<blocks_sys, BLOCK>>>(sv_sys_p, ns_sys, q1_sys);
                apply_y_gate<<<blocks_sys, BLOCK>>>(sv_sys_p, ns_sys, q2_sys);
            } else {
                apply_z_gate<<<blocks_sys, BLOCK>>>(sv_sys_p, ns_sys, q1_sys);
                apply_z_gate<<<blocks_sys, BLOCK>>>(sv_sys_p, ns_sys, q2_sys);
            }

            insert_odd_sector<<<(ns_sys/BLOCK+1), BLOCK>>>(sv_sys_p, sv_re, ns_sys);
            cudaFree(sv_sys_p);
            cudaDeviceSynchronize();

            // Un-prepare the ancilla and measure the real part.
            apply_x_gate   <<<blocks, BLOCK>>>(sv_re, ns, 0);
            apply_cx       <<<blocks, BLOCK>>>(sv_re, ns, 0, excBit + 1);
            apply_x_gate   <<<blocks, BLOCK>>>(sv_re, ns, 0);
            apply_hadamard <<<blocks, BLOCK>>>(sv_re, ns, 0);
            cudaDeviceSynchronize();
            double re_h = measure_x(sv_re);
            cudaFree(sv_re);

            // Imaginary part of H: same approach but rotate into Y basis before measuring.
            double2 *sv_im;
            cudaMalloc(&sv_im, sv_bytes);
            cudaMemcpy(sv_im, sv_base, sv_bytes, cudaMemcpyDeviceToDevice);

            double2 *sv_sys_p2;
            cudaMalloc(&sv_sys_p2, ns_sys * sizeof(double2));
            extract_odd_sector<<<(ns_sys/BLOCK+1), BLOCK>>>(sv_im, sv_sys_p2, ns_sys);
            cudaDeviceSynchronize();

            if (pauli == 0) {
                apply_x_gate<<<blocks_sys, BLOCK>>>(sv_sys_p2, ns_sys, q1_sys);
                apply_x_gate<<<blocks_sys, BLOCK>>>(sv_sys_p2, ns_sys, q2_sys);
            } else if (pauli == 1) {
                apply_y_gate<<<blocks_sys, BLOCK>>>(sv_sys_p2, ns_sys, q1_sys);
                apply_y_gate<<<blocks_sys, BLOCK>>>(sv_sys_p2, ns_sys, q2_sys);
            } else {
                apply_z_gate<<<blocks_sys, BLOCK>>>(sv_sys_p2, ns_sys, q1_sys);
                apply_z_gate<<<blocks_sys, BLOCK>>>(sv_sys_p2, ns_sys, q2_sys);
            }

            insert_odd_sector<<<(ns_sys/BLOCK+1), BLOCK>>>(sv_sys_p2, sv_im, ns_sys);
            cudaFree(sv_sys_p2);
            cudaDeviceSynchronize();

            // Un-prepare the ancilla with Y-basis rotation, then measure.
            apply_x_gate   <<<blocks, BLOCK>>>(sv_im, ns, 0);
            apply_cx       <<<blocks, BLOCK>>>(sv_im, ns, 0, excBit + 1);
            apply_x_gate   <<<blocks, BLOCK>>>(sv_im, ns, 0);
            apply_sdg_gate <<<blocks, BLOCK>>>(sv_im, ns, 0);
            apply_hadamard <<<blocks, BLOCK>>>(sv_im, ns, 0);
            cudaDeviceSynchronize();
            double im_h = measure_x(sv_im);
            cudaFree(sv_im);

            outH += coeffs[pauli] * Complex(re_h, im_h);
        }
    }

    cudaFree(sv_base);
}

int main(int argc, char **argv) {

    double JX, JY, JZ;
    int nQ;
    {
        std::ifstream f("input.txt");
        if (!f.is_open()) { std::cerr << "Cannot open input.txt\n"; return 1; }
        f >> JX >> JY >> JZ >> nQ;
    }

    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    if (rank == 0)
        std::cout << "JX=" << JX << " JY=" << JY << " JZ=" << JZ << " nQ=" << nQ << "\n";

    if (nranks != 3) {
        if (rank == 0) std::cerr << "Need exactly 3 MPI ranks\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Pin each MPI rank to its own GPU.
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    cudaSetDevice(rank % num_gpus);
    cudaFree(0);

    // Build the single-particle sector Hamiltonian (much smaller than the full 2^nQ matrix).
    int excBit = (nQ / 2)+1;

    CMat H_proj(nQ, std::vector<Complex>(nQ, 0));
    for (int i = 0; i < nQ; i++) {
        int bonds_touching     = (i > 0 ? 1 : 0) + (i < nQ-1 ? 1 : 0);
        int bonds_not_touching = (nQ - 1) - bonds_touching;
        double z_val = bonds_not_touching * JZ - bonds_touching * JZ;
        H_proj[i][i] = Complex(z_val, 0);
        if (i < nQ - 1) {
            H_proj[i][i+1] = Complex(JX + JY, 0);
            H_proj[i+1][i] = Complex(JX + JY, 0);
        }
    }

    CMat S_proj(nQ, std::vector<Complex>(nQ, 0));
    for (int i = 0; i < nQ; i++) S_proj[i][i] = Complex(1, 0);

    // Compute the exact ground state energy via classical diagonalization.
    double E0_exact = solve_gen_eig(H_proj, S_proj, 1e-12);

    // The k=0 overlap and Hamiltonian elements can be computed analytically.
    Complex S0 = Complex(1.0, 0.0);
    int bonds_touching     = (excBit > 0 ? 1 : 0) + (excBit < nQ-1 ? 1 : 0);
    int bonds_not_touching = (nQ - 1) - bonds_touching;
    Complex H0 = Complex(JZ * (bonds_not_touching - bonds_touching), 0.0);

    // Algorithm parameters for the Krylov method.
    double dt = PI / Compute_SpectralNorm(H_proj);
    int krylov_dim    = 4;
    int trotter_steps = 4;

    // Open the output file on rank 0 (stays open for appending results later).
    std::ofstream out;
    if (rank == 0) {
        out.open("qkd_results.txt");
        if (!out.is_open()) {
            std::cerr << "Failed to open qkd_results.txt\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::cout << "Classical exact E0 (single-particle sector): "
                  << std::fixed << std::setprecision(8) << E0_exact << "\n";
        std::cout << "  k=0  S(0)=(" << S0.real() << ", " << S0.imag() << ")"
                  << "  H(0)=(" << H0.real() << ", " << H0.imag() << ")  [analytical]\n";

        out << "JX=" << JX << " JY=" << JY << " JZ=" << JZ << " nQ=" << nQ << "\n";
        out << "excBit=" << excBit << "  (middle qubit, 0-indexed)\n\n";
        out << "Classical exact E0: "
            << std::fixed << std::setprecision(8) << E0_exact << "\n";
        out << "  k=0  S(0)=(" << std::fixed << std::setprecision(8)
            << S0.real() << ", " << S0.imag() << ")"
            << "  H(0)=(" << H0.real() << ", " << H0.imag() << ")  [analytical]\n\n";
    }

    // Distribute the Krylov steps across MPI ranks.
    // Rank 0 handles k=3, rank 1 handles k=1, rank 2 handles k=2.
    auto k_for = [](int r) -> int { return r == 0 ? 3 : r; };
    int my_k = k_for(rank);
    Complex my_S(0, 0), my_H(0, 0);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start_time = std::chrono::high_resolution_clock::now();

    // Run the quantum simulation for this rank's assigned Krylov step.
    compute_sk_hk_rank(nQ, excBit, trotter_steps,
                       JX, JY, JZ,
                       my_k * dt,
                       my_S, my_H, rank);

    cudaDeviceSynchronize();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Print each rank's result one at a time to avoid interleaving.
    for (int r = 0; r < nranks; r++) {
        if (rank == r) {
            std::cout << "  k=" << my_k
                      << "  S(k)=(" << std::fixed << std::setprecision(8)
                      << my_S.real() << ", " << my_S.imag() << ")"
                      << "  H(k)=(" << my_H.real() << ", " << my_H.imag() << ")"
                      << "  Time: " << duration.count() << " ms\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Gather all results to rank 0 via MPI.
    double send4[4]   = { my_S.real(), my_S.imag(), my_H.real(), my_H.imag() };
    double recv16[16] = {};
    MPI_Gather(send4, 4, MPI_DOUBLE, recv16, 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // On rank 0, assemble the Toeplitz Krylov matrices and solve for the ground state.
    if (rank == 0) {
        Complex SR[4] = {}, HR[4] = {};

        // k=0 analytical
        SR[0] = S0;
        HR[0] = H0;

        // rank 0 computed k=3
        SR[3] = Complex(recv16[0], recv16[1]);
        HR[3] = Complex(recv16[2], recv16[3]);

        // ranks 1,2 computed k=1,2
        for (int src = 1; src <= 2; src++) {
            int kk = k_for(src);
            SR[kk] = Complex(recv16[src*4+0], recv16[src*4+1]);
            HR[kk] = Complex(recv16[src*4+2], recv16[src*4+3]);
        }

        // Print all first-row elements
        for (int k = 0; k < krylov_dim; k++) {
            std::cout << "  k=" << k
                      << "  S=(" << std::fixed << std::setprecision(8)
                      << SR[k].real() << ", " << SR[k].imag() << ")"
                      << "  H=(" << HR[k].real() << ", " << HR[k].imag() << ")\n";
            out << "  k=" << k
                << "  S=(" << std::fixed << std::setprecision(8)
                << SR[k].real() << ", " << SR[k].imag() << ")"
                << "  H=(" << HR[k].real() << ", " << HR[k].imag() << ")\n";
        }
        std::cout << "\n";
        out << "\n";

        // Krylov convergence sweep K=1..krylov_dim
        for (int K = 1; K <= krylov_dim; K++) {
            CMat Sk(K, std::vector<Complex>(K));
            CMat Hk(K, std::vector<Complex>(K));
            for (int i = 0; i < K; i++) {
                for (int j = 0; j < K; j++) {
                    int d = std::abs(i - j);
                    Sk[i][j] = (i <= j) ? SR[d] : std::conj(SR[d]);
                    Hk[i][j] = (i <= j) ? HR[d] : std::conj(HR[d]);
                }
                // Tikhonov regularisation — keeps S invertible at higher K
                Sk[i][i] += Complex(1e-8, 0);
            }
            double e0  = solve_gen_eig(Hk, Sk, 1e-3);
            double err = std::abs(e0 - E0_exact);
            std::cout << "  K=" << K
                      << "  E0=" << std::fixed << std::setprecision(8) << e0
                      << "  |err|=" << err << "\n";
            out << "  K=" << K
                << "  E0=" << std::fixed << std::setprecision(8) << e0
                << "  |err|=" << err << "\n";
        }

        std::cout << "\nResults saved to qkd_results.txt\n";
        out.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}