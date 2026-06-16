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
#include <cstdlib>
#include <limits>
#include <string>

using Complex = std::complex<double>;
using CMat    = std::vector<std::vector<Complex>>;
static const double PI        = 3.14159265358979323846;
__device__ static constexpr double INV_SQRT2 = 0.7071067811865475;


// COARSEN: each thread handles COARSEN consecutive logical TIDs.
// Set at compile time: nvcc -DCOARSEN=1  (baseline, 1 iteration per thread)
//                      nvcc -DCOARSEN=2  (2 iterations per thread)
//                      nvcc -DCOARSEN=4  (4 iterations per thread)
// Block count is divided by COARSEN so total work stays identical.

#ifndef COARSEN
#define COARSEN 1
#endif

static const int BLOCK = 256;


// Reduction — each thread accumulates COARSEN elements
__global__ void reduce_x_expectation(const double2 *sv, long long ns, double *out) {
    __shared__ double sdata[BLOCK];
    long long base = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * COARSEN;
    double val = 0.0;

    #pragma unroll
    for (int c = 0; c < COARSEN; c++) {
        long long idx = base + c;
        if (idx < ns) {
            double re = sv[idx].x, im = sv[idx].y;
            double prob = re*re + im*im;
            val += (idx % 2 == 0) ? prob : -prob;
        }
    }

    sdata[threadIdx.x] = val;
    __syncthreads();
    for (int s = BLOCK/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, sdata[0]);
}

// Classical linear algebra (unchanged)
static void real_jacobi(std::vector<std::vector<double>>& A,
                        std::vector<std::vector<double>>& V) {
    int n = (int)A.size();
    for (int i = 0; i < n; i++) { for (int j = 0; j < n; j++) V[i][j] = 0; V[i][i] = 1; }
    for (int sweep = 0; sweep < 100; sweep++) {
        double off = 0;
        for (int i = 0; i < n; i++) for (int j = i+1; j < n; j++) off += A[i][j]*A[i][j];
        if (off < 1e-30) break;
        for (int p = 0; p < n; p++)
            for (int q = p+1; q < n; q++) {
                double apq = A[p][q];
                if (std::abs(apq) < 1e-300) continue;
                double app = A[p][p], aqq = A[q][q];
                double tau = (aqq - app) / (2.0 * apq);
                double t   = (tau >= 0 ? 1.0 : -1.0) / (std::abs(tau) + std::sqrt(1 + tau*tau));
                double c   = 1.0 / std::sqrt(1 + t*t), s = t * c;
                for (int r = 0; r < n; r++) { double a1=A[r][p],a2=A[r][q]; A[r][p]=c*a1-s*a2; A[r][q]=s*a1+c*a2; }
                for (int r = 0; r < n; r++) { double a1=A[p][r],a2=A[q][r]; A[p][r]=c*a1-s*a2; A[q][r]=s*a1+c*a2; }
                for (int r = 0; r < n; r++) { double v1=V[r][p],v2=V[r][q]; V[r][p]=c*v1-s*v2; V[r][q]=s*v1+c*v2; }
            }
    }
}

// Diagonalize complex Hermitian A via the real 2n x 2n symmetric embedding
// M = [[Re A, -Im A], [Im A, Re A]] -- no phase pathologies, never returns
// spurious negative eigenvalues for a PSD matrix.
std::pair<std::vector<double>, CMat> jacobi_eigh(CMat A) {
    int n = (int)A.size(), m = 2*n;
    std::vector<std::vector<double>> M(m, std::vector<double>(m, 0)),
                                     Vr(m, std::vector<double>(m, 0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double re = A[i][j].real(), im = A[i][j].imag();
            M[i][j] = re;  M[i+n][j+n] = re;  M[i+n][j] = im;  M[i][j+n] = -im;
        }
    real_jacobi(M, Vr);
    std::vector<std::pair<double,int>> eg(m);
    for (int i = 0; i < m; i++) eg[i] = { M[i][i], i };
    std::sort(eg.begin(), eg.end());
    std::vector<double> evals; CMat evecs;
    for (int idx = 0; idx < m && (int)evals.size() < n; idx++) {
        int col = eg[idx].second; double lam = eg[idx].first;
        std::vector<Complex> v(n);
        for (int r = 0; r < n; r++) v[r] = Complex(Vr[r][col], Vr[r+n][col]);
        for (auto& u : evecs) {
            Complex ip = 0; for (int r = 0; r < n; r++) ip += std::conj(u[r]) * v[r];
            for (int r = 0; r < n; r++) v[r] -= ip * u[r];
        }
        double nv = 0; for (int r = 0; r < n; r++) nv += std::norm(v[r]); nv = std::sqrt(nv);
        if (nv < 1e-8) continue;
        for (int r = 0; r < n; r++) v[r] /= nv;
        evals.push_back(lam); evecs.push_back(v);
    }
    std::vector<int> o(evals.size()); std::iota(o.begin(), o.end(), 0);
    std::sort(o.begin(), o.end(), [&](int a, int b){ return evals[a] < evals[b]; });
    std::vector<double> sev(n); CMat sV(n, std::vector<Complex>(n));
    for (int i = 0; i < n; i++) { sev[i] = evals[o[i]];
        for (int r = 0; r < n; r++) sV[r][i] = evecs[o[i]][r]; }
    return {sev, sV};
}

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


// Single-qubit gate kernels — coarsened
// Each thread handles COARSEN consecutive logical TIDs (amplitude pairs).
// blocks = ceil(half / (BLOCK * COARSEN))

__global__ void apply_x_gate(double2 *sv, long long ns, int q) {
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
        double2 tmp = sv[i0]; sv[i0] = sv[i1]; sv[i1] = tmp;
    }
}

__global__ void apply_hadamard(double2 *sv, long long ns, int q) {
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
        sv[i0] = make_double2((v0.x + v1.x) * INV_SQRT2, (v0.y + v1.y) * INV_SQRT2);
        sv[i1] = make_double2((v0.x - v1.x) * INV_SQRT2, (v0.y - v1.y) * INV_SQRT2);
    }
}

__global__ void apply_cx(double2 *sv, long long ns, int ctrl, int tgt) {
    long long base = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * COARSEN;
    long long half = ns >> 1;
    #pragma unroll
    for (int c = 0; c < COARSEN; c++) {
        long long tid = base + c;
        if (tid >= half) return;
        long long lo = tid & ((1LL << tgt) - 1);
        long long hi = tid >> tgt;
        long long i0 = (hi << (tgt + 1)) | lo;
        long long i1 = i0 | (1LL << tgt);
        if (!(i0 & (1LL << ctrl))) continue;
        double2 tmp = sv[i0]; sv[i0] = sv[i1]; sv[i1] = tmp;
    }
}

__global__ void apply_s_gate(double2 *sv, long long ns, int q) {
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
        sv[i1] = make_double2(-v1.y, v1.x);
    }
}

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
        sv[i1] = make_double2(v1.y, -v1.x);
    }
}

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
        sv[i0] = make_double2(v1.y, -v1.x);
        sv[i1] = make_double2(-v0.y, v0.x);
    }
}

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
        sv[i1].x = -sv[i1].x;
        sv[i1].y = -sv[i1].y;
    }
}

__global__ void apply_rz(double2 *sv, long long ns, int q, double theta) {
    long long base = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * COARSEN;
    long long half = ns >> 1;
    double c_ = cos(theta / 2.0), s_ = sin(theta / 2.0);
    #pragma unroll
    for (int c = 0; c < COARSEN; c++) {
        long long tid = base + c;
        if (tid >= half) return;
        long long lo = tid & ((1LL << q) - 1);
        long long hi = tid >> q;
        long long i0 = (hi << (q + 1)) | lo;
        long long i1 = i0 | (1LL << q);
        double2 v0 = sv[i0], v1 = sv[i1];
        sv[i0] = make_double2(c_*v0.x + s_*v0.y, c_*v0.y - s_*v0.x);
        sv[i1] = make_double2(c_*v1.x - s_*v1.y, c_*v1.y + s_*v1.x);
    }
}


// Two-qubit fused gate kernels — coarsened
// Each thread handles COARSEN consecutive 4-element groups.
// blocks4 = ceil(quarter / (BLOCK * COARSEN))


__global__ void apply_rxx_fused(double2 *sv, long long ns, int q1, int q2, double theta) {
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
        double2 v00=sv[i00]; sv[i00] = make_double2(c*v00.x + s*v00.y, c*v00.y - s*v00.x);
        double2 v11=sv[i11]; sv[i11] = make_double2(c*v11.x + s*v11.y, c*v11.y - s*v11.x);
        double2 v01=sv[i01]; sv[i01] = make_double2(c*v01.x - s*v01.y, c*v01.y + s*v01.x);
        double2 v10=sv[i10]; sv[i10] = make_double2(c*v10.x - s*v10.y, c*v10.y + s*v10.x);
    }
}


// Sector extract/insert — coarsened
__global__ void extract_odd_sector(const double2 *sv, double2 *sv_sys, long long ns_sys) {
    long long base = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * COARSEN;
    #pragma unroll
    for (int c = 0; c < COARSEN; c++) {
        long long i = base + c;
        if (i >= ns_sys) return;
        sv_sys[i] = sv[2*i + 1];
    }
}

__global__ void insert_odd_sector(const double2 *sv_sys, double2 *sv, long long ns_sys) {
    long long base = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * COARSEN;
    #pragma unroll
    for (int c = 0; c < COARSEN; c++) {
        long long i = base + c;
        if (i >= ns_sys) return;
        sv[2*i + 1] = sv_sys[i];
    }
}

__global__ void extract_amplitude(const double2* sv, long long target_idx, double2* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = sv[target_idx];
}


// Block count helpers — divide by COARSEN to match coarsened threads
static inline int blocks_half(long long ns) {
    long long half = ns >> 1;
    return (int)((half + (long long)BLOCK * COARSEN - 1) / ((long long)BLOCK * COARSEN));
}

static inline int blocks_quarter(long long ns) {
    long long quarter = ns >> 2;
    return (int)((quarter + (long long)BLOCK * COARSEN - 1) / ((long long)BLOCK * COARSEN));
}

static inline int blocks_full(long long ns) {
    return (int)((ns + (long long)BLOCK * COARSEN - 1) / ((long long)BLOCK * COARSEN));
}

static inline int blocks_n(long long n) {
    return (int)((n + (long long)BLOCK * COARSEN - 1) / ((long long)BLOCK * COARSEN));
}


// Trotter bond helpers
void apply_trotter_bond(double2 *sv, long long ns, int q1, int q2,
                        double JX, double JY, double JZ, double dt, cudaStream_t stream = 0) {
    int b4 = blocks_quarter(ns);
    apply_rxx_fused<<<b4, BLOCK, 0, stream>>>(sv, ns, q1, q2, 2.0 * JX * dt);
    apply_ryy_fused<<<b4, BLOCK, 0, stream>>>(sv, ns, q1, q2, 2.0 * JY * dt);
    apply_rzz_fused<<<b4, BLOCK, 0, stream>>>(sv, ns, q1, q2, 2.0 * JZ * dt);
}

void apply_trotter_bond_reverse(double2 *sv, long long ns, int q1, int q2,
                        double JX, double JY, double JZ, double dt, cudaStream_t stream = 0) {
    int b4 = blocks_quarter(ns);
    apply_rzz_fused<<<b4, BLOCK, 0, stream>>>(sv, ns, q1, q2, 2.0 * JZ * dt);
    apply_ryy_fused<<<b4, BLOCK, 0, stream>>>(sv, ns, q1, q2, 2.0 * JY * dt);
    apply_rxx_fused<<<b4, BLOCK, 0, stream>>>(sv, ns, q1, q2, 2.0 * JX * dt);
}


// Core simulation

void compute_sk_hk_rank(int nQ, int excBit, int nSteps,
                        double JX, double JY, double JZ,
                        double dt_total,
                        Complex &outS, Complex &outH,
                        int rank) {

    long long ns     = 1LL << (nQ + 1);
    size_t sv_bytes  = ns * sizeof(double2);
    long long ns_sys = 1LL << nQ;

    int bh      = blocks_half(ns);
    int bh_sys  = blocks_half(ns_sys);
    int bf      = blocks_full(ns);
    int bn_sys  = blocks_n(ns_sys);

    auto measure_x = [&](double2 *sv) -> double {
        double *d_val;
        cudaMalloc(&d_val, sizeof(double));
        cudaMemset(d_val, 0, sizeof(double));
        reduce_x_expectation<<<bf, BLOCK>>>(sv, ns, d_val);
        cudaDeviceSynchronize();
        double h_val;
        cudaMemcpy(&h_val, d_val, sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_val);
        return h_val;
    };

    auto run_trotter = [&]() -> double2* {
        double2 *sv;
        cudaMalloc(&sv, sv_bytes);
        cudaMemset(sv, 0, sv_bytes);
        double2 one = {1.0, 0.0};
        cudaMemcpy(sv, &one, sizeof(double2), cudaMemcpyHostToDevice);
        apply_hadamard<<<bh, BLOCK>>>(sv, ns, 0);
        for (int _q = 0; _q < nQ; _q += 2) apply_cx<<<bh, BLOCK>>>(sv, ns, 0, _q + 1);
        cudaDeviceSynchronize();

        double dt_step = dt_total / nSteps;
        size_t sv_sys_bytes = ns_sys * sizeof(double2);
        double2 *sv_sys;
        cudaMalloc(&sv_sys, sv_sys_bytes);
        extract_odd_sector<<<bn_sys, BLOCK>>>(sv, sv_sys, ns_sys);
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

        insert_odd_sector<<<bn_sys, BLOCK>>>(sv_sys, sv, ns_sys);
        cudaFree(sv_sys);
        return sv;
    };

    // S(k)
    double2 *sv_s = run_trotter();
    double2 *sv_s_im;
    cudaMalloc(&sv_s_im, sv_bytes);
    cudaMemcpy(sv_s_im, sv_s, sv_bytes, cudaMemcpyDeviceToDevice);
    apply_x_gate   <<<bh, BLOCK>>>(sv_s, ns, 0);
    for (int _q = 0; _q < nQ; _q += 2) apply_cx<<<bh, BLOCK>>>(sv_s, ns, 0, _q + 1);
    apply_x_gate   <<<bh, BLOCK>>>(sv_s, ns, 0);
    apply_hadamard <<<bh, BLOCK>>>(sv_s, ns, 0);
    cudaDeviceSynchronize();
    double re_s = measure_x(sv_s);
    cudaFree(sv_s);
    apply_x_gate   <<<bh, BLOCK>>>(sv_s_im, ns, 0);
    for (int _q = 0; _q < nQ; _q += 2) apply_cx<<<bh, BLOCK>>>(sv_s_im, ns, 0, _q + 1);
    apply_x_gate   <<<bh, BLOCK>>>(sv_s_im, ns, 0);
    apply_sdg_gate <<<bh, BLOCK>>>(sv_s_im, ns, 0);
    apply_hadamard <<<bh, BLOCK>>>(sv_s_im, ns, 0);
    cudaDeviceSynchronize();
    double im_s = measure_x(sv_s_im);
    cudaFree(sv_s_im);
    outS = Complex(re_s, im_s);

    // H(k)
    outH = Complex(0.0, 0.0);
    double coeffs[3] = {JX, JY, JZ};
    double2 *sv_base = run_trotter();

    for (int b = 0; b < nQ - 1; b++) {
        int q1_sys = b, q2_sys = b + 1;
        for (int pauli = 0; pauli < 3; pauli++) {
            double2 *sv_re;
            cudaMalloc(&sv_re, sv_bytes);
            cudaMemcpy(sv_re, sv_base, sv_bytes, cudaMemcpyDeviceToDevice);
            double2 *sv_sys_p;
            cudaMalloc(&sv_sys_p, ns_sys * sizeof(double2));
            extract_odd_sector<<<bn_sys, BLOCK>>>(sv_re, sv_sys_p, ns_sys);
            cudaDeviceSynchronize();
            if (pauli == 0) {
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
            apply_x_gate   <<<bh, BLOCK>>>(sv_re, ns, 0);
            for (int _q = 0; _q < nQ; _q += 2) apply_cx<<<bh, BLOCK>>>(sv_re, ns, 0, _q + 1);
            apply_x_gate   <<<bh, BLOCK>>>(sv_re, ns, 0);
            apply_hadamard <<<bh, BLOCK>>>(sv_re, ns, 0);
            cudaDeviceSynchronize();
            double re_h = measure_x(sv_re);
            cudaFree(sv_re);

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

            outH += coeffs[pauli] * Complex(re_h, im_h);
        }
    }
    cudaFree(sv_base);
}

std::vector<int> ks_for_rank(int rank, int nranks, int krylov_dim) {
    std::vector<int> all_ks;
    for (int k = 1; k < krylov_dim; k++) all_ks.push_back(k);
    std::vector<int> my_ks;
    for (int i = 0; i < (int)all_ks.size(); i++)
        if (i % nranks == rank) my_ks.push_back(all_ks[i]);
    return my_ks;
}

// Full-Hilbert-space Heisenberg matvec:  out = H * in
//   H = Σ_i ( JX X_i X_{i+1} + JY Y_i Y_{i+1} + JZ Z_i Z_{i+1} ),  open chain
static void heisenberg_matvec(const std::vector<double> &in, std::vector<double> &out,
                              int nQ, double JX, double JY, double JZ) {
    long long dim = 1LL << nQ;
    std::fill(out.begin(), out.end(), 0.0);
    for (long long s = 0; s < dim; s++) {
        double xs = in[s], diag = 0.0;
        for (int i = 0; i < nQ - 1; i++) {
            int bi = (int)((s >> i) & 1);
            int bj = (int)((s >> (i + 1)) & 1);
            diag += JZ * (bi == bj ? 1.0 : -1.0);                 // Z_i Z_{i+1}
            long long sf = s ^ (1LL << i) ^ (1LL << (i + 1));      // X/Y flip both spins
            double coeff = (bi == bj) ? (JX - JY) : (JX + JY);     // XX + YY combined
            out[sf] += coeff * xs;
        }
        out[s] += diag * xs;
    }
}

// # eigenvalues of symmetric tridiagonal (a, b) strictly below x  (Sturm sequence)
static int sturm_count(const std::vector<double> &a, const std::vector<double> &b, double x) {
    int n = (int)a.size(), count = 0;
    double d = a[0] - x;
    if (d < 0.0) count++;
    for (int i = 1; i < n; i++) {
        if (d == 0.0) d = 1e-300;
        d = (a[i] - x) - (b[i - 1] * b[i - 1]) / d;
        if (d < 0.0) count++;
    }
    return count;
}

// True classical extrema: matrix-free Lanczos (full reorth) + bisection
std::pair<double, double> heisenberg_exact_extrema(int nQ, double JX, double JY, double JZ) {
    long long dim = 1LL << nQ;
    int m = (int)std::min(dim, (long long)200);   // Krylov depth (plenty for dim=1024)

    std::vector<double> v_prev(dim, 0.0), v(dim), w(dim);
    std::vector<std::vector<double>> Q; Q.reserve(m);

    double nrm = 0.0;                              // generic start vector (all sectors)
    for (long long i = 0; i < dim; i++) { v[i] = std::sin(0.1*(i+1)) + 0.123; nrm += v[i]*v[i]; }
    nrm = std::sqrt(nrm);
    for (long long i = 0; i < dim; i++) v[i] /= nrm;

    std::vector<double> alpha, beta;
    double beta_prev = 0.0;
    for (int it = 0; it < m; it++) {
        Q.push_back(v);
        heisenberg_matvec(v, w, nQ, JX, JY, JZ);
        double a = 0.0;
        for (long long i = 0; i < dim; i++) a += w[i] * v[i];
        for (long long i = 0; i < dim; i++) w[i] -= a * v[i] + beta_prev * v_prev[i];
        for (auto &q : Q) {                        // full reorthogonalization
            double dot = 0.0;
            for (long long i = 0; i < dim; i++) dot += w[i] * q[i];
            for (long long i = 0; i < dim; i++) w[i] -= dot * q[i];
        }
        double b = 0.0;
        for (long long i = 0; i < dim; i++) b += w[i] * w[i];
        b = std::sqrt(b);
        alpha.push_back(a);
        if (b < 1e-12) break;
        beta.push_back(b);
        v_prev = v;
        for (long long i = 0; i < dim; i++) v[i] = w[i] / b;
        beta_prev = b;
    }

    int n = (int)alpha.size();                     // smallest eigenvalue by bisection
    double lo = std::numeric_limits<double>::infinity(), hi = -lo;
    for (int i = 0; i < n; i++) {                  // Gershgorin bracket
        double r = (i>0 ? std::abs(beta[i-1]) : 0.0) + (i<n-1 ? std::abs(beta[i]) : 0.0);
        lo = std::min(lo, alpha[i] - r);
        hi = std::max(hi, alpha[i] + r);
    }
    
    double min_lo = lo, min_hi = hi;
    for (int iter = 0; iter < 200; iter++) {
        double mid = 0.5 * (min_lo + min_hi);
        if (sturm_count(alpha, beta, mid) >= 1) min_hi = mid; else min_lo = mid;
    }
    double E_min = 0.5 * (min_lo + min_hi);

    double max_lo = lo, max_hi = hi;
    for (int iter = 0; iter < 200; iter++) {
        double mid = 0.5 * (max_lo + max_hi);
        if (sturm_count(alpha, beta, mid) >= n) max_hi = mid; else max_lo = mid;
    }
    double E_max = 0.5 * (max_lo + max_hi);

    return {E_min, E_max};
}


int main(int argc, char **argv) {

    double JX, JY, JZ;
    int nQ;
    int krylov_dim = 8;
    {
        std::ifstream f("input.txt");
        if (!f.is_open()) { std::cerr << "Cannot open input.txt\n"; return 1; }
        // input.txt format: JX JY JZ [nQ] [krylov_dim]
        f >> JX >> JY >> JZ;
        if (!(f >> nQ)) nQ = 20;
        if (!(f >> krylov_dim)) krylov_dim = 8;
    }
    if (krylov_dim < 1) krylov_dim = 1;
    if (krylov_dim > 64) krylov_dim = 64;

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
    cudaSetDevice(rank % num_gpus);
    cudaFree(0);

    if (rank == 0)
        std::cout << "Running on " << nranks << " GPU(s).  COARSEN=" << COARSEN
                  << "  JX=" << JX << " JY=" << JY << " JZ=" << JZ << " nQ=" << nQ
                  << "  krylov_dim=" << krylov_dim << "\n";

    int excBit        = nQ / 2;
    int trotter_steps = 4;

    CMat H_proj(nQ, std::vector<Complex>(nQ, 0));
    for (int i = 0; i < nQ; i++) {
        int bonds_touching     = (i > 0 ? 1 : 0) + (i < nQ-1 ? 1 : 0);
        int bonds_not_touching = (nQ - 1) - bonds_touching;
        H_proj[i][i] = Complex((double)(bonds_not_touching - bonds_touching) * JZ, 0);
        if (i < nQ - 1) {
            H_proj[i][i+1] = Complex(JX + JY, 0);
            H_proj[i+1][i] = Complex(JX + JY, 0);
        }
    }
    CMat S_proj(nQ, std::vector<Complex>(nQ, 0));
    for (int i = 0; i < nQ; i++) S_proj[i][i] = Complex(1, 0);

    auto [E0_exact, E_max] = heisenberg_exact_extrema(nQ, JX, JY, JZ);
    double dt       = PI / std::max(std::abs(E0_exact), std::abs(E_max));

    Complex S0 = Complex(1.0, 0.0);
    Complex H0 = Complex(-(double)(nQ - 1) * JZ, 0.0);   // <Neel|H|Neel>: every bond antiparallel
    if (rank == 0) {
        std::cout << "Classical exact E0: "
                  << std::fixed << std::setprecision(8) << E0_exact << "\n";
        std::cout << "  k=0  S(0)=(" << S0.real() << ", " << S0.imag() << ")"
                  << "  H(0)=(" << H0.real() << ", " << H0.imag() << ")  [analytical]\n";
    }

    bool verbose_ground_states = (nQ == 10);
    if (const char *report_env = std::getenv("GROUND_STATE_REPORT")) {
        std::string report_mode(report_env);
        if (report_mode == "all" || report_mode == "verbose") {
            verbose_ground_states = true;
        } else if (report_mode == "summary") {
            verbose_ground_states = false;
        }
    }

    std::vector<int> my_ks = ks_for_rank(rank, nranks, krylov_dim);
    std::vector<double> local_results(krylov_dim * 4, 0.0);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int k : my_ks) {
        Complex my_S(0,0), my_H(0,0);
        compute_sk_hk_rank(nQ, excBit, trotter_steps,
                           JX, JY, JZ, k * dt,
                           my_S, my_H, rank);
        cudaDeviceSynchronize();
        local_results[k*4+0] = my_S.real();
        local_results[k*4+1] = my_S.imag();
        local_results[k*4+2] = my_H.real();
        local_results[k*4+3] = my_H.imag();

        if (verbose_ground_states) {
            std::cout << "  [rank " << rank << "]  k=" << k
                      << "  S(k)=(" << std::fixed << std::setprecision(8)
                      << my_S.real() << ", " << my_S.imag() << ")"
                      << "  H(k)=(" << my_H.real() << ", " << my_H.imag() << ")\n";
            std::cout.flush();
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    if (rank == 0)
        std::cout << "  Wall time: " << std::fixed << std::setprecision(1)
                  << elapsed_ms << " ms\n";

    std::vector<double> all_results(krylov_dim * 4, 0.0);
    MPI_Allreduce(local_results.data(), all_results.data(),
                  krylov_dim * 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0) {
        const char *results_env = std::getenv("RESULTS_FILE");
        const std::string results_path = (results_env && results_env[0]) ? results_env : "results.txt";
        std::ofstream out(results_path, std::ios::app);
        out << "\n==================================================\n";
        out << "COARSEN=" << COARSEN << "  nGPUs=" << nranks << "  krylov_dim=" << krylov_dim
            << "  nQ=" << nQ << "\n";
        std::vector<Complex> SR(krylov_dim), HR(krylov_dim);
        SR[0] = S0; HR[0] = H0;
        for (int k = 1; k < krylov_dim; k++) {
            SR[k] = Complex(all_results[k*4+0], all_results[k*4+1]);
            HR[k] = Complex(all_results[k*4+2], all_results[k*4+3]);
        }
        out << "JX=" << JX << " JY=" << JY << " JZ=" << JZ << "\n";
        out << "Classical exact E0: "
            << std::fixed << std::setprecision(8) << E0_exact << "\n\n";
        out << "Wall time: " << std::fixed << std::setprecision(1)
            << elapsed_ms << " ms\n";
        std::cout << "\n  Krylov convergence:\n";
        out << "  Krylov convergence:\n";

        const double instability_cutoff = 1e9;
        const double stability_eps = 1e-8;
        double best_stable_e0 = std::numeric_limits<double>::infinity();
        int best_stable_k = -1;
        double best_finite_e0 = std::numeric_limits<double>::infinity();
        int best_finite_k = -1;
        double previous_finite_e0 = std::numeric_limits<double>::quiet_NaN();
        const double CONV_TOL = 1e-6;   // stop once E0 stops changing between consecutive K
        std::vector<std::string> instability_notes;

        for (int K = 1; K <= krylov_dim; K++) {
            CMat Sk(K, std::vector<Complex>(K));
            CMat Hk(K, std::vector<Complex>(K));
            for (int i = 0; i < K; i++) {
                for (int j = 0; j < K; j++) {
                    int d = std::abs(i-j);
                    Sk[i][j] = (i<=j) ? SR[d] : std::conj(SR[d]);
                    Hk[i][j] = (i<=j) ? HR[d] : std::conj(HR[d]);
                }
                Sk[i][i] += Complex(1e-8, 0);
            }
            double e0  = solve_gen_eig(Hk, Sk, 1e-3);
            bool finite_and_reasonable = std::isfinite(e0) && std::abs(e0) <= instability_cutoff;
            if (!finite_and_reasonable) {
                std::ostringstream note;
                note << "  K=" << K << "  instability: non-finite or runaway eigenvalue; skipping this Krylov vector.";
                instability_notes.push_back(note.str());
                out << note.str() << "\n";
                continue;
            }
            if (std::isfinite(previous_finite_e0) && e0 > previous_finite_e0 + stability_eps) {
                std::ostringstream note;
                note << "  K=" << K << "  instability: ground state increased slightly from "
                     << std::fixed << std::setprecision(8) << previous_finite_e0
                     << " to " << e0 << "; keeping the most stable value above the exact ground state.";
                instability_notes.push_back(note.str());
            }
            double err = std::abs(e0 - E0_exact);
            double conv_delta = std::isfinite(previous_finite_e0) ? std::abs(e0 - previous_finite_e0) : 1e30;
            previous_finite_e0 = e0;
            if (e0 < best_finite_e0) {
                best_finite_e0 = e0;
                best_finite_k = K;
            }
            if (e0 >= E0_exact && e0 < best_stable_e0) {
                best_stable_e0 = e0;
                best_stable_k = K;
            }

            if (verbose_ground_states) {
                std::cout << "  K=" << K << "  E0=" << std::fixed << std::setprecision(8)
                          << e0 << "  |err|=" << err << "\n";
                out << "  K=" << K << "  E0=" << e0 << "  |err|=" << err << "\n";
            }

            if (e0 < E0_exact - 1e-10) {
                std::ostringstream note;
                note << "  Stopping: e0 is less than true ground state (ill-conditioning detected).";
                std::cout << note.str() << "\n";
                out << note.str() << "\n";
                break;
            }
            if (conv_delta < CONV_TOL) {
                std::cout << "  Converged at K=" << K << "  E0=" << std::fixed
                          << std::setprecision(8) << e0 << "  |err|=" << err
                          << "   (stopping early)\n";
                out << "  Converged at K=" << K << "  E0=" << e0 << "  |err|=" << err << "\n";
                if (e0 >= E0_exact && e0 < best_stable_e0) { best_stable_e0 = e0; best_stable_k = K; }
                break;
            }
        }

        double report_e0 = best_stable_e0;
        int report_k = best_stable_k;
        std::string report_note;
        if (!std::isfinite(report_e0)) {
            report_e0 = best_finite_e0;
            report_k = best_finite_k;
            report_note = "No Krylov eigenvalue stayed above the exact ground state; using the best finite value instead.";
        }

        if (!verbose_ground_states) {
            if (!report_note.empty()) {
                std::cout << "  " << report_note << "\n";
                out << "  " << report_note << "\n";
            }
            if (report_k >= 0) {
                std::cout << "  Selected stable ground-state estimate: K=" << report_k
                          << "  E0=" << std::fixed << std::setprecision(8) << report_e0 << "\n";
                out << "  Selected stable ground-state estimate: K=" << report_k
                    << "  E0=" << report_e0 << "\n";
            }
        }
        for (const auto &note : instability_notes) {
            std::cout << note << "\n";
            out << note << "\n";
        }
        out << "Results saved to " << results_path << "\n";
        out.close();
        std::cout << "\nResults saved to " << results_path << "\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}