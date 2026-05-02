#include <mpi.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <nvtx3/nvToolsExt.h>   // NVTX markers — nsys reads these as named lanes

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

// This is the block size we sweep for the strong scaling experiment.
// You can override it at compile time with: nvcc -DBLOCK_SIZE=256 ...
// The run.sh script recompiles with different values and records the timing.
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
static const int BLOCK = BLOCK_SIZE;

// Colors for the NVTX timeline in Nsight Systems.
// Each phase of the algorithm gets its own color so you can tell them apart.
#define NVTX_COLOR_TROTTER   0xFF4477AA   // blue for Trotter evolution
#define NVTX_COLOR_MEASURE_S 0xFF44AA77   // green for S(k) measurement
#define NVTX_COLOR_MEASURE_H 0xFFAA7744   // amber for H(k) measurement
#define NVTX_COLOR_MPI       0xFFAA4444   // red for MPI communication
#define NVTX_COLOR_BOND      0xFF8877CC   // purple for individual bond gates

// Convenience wrappers so call sites stay readable
static inline nvtxRangeId_t nvtx_push(const char* label, uint32_t color) {
    nvtxEventAttributes_t attr = {};
    attr.version       = NVTX_VERSION;
    attr.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType     = NVTX_COLOR_ARGB;
    attr.color         = color;
    attr.messageType   = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = label;
    return nvtxRangeStartEx(&attr);
}
static inline void nvtx_pop(nvtxRangeId_t id) { nvtxRangeEnd(id); }

// Simple CUDA event timer wrapper. Call .start() before a kernel launch
// and .stop() after, then .ms() to get the elapsed time in milliseconds.
// We use GPU-side event timing here because it excludes CPU overhead and
// driver latency, giving us the actual kernel execution time.
struct CudaTimer {
    cudaEvent_t t0, t1;
    CudaTimer()  { cudaEventCreate(&t0); cudaEventCreate(&t1); }
    ~CudaTimer() { cudaEventDestroy(t0); cudaEventDestroy(t1); }
    void start(cudaStream_t s = 0) { cudaEventRecord(t0, s); }
    void stop(cudaStream_t  s = 0) { cudaEventRecord(t1, s); }
    float ms() {
        cudaEventSynchronize(t1);
        float elapsed;
        cudaEventElapsedTime(&elapsed, t0, t1);
        return elapsed;
    }
};

// Global accumulators that track GPU kernel time across all bonds and steps.
// Each MPI rank has its own copy, so no sharing or synchronization is needed.
// The timing breakdown is printed at the end of compute_sk_hk_rank.
static float g_ms_trotter_rxx = 0.f;
static float g_ms_trotter_ryy = 0.f;
static float g_ms_trotter_rzz = 0.f;
static float g_ms_extract     = 0.f;   // extract_odd_sector / insert_odd_sector
static float g_ms_pauli       = 0.f;   // Pauli gate applications inside H(k)
static float g_ms_unprep      = 0.f;   // CX + H/Sdg + readback per measurement

// CPU-side linear algebra routines (same as the non-profiling version).

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
            A[r][p] = cs_c * arp + cs_s * arq;             A[p][r] = std::conj(A[r][p]);
            A[r][q] = -std::conj(cs_s) * arp + cs_c * arq; A[q][r] = std::conj(A[r][q]);
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
    double spec_norm = std::max(std::abs(evals.front()), std::abs(evals.back()));
    return PI / spec_norm;
}

// CUDA gate kernels. These are the same as the non-profiling version,
// except BLOCK is now a compile-time parameter so we can sweep it.

__global__ void apply_x_gate(double2 *sv, long long ns, int q) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ns >> 1) return;
    long long lo = tid & ((1LL << q) - 1), hi = tid >> q;
    long long i0 = (hi << (q+1)) | lo, i1 = i0 | (1LL << q);
    double2 tmp = sv[i0]; sv[i0] = sv[i1]; sv[i1] = tmp;
}
__global__ void apply_hadamard(double2 *sv, long long ns, int q) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ns >> 1) return;
    long long lo = tid & ((1LL << q) - 1), hi = tid >> q;
    long long i0 = (hi << (q+1)) | lo, i1 = i0 | (1LL << q);
    double2 v0 = sv[i0], v1 = sv[i1];
    sv[i0] = make_double2((v0.x+v1.x)*INV_SQRT2, (v0.y+v1.y)*INV_SQRT2);
    sv[i1] = make_double2((v0.x-v1.x)*INV_SQRT2, (v0.y-v1.y)*INV_SQRT2);
}
__global__ void apply_cx(double2 *sv, long long ns, int ctrl, int tgt) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ns >> 1) return;
    long long lo = tid & ((1LL << tgt) - 1), hi = tid >> tgt;
    long long i0 = (hi << (tgt+1)) | lo, i1 = i0 | (1LL << tgt);
    if (!(i0 & (1LL << ctrl))) return;
    double2 tmp = sv[i0]; sv[i0] = sv[i1]; sv[i1] = tmp;
}
__global__ void apply_sdg_gate(double2 *sv, long long ns, int q) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ns >> 1) return;
    long long lo = tid & ((1LL << q) - 1), hi = tid >> q;
    long long i1 = ((hi << (q+1)) | lo) | (1LL << q);
    double2 v1 = sv[i1];
    sv[i1] = make_double2(v1.y, -v1.x);
}
__global__ void apply_y_gate(double2 *sv, long long ns, int q) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ns >> 1) return;
    long long lo = tid & ((1LL << q) - 1), hi = tid >> q;
    long long i0 = (hi << (q+1)) | lo, i1 = i0 | (1LL << q);
    double2 v0 = sv[i0], v1 = sv[i1];
    sv[i0] = make_double2(v1.y, -v1.x);
    sv[i1] = make_double2(-v0.y, v0.x);
}
__global__ void apply_z_gate(double2 *sv, long long ns, int q) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ns >> 1) return;
    long long lo = tid & ((1LL << q) - 1), hi = tid >> q;
    long long i1 = ((hi << (q+1)) | lo) | (1LL << q);
    sv[i1].x = -sv[i1].x; sv[i1].y = -sv[i1].y;
}
__global__ void apply_rxx_fused(double2 *sv, long long ns, int q1, int q2, double theta) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ns >> 2) return;
    int lo_bit = min(q1,q2), hi_bit = max(q1,q2);
    long long inner = tid & ((1LL << lo_bit) - 1);
    long long mid   = (tid >> lo_bit) & ((1LL << (hi_bit-lo_bit-1)) - 1);
    long long outer = tid >> (hi_bit-1);
    long long i00 = (outer<<(hi_bit+1))|(mid<<(lo_bit+1))|inner;
    long long i01=i00|(1LL<<lo_bit), i10=i00|(1LL<<hi_bit), i11=i01|(1LL<<hi_bit);
    double c=cos(theta/2.0), s=sin(theta/2.0);
    double2 v00=sv[i00],v01=sv[i01],v10=sv[i10],v11=sv[i11];
    sv[i00]=make_double2(c*v00.x+s*v11.y, c*v00.y-s*v11.x);
    sv[i01]=make_double2(c*v01.x+s*v10.y, c*v01.y-s*v10.x);
    sv[i10]=make_double2(c*v10.x+s*v01.y, c*v10.y-s*v01.x);
    sv[i11]=make_double2(c*v11.x+s*v00.y, c*v11.y-s*v00.x);
}
__global__ void apply_ryy_fused(double2 *sv, long long ns, int q1, int q2, double theta) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ns >> 2) return;
    int lo_bit = min(q1,q2), hi_bit = max(q1,q2);
    long long inner = tid & ((1LL << lo_bit) - 1);
    long long mid   = (tid >> lo_bit) & ((1LL << (hi_bit-lo_bit-1)) - 1);
    long long outer = tid >> (hi_bit-1);
    long long i00 = (outer<<(hi_bit+1))|(mid<<(lo_bit+1))|inner;
    long long i01=i00|(1LL<<lo_bit), i10=i00|(1LL<<hi_bit), i11=i01|(1LL<<hi_bit);
    double c=cos(theta/2.0), s=sin(theta/2.0);
    double2 v00=sv[i00],v01=sv[i01],v10=sv[i10],v11=sv[i11];
    sv[i00]=make_double2(c*v00.x-s*v11.y, c*v00.y+s*v11.x);
    sv[i01]=make_double2(c*v01.x+s*v10.y, c*v01.y-s*v10.x);
    sv[i10]=make_double2(c*v10.x+s*v01.y, c*v10.y-s*v01.x);
    sv[i11]=make_double2(c*v11.x-s*v00.y, c*v11.y+s*v00.x);
}
__global__ void apply_rzz_fused(double2 *sv, long long ns, int q1, int q2, double theta) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ns >> 2) return;
    int lo_bit = min(q1,q2), hi_bit = max(q1,q2);
    long long inner = tid & ((1LL << lo_bit) - 1);
    long long mid   = (tid >> lo_bit) & ((1LL << (hi_bit-lo_bit-1)) - 1);
    long long outer = tid >> (hi_bit-1);
    long long i00 = (outer<<(hi_bit+1))|(mid<<(lo_bit+1))|inner;
    long long i01=i00|(1LL<<lo_bit), i10=i00|(1LL<<hi_bit), i11=i01|(1LL<<hi_bit);
    double c=cos(theta/2.0), s=sin(theta/2.0);
    double2 v00=sv[i00]; sv[i00]=make_double2(c*v00.x+s*v00.y, c*v00.y-s*v00.x);
    double2 v11=sv[i11]; sv[i11]=make_double2(c*v11.x+s*v11.y, c*v11.y-s*v11.x);
    double2 v01=sv[i01]; sv[i01]=make_double2(c*v01.x-s*v01.y, c*v01.y+s*v01.x);
    double2 v10=sv[i10]; sv[i10]=make_double2(c*v10.x-s*v10.y, c*v10.y+s*v10.x);
}
__global__ void extract_odd_sector(const double2 *sv, double2 *sv_sys, long long ns_sys) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ns_sys) return;
    sv_sys[i] = sv[2*i+1];
}
__global__ void insert_odd_sector(const double2 *sv_sys, double2 *sv, long long ns_sys) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ns_sys) return;
    sv[2*i+1] = sv_sys[i];
}

// Applies a Trotter bond with timing. Each gate's execution time is
// accumulated into the global timing counters for the final breakdown.

void apply_trotter_bond_timed(double2 *sv, long long ns,
                              int q1, int q2,
                              double JX, double JY, double JZ,
                              double dt, bool reverse,
                              cudaStream_t stream = 0) {
    int blocks4 = (int)((ns/4 + BLOCK - 1) / BLOCK);

    // Tag this bond in the Nsight Systems timeline for easy identification.
    char label[64];
    snprintf(label, sizeof(label), "bond(%d,%d)_%s", q1, q2, reverse?"rev":"fwd");
    auto nvtx_id = nvtx_push(label, NVTX_COLOR_BOND);

    CudaTimer t;

    if (!reverse) {
        t.start(stream);
        apply_rxx_fused<<<blocks4, BLOCK, 0, stream>>>(sv, ns, q1, q2, 2.0*JX*dt);
        t.stop(stream); g_ms_trotter_rxx += t.ms();

        t.start(stream);
        apply_ryy_fused<<<blocks4, BLOCK, 0, stream>>>(sv, ns, q1, q2, 2.0*JY*dt);
        t.stop(stream); g_ms_trotter_ryy += t.ms();

        t.start(stream);
        apply_rzz_fused<<<blocks4, BLOCK, 0, stream>>>(sv, ns, q1, q2, 2.0*JZ*dt);
        t.stop(stream); g_ms_trotter_rzz += t.ms();
    } else {
        t.start(stream);
        apply_rzz_fused<<<blocks4, BLOCK, 0, stream>>>(sv, ns, q1, q2, 2.0*JZ*dt);
        t.stop(stream); g_ms_trotter_rzz += t.ms();

        t.start(stream);
        apply_ryy_fused<<<blocks4, BLOCK, 0, stream>>>(sv, ns, q1, q2, 2.0*JY*dt);
        t.stop(stream); g_ms_trotter_ryy += t.ms();

        t.start(stream);
        apply_rxx_fused<<<blocks4, BLOCK, 0, stream>>>(sv, ns, q1, q2, 2.0*JX*dt);
        t.stop(stream); g_ms_trotter_rxx += t.ms();
    }

    nvtx_pop(nvtx_id);
}

// Computes S(k) and H(k) for one Krylov step, instrumented with NVTX markers
// and CUDA event timers so we can profile every phase.

void compute_sk_hk_rank(int nQ, int excBit, int nSteps,
                        double JX, double JY, double JZ,
                        double dt_total,
                        Complex &outS, Complex &outH,
                        int rank) {

    long long ns      = 1LL << (nQ + 1);
    size_t sv_bytes   = ns * sizeof(double2);
    long long ns_sys  = 1LL << nQ;

    // Pre-allocate scratch buffers once, rather than calling cudaMalloc in the
    // inner loop. This was the biggest non-algorithmic overhead in the original code.
    double2 *sv_re_buf, *sv_im_buf, *sv_sys_p_buf;
    cudaMalloc(&sv_re_buf,    sv_bytes);
    cudaMalloc(&sv_im_buf,    sv_bytes);
    cudaMalloc(&sv_sys_p_buf, ns_sys * sizeof(double2));

    int blocks     = (int)((ns/2     + BLOCK - 1) / BLOCK);
    int blocks_sys = (int)((ns_sys/2 + BLOCK - 1) / BLOCK);
    int blocks_ext = (int)((ns_sys   + BLOCK - 1) / BLOCK);

    CudaTimer timer;

    // Helper: prepares the Hadamard test state and runs Trotter evolution.
    auto run_trotter = [&](const char* label) -> double2* {
        auto nvid = nvtx_push(label, NVTX_COLOR_TROTTER);

        double2 *sv;
        cudaMalloc(&sv, sv_bytes);
        cudaMemset(sv, 0, sv_bytes);
        double2 one = {1.0, 0.0};
        cudaMemcpy(sv, &one, sizeof(double2), cudaMemcpyHostToDevice);

        apply_hadamard<<<blocks, BLOCK>>>(sv, ns, 0);
        apply_cx<<<blocks, BLOCK>>>(sv, ns, 0, excBit+1);
        cudaDeviceSynchronize();

        double dt_step = dt_total / nSteps;
        double2 *sv_sys;
        cudaMalloc(&sv_sys, ns_sys * sizeof(double2));

        timer.start();
        extract_odd_sector<<<blocks_ext, BLOCK>>>(sv, sv_sys, ns_sys);
        timer.stop(); g_ms_extract += timer.ms();

        for (int j = 0; j < nSteps; j++) {
            if (j % 2 == 0) {
                for (int q = 0; q < nQ-1; q += 2)
                    apply_trotter_bond_timed(sv_sys, ns_sys, q, q+1,
                                             JX, JY, JZ, dt_step, false);
                cudaDeviceSynchronize();
                for (int q = 1; q < nQ-1; q += 2)
                    apply_trotter_bond_timed(sv_sys, ns_sys, q, q+1,
                                             JX, JY, JZ, dt_step, false);
                cudaDeviceSynchronize();
            } else {
                for (int q = 1; q < nQ-1; q += 2)
                    apply_trotter_bond_timed(sv_sys, ns_sys, q, q+1,
                                             JX, JY, JZ, dt_step, true);
                cudaDeviceSynchronize();
                for (int q = 0; q < nQ-1; q += 2)
                    apply_trotter_bond_timed(sv_sys, ns_sys, q, q+1,
                                             JX, JY, JZ, dt_step, true);
                cudaDeviceSynchronize();
            }
        }

        timer.start();
        insert_odd_sector<<<blocks_ext, BLOCK>>>(sv_sys, sv, ns_sys);
        timer.stop(); g_ms_extract += timer.ms();

        cudaFree(sv_sys);
        nvtx_pop(nvid);
        return sv;
    };

    // Measure the overlap matrix element S(k).
    {
        auto nvid = nvtx_push("S(k)_measurement", NVTX_COLOR_MEASURE_S);

        double2 *sv_s = run_trotter("trotter_for_S");
        double2 *sv_s_im;
        cudaMalloc(&sv_s_im, sv_bytes);
        cudaMemcpy(sv_s_im, sv_s, sv_bytes, cudaMemcpyDeviceToDevice);

        // Re(S): CX -> H -> read [0]
        timer.start();
        apply_cx<<<blocks, BLOCK>>>(sv_s, ns, 0, excBit+1);
        apply_hadamard<<<blocks, BLOCK>>>(sv_s, ns, 0);
        timer.stop(); g_ms_unprep += timer.ms();
        cudaDeviceSynchronize();
        double2 val;
        cudaMemcpy(&val, &sv_s[0], sizeof(double2), cudaMemcpyDeviceToHost);
        double re_s = 2.0*val.x - 1.0;

        // Im(S): CX -> Sdg -> H -> read [0]
        timer.start();
        apply_cx<<<blocks, BLOCK>>>(sv_s_im, ns, 0, excBit+1);
        apply_sdg_gate<<<blocks, BLOCK>>>(sv_s_im, ns, 0);
        apply_hadamard<<<blocks, BLOCK>>>(sv_s_im, ns, 0);
        timer.stop(); g_ms_unprep += timer.ms();
        cudaDeviceSynchronize();
        cudaMemcpy(&val, &sv_s_im[0], sizeof(double2), cudaMemcpyDeviceToHost);
        double im_s = 2.0*val.x - 1.0;

        outS = Complex(re_s, im_s);
        cudaFree(sv_s);
        cudaFree(sv_s_im);
        nvtx_pop(nvid);
    }

    // Measure the Hamiltonian matrix element H(k) by summing over bonds and Paulis.
    {
        auto nvid = nvtx_push("H(k)_measurement", NVTX_COLOR_MEASURE_H);
        outH = Complex(0.0, 0.0);
        double coeffs[3] = {JX, JY, JZ};

        double2 *sv_base = run_trotter("trotter_for_H");

        for (int b = 0; b < nQ-1; b++) {
            char bond_label[32];
            snprintf(bond_label, sizeof(bond_label), "Hk_bond_%d", b);
            auto bond_nvid = nvtx_push(bond_label, NVTX_COLOR_MEASURE_H);

            for (int pauli = 0; pauli < 3; pauli++) {

                // Real part of H for this bond and Pauli.
                cudaMemcpy(sv_re_buf, sv_base, sv_bytes, cudaMemcpyDeviceToDevice);
                extract_odd_sector<<<blocks_ext, BLOCK>>>(sv_re_buf, sv_sys_p_buf, ns_sys);
                cudaDeviceSynchronize();

                timer.start();
                if (pauli == 0) {
                    apply_x_gate<<<blocks_sys, BLOCK>>>(sv_sys_p_buf, ns_sys, b);
                    apply_x_gate<<<blocks_sys, BLOCK>>>(sv_sys_p_buf, ns_sys, b+1);
                } else if (pauli == 1) {
                    apply_y_gate<<<blocks_sys, BLOCK>>>(sv_sys_p_buf, ns_sys, b);
                    apply_y_gate<<<blocks_sys, BLOCK>>>(sv_sys_p_buf, ns_sys, b+1);
                } else {
                    apply_z_gate<<<blocks_sys, BLOCK>>>(sv_sys_p_buf, ns_sys, b);
                    apply_z_gate<<<blocks_sys, BLOCK>>>(sv_sys_p_buf, ns_sys, b+1);
                }
                timer.stop(); g_ms_pauli += timer.ms();

                insert_odd_sector<<<blocks_ext, BLOCK>>>(sv_sys_p_buf, sv_re_buf, ns_sys);
                cudaDeviceSynchronize();

                timer.start();
                apply_cx<<<blocks, BLOCK>>>(sv_re_buf, ns, 0, excBit+1);
                apply_hadamard<<<blocks, BLOCK>>>(sv_re_buf, ns, 0);
                timer.stop(); g_ms_unprep += timer.ms();
                cudaDeviceSynchronize();

                double2 val;
                cudaMemcpy(&val, &sv_re_buf[0], sizeof(double2), cudaMemcpyDeviceToHost);
                double re_h = 2.0*val.x - 1.0;

                // Imaginary part of H for this bond and Pauli.
                cudaMemcpy(sv_im_buf, sv_base, sv_bytes, cudaMemcpyDeviceToDevice);
                extract_odd_sector<<<blocks_ext, BLOCK>>>(sv_im_buf, sv_sys_p_buf, ns_sys);
                cudaDeviceSynchronize();

                timer.start();
                if (pauli == 0) {
                    apply_x_gate<<<blocks_sys, BLOCK>>>(sv_sys_p_buf, ns_sys, b);
                    apply_x_gate<<<blocks_sys, BLOCK>>>(sv_sys_p_buf, ns_sys, b+1);
                } else if (pauli == 1) {
                    apply_y_gate<<<blocks_sys, BLOCK>>>(sv_sys_p_buf, ns_sys, b);
                    apply_y_gate<<<blocks_sys, BLOCK>>>(sv_sys_p_buf, ns_sys, b+1);
                } else {
                    apply_z_gate<<<blocks_sys, BLOCK>>>(sv_sys_p_buf, ns_sys, b);
                    apply_z_gate<<<blocks_sys, BLOCK>>>(sv_sys_p_buf, ns_sys, b+1);
                }
                timer.stop(); g_ms_pauli += timer.ms();

                insert_odd_sector<<<blocks_ext, BLOCK>>>(sv_sys_p_buf, sv_im_buf, ns_sys);
                cudaDeviceSynchronize();

                timer.start();
                apply_cx<<<blocks, BLOCK>>>(sv_im_buf, ns, 0, excBit+1);
                apply_sdg_gate<<<blocks, BLOCK>>>(sv_im_buf, ns, 0);
                apply_hadamard<<<blocks, BLOCK>>>(sv_im_buf, ns, 0);
                timer.stop(); g_ms_unprep += timer.ms();
                cudaDeviceSynchronize();

                cudaMemcpy(&val, &sv_im_buf[0], sizeof(double2), cudaMemcpyDeviceToHost);
                double im_h = 2.0*val.x - 1.0;

                outH += coeffs[pauli] * Complex(re_h, im_h);
            }
            nvtx_pop(bond_nvid);
        }

        cudaFree(sv_base);
        nvtx_pop(nvid);
    }

    // Free the scratch buffers we allocated earlier.
    cudaFree(sv_re_buf);
    cudaFree(sv_im_buf);
    cudaFree(sv_sys_p_buf);
}

// Main entry point for the profiling version.

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
        std::cout << "JX=" << JX << " JY=" << JY << " JZ=" << JZ
                  << " nQ=" << nQ << "  BLOCK=" << BLOCK << "\n";

    if (nranks != 3) {
        if (rank == 0) std::cerr << "Need exactly 3 MPI ranks\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int excBit = nQ / 2;

    // Compute the classical reference energy from the single-particle sector.
    CMat H_proj(nQ, std::vector<Complex>(nQ, 0));
    for (int i = 0; i < nQ; i++) {
        int bt = (i>0?1:0)+(i<nQ-1?1:0);
        H_proj[i][i] = Complex(((nQ-1)-bt)*JZ - bt*JZ, 0);
        if (i < nQ-1) {
            H_proj[i][i+1] = Complex(JX+JY, 0);
            H_proj[i+1][i] = Complex(JX+JY, 0);
        }
    }
    CMat S_proj(nQ, std::vector<Complex>(nQ, 0));
    for (int i = 0; i < nQ; i++) S_proj[i][i] = 1;
    double E0_exact = solve_gen_eig(H_proj, S_proj, 1e-12);
    if (rank == 0)
        std::cout << "Classical exact E0: "
                  << std::fixed << std::setprecision(8) << E0_exact << "\n";

    double dt = Compute_SpectralNorm(H_proj);
    int krylov_dim   = 4;
    int trotter_steps = 4;
    Complex S0 = Complex(1.0, 0);
    Complex H0 = H_proj[excBit][excBit];

    // Set up the GPU for this rank.
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    cudaSetDevice(rank % num_gpus);
    cudaFree(0);   // force context creation now

    auto k_for = [](int r) -> int { return r == 0 ? 3 : r; };
    int my_k = k_for(rank);
    Complex my_S(0,0), my_H(0,0);

    MPI_Barrier(MPI_COMM_WORLD);

    // Main compute region. We wrap the whole thing in an NVTX range
    // so Nsight Systems shows all three ranks side by side.
    char top_label[32];
    snprintf(top_label, sizeof(top_label), "rank%d_k%d", rank, my_k);
    auto top_nvid = nvtx_push(top_label, 0xFF22BBFF);

    auto wall_start = std::chrono::high_resolution_clock::now();
    compute_sk_hk_rank(nQ, excBit, trotter_steps, JX, JY, JZ,
                       my_k * dt, my_S, my_H, rank);
    cudaDeviceSynchronize();
    auto wall_end = std::chrono::high_resolution_clock::now();

    nvtx_pop(top_nvid);

    long long wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>
                        (wall_end - wall_start).count();

    // Print the per-rank timing breakdown in order. Each line is easy to
    // parse, following the format:
    // BLOCK nQ rank k wall_ms rxx_ms ryy_ms rzz_ms extract_ms pauli_ms unprep_ms
    for (int r = 0; r < nranks; r++) {
        if (rank == r) {
            std::cout << "TIMING"
                      << "  BLOCK="    << BLOCK
                      << "  nQ="       << nQ
                      << "  rank="     << rank
                      << "  k="        << my_k
                      << "  wall_ms="  << wall_ms
                      << "  rxx_ms="   << std::fixed << std::setprecision(3) << g_ms_trotter_rxx
                      << "  ryy_ms="   << g_ms_trotter_ryy
                      << "  rzz_ms="   << g_ms_trotter_rzz
                      << "  extract_ms=" << g_ms_extract
                      << "  pauli_ms=" << g_ms_pauli
                      << "  unprep_ms=" << g_ms_unprep
                      << "\n";
            std::cout.flush();
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Gather results to rank 0 and solve the Krylov eigenvalue problem.
    {
        auto nvid = nvtx_push("MPI_Gather", NVTX_COLOR_MPI);
        double send4[4]  = { my_S.real(), my_S.imag(), my_H.real(), my_H.imag() };
        double recv16[16] = {};
        MPI_Gather(send4, 4, MPI_DOUBLE, recv16, 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        nvtx_pop(nvid);

        if (rank == 0) {
            Complex SR[4]={}, HR[4]={};
            SR[0]=S0; HR[0]=H0;
            SR[3]=Complex(recv16[0],recv16[1]); HR[3]=Complex(recv16[2],recv16[3]);
            for (int src=1; src<=2; src++) {
                int kk=k_for(src);
                SR[kk]=Complex(recv16[src*4+0],recv16[src*4+1]);
                HR[kk]=Complex(recv16[src*4+2],recv16[src*4+3]);
            }
            for (int K=1; K<=krylov_dim; K++) {
                CMat Sk(K,std::vector<Complex>(K)), Hk(K,std::vector<Complex>(K));
                for (int i=0;i<K;i++) {
                    for (int j=0;j<K;j++) {
                        int d=std::abs(i-j);
                        Sk[i][j]=(i<=j)?SR[d]:std::conj(SR[d]);
                        Hk[i][j]=(i<=j)?HR[d]:std::conj(HR[d]);
                    }
                    Sk[i][i]+=Complex(1e-8,0);
                }
                double e0=solve_gen_eig(Hk,Sk,1e-3);
                std::cout << "  K=" << K
                          << "  E0=" << e0
                          << "  |err|=" << std::abs(e0-E0_exact) << "\n";
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}