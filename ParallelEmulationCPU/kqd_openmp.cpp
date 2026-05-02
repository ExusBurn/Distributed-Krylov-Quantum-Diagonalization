// kqd_openmp.cpp
//
// This is the CPU/OpenMP version of the Krylov Quantum Diagonalization.
// The math is exactly the same as the CUDA version (linebyline.cu), but
// the execution model is different:
//   - CUDA kernels become OpenMP parallel for loops
//   - GPU memory allocation becomes std::vector<double2>
//   - Device-to-device copies become simple std::copy or memcpy calls
//   - Synchronization is handled implicitly by OpenMP barriers
//   - Instead of 3 GPU ranks, we use 3 MPI ranks on any nodes
//
// Each MPI rank uses OMP_NUM_THREADS threads to parallelize the gate
// sweeps over the statevector.
//
// For the strong-scaling experiment, run.sh launches 3 MPI tasks and
// sweeps the thread count through {32, 64, 128}.

#include <mpi.h>
#include <omp.h>

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cstring>

using Complex = std::complex<double>;
using CMat    = std::vector<std::vector<Complex>>;

static const double PI       = 3.14159265358979323846;
static const double INV_SQRT2 = 0.7071067811865475;

// A simple double2 struct so the gate code looks the same as the CUDA version.
struct double2 { double x, y; };
static inline double2 make_double2(double x, double y) { return {x, y}; }

// Classical linear algebra routines (same as the CUDA version).
std::pair<std::vector<double>, CMat> jacobi_eigh(CMat A) {
    int n = (int)A.size();
    CMat V(n, std::vector<Complex>(n, 0));
    for (int i = 0; i < n; i++) V[i][i] = 1;

    for (int iter = 0; iter < 3000; iter++) {
        double maxOff = 0; int p = 0, q = 1;
        for (int i = 0; i < n; i++)
            for (int j = i+1; j < n; j++)
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
            A[r][q] = -std::conj(cs_s)*arp + cs_c*arq;     A[q][r] = std::conj(A[r][q]);
        }
        A[p][p] = app - t * std::abs(apq);
        A[q][q] = aqq + t * std::abs(apq);
        A[p][q] = A[q][p] = 0;

        for (int r = 0; r < n; r++) {
            Complex vrp = V[r][p], vrq = V[r][q];
            V[r][p] = cs_c * vrp + cs_s * vrq;
            V[r][q] = -std::conj(cs_s)*vrp + cs_c*vrq;
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
    return PI / std::max(std::abs(evals.front()), std::abs(evals.back()));
}

// Gate implementations. Each function replaces a CUDA kernel.
// The index arithmetic is identical to the GPU version. The only difference
// is that OpenMP parallelizes the outer loop instead of GPU threads.

// Single-qubit gates.

void apply_hadamard(double2 *sv, long long ns, int q) {
    long long half = ns >> 1;
    #pragma omp parallel for schedule(dynamic)
    for (long long tid = 0; tid < half; tid++) {
        long long lo = tid & ((1LL << q) - 1);
        long long hi = tid >> q;
        long long i0 = (hi << (q+1)) | lo;
        long long i1 = i0 | (1LL << q);
        double2 v0 = sv[i0], v1 = sv[i1];
        sv[i0] = make_double2((v0.x+v1.x)*INV_SQRT2, (v0.y+v1.y)*INV_SQRT2);
        sv[i1] = make_double2((v0.x-v1.x)*INV_SQRT2, (v0.y-v1.y)*INV_SQRT2);
    }
}

void apply_x_gate(double2 *sv, long long ns, int q) {
    long long half = ns >> 1;
    #pragma omp parallel for schedule(dynamic)
    for (long long tid = 0; tid < half; tid++) {
        long long lo = tid & ((1LL << q) - 1);
        long long hi = tid >> q;
        long long i0 = (hi << (q+1)) | lo;
        long long i1 = i0 | (1LL << q);
        double2 tmp = sv[i0]; sv[i0] = sv[i1]; sv[i1] = tmp;
    }
}

void apply_y_gate(double2 *sv, long long ns, int q) {
    long long half = ns >> 1;
    #pragma omp parallel for schedule(dynamic)
    for (long long tid = 0; tid < half; tid++) {
        long long lo = tid & ((1LL << q) - 1);
        long long hi = tid >> q;
        long long i0 = (hi << (q+1)) | lo;
        long long i1 = i0 | (1LL << q);
        double2 v0 = sv[i0], v1 = sv[i1];
        sv[i0] = make_double2( v1.y, -v1.x);   // i·v1 rotated
        sv[i1] = make_double2(-v0.y,  v0.x);
    }
}

void apply_z_gate(double2 *sv, long long ns, int q) {
    long long half = ns >> 1;
    #pragma omp parallel for schedule(dynamic)
    for (long long tid = 0; tid < half; tid++) {
        long long lo = tid & ((1LL << q) - 1);
        long long hi = tid >> q;
        long long i1 = ((hi << (q+1)) | lo) | (1LL << q);
        sv[i1].x = -sv[i1].x;
        sv[i1].y = -sv[i1].y;
    }
}

// Add this function alongside the gate implementations
double measure_x(const double2 *sv, long long ns) {
    double val = 0.0;
    #pragma omp parallel for reduction(+:val) schedule(static)
    for (long long i = 0; i < ns; i++) {
        double prob = sv[i].x * sv[i].x + sv[i].y * sv[i].y;
        val += (i % 2 == 0) ? prob : -prob;
    }
    return val;
}

void apply_s_gate(double2 *sv, long long ns, int q) {
    long long half = ns >> 1;
    #pragma omp parallel for schedule(dynamic)
    for (long long tid = 0; tid < half; tid++) {
        long long lo = tid & ((1LL << q) - 1);
        long long hi = tid >> q;
        long long i1 = ((hi << (q+1)) | lo) | (1LL << q);
        double2 v1 = sv[i1];
        sv[i1] = make_double2(-v1.y, v1.x);
    }
}

void apply_sdg_gate(double2 *sv, long long ns, int q) {
    long long half = ns >> 1;
    #pragma omp parallel for schedule(dynamic)
    for (long long tid = 0; tid < half; tid++) {
        long long lo = tid & ((1LL << q) - 1);
        long long hi = tid >> q;
        long long i1 = ((hi << (q+1)) | lo) | (1LL << q);
        double2 v1 = sv[i1];
        sv[i1] = make_double2(v1.y, -v1.x);
    }
}

void apply_cx(double2 *sv, long long ns, int ctrl, int tgt) {
    long long half = ns >> 1;
    #pragma omp parallel for schedule(dynamic)
    for (long long tid = 0; tid < half; tid++) {
        long long lo = tid & ((1LL << tgt) - 1);
        long long hi = tid >> tgt;
        long long i0 = (hi << (tgt+1)) | lo;
        long long i1 = i0 | (1LL << tgt);
        if (!(i0 & (1LL << ctrl))) continue;
        double2 tmp = sv[i0]; sv[i0] = sv[i1]; sv[i1] = tmp;
    }
}

// Two-qubit fused gates (same math as the GPU kernels).

void apply_rxx_fused(double2 *sv, long long ns, int q1, int q2, double theta) {
    int lo_bit = std::min(q1,q2), hi_bit = std::max(q1,q2);
    long long quarter = ns >> 2;
    double c = std::cos(theta/2.0), s = std::sin(theta/2.0);

    #pragma omp parallel for schedule(dynamic)
    for (long long tid = 0; tid < quarter; tid++) {
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

void apply_ryy_fused(double2 *sv, long long ns, int q1, int q2, double theta) {
    int lo_bit = std::min(q1,q2), hi_bit = std::max(q1,q2);
    long long quarter = ns >> 2;
    double c = std::cos(theta/2.0), s = std::sin(theta/2.0);

    #pragma omp parallel for schedule(dynamic)
    for (long long tid = 0; tid < quarter; tid++) {
        long long inner = tid & ((1LL << lo_bit) - 1);
        long long mid   = (tid >> lo_bit) & ((1LL << (hi_bit - lo_bit - 1)) - 1);
        long long outer = tid >> (hi_bit - 1);

        long long i00 = (outer << (hi_bit+1)) | (mid << (lo_bit+1)) | inner;
        long long i01 = i00 | (1LL << lo_bit);
        long long i10 = i00 | (1LL << hi_bit);
        long long i11 = i00 | (1LL << lo_bit) | (1LL << hi_bit);

        double2 v00=sv[i00], v01=sv[i01], v10=sv[i10], v11=sv[i11];

        sv[i00] = make_double2(c*v00.x - s*v11.y,  c*v00.y + s*v11.x);  // opposite sign vs RXX
        sv[i01] = make_double2(c*v01.x + s*v10.y,  c*v01.y - s*v10.x);
        sv[i10] = make_double2(c*v10.x + s*v01.y,  c*v10.y - s*v01.x);
        sv[i11] = make_double2(c*v11.x - s*v00.y,  c*v11.y + s*v00.x);  // opposite sign vs RXX
    }
}

void apply_rzz_fused(double2 *sv, long long ns, int q1, int q2, double theta) {
    int lo_bit = std::min(q1,q2), hi_bit = std::max(q1,q2);
    long long quarter = ns >> 2;
    double c = std::cos(theta/2.0), s = std::sin(theta/2.0);

    #pragma omp parallel for schedule(dynamic)
    for (long long tid = 0; tid < quarter; tid++) {
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

// Extract and insert the ancilla=1 sector of the statevector.

void extract_odd_sector(const double2 *sv, double2 *sv_sys, long long ns_sys) {
    #pragma omp parallel for schedule(dynamic)
    for (long long i = 0; i < ns_sys; i++)
        sv_sys[i] = sv[2*i + 1];
}

void insert_odd_sector(const double2 *sv_sys, double2 *sv, long long ns_sys) {
    #pragma omp parallel for schedule(dynamic)
    for (long long i = 0; i < ns_sys; i++)
        sv[2*i + 1] = sv_sys[i];
}

// Apply the Trotter bond gates (same ordering as the GPU version).

void apply_trotter_bond(double2 *sv, long long ns, int q1, int q2,
                        double JX, double JY, double JZ, double dt) {
    apply_rxx_fused(sv, ns, q1, q2, 2.0*JX*dt);
    apply_ryy_fused(sv, ns, q1, q2, 2.0*JY*dt);
    apply_rzz_fused(sv, ns, q1, q2, 2.0*JZ*dt);
}

void apply_trotter_bond_reverse(double2 *sv, long long ns, int q1, int q2,
                                double JX, double JY, double JZ, double dt) {
    apply_rzz_fused(sv, ns, q1, q2, 2.0*JZ*dt);
    apply_ryy_fused(sv, ns, q1, q2, 2.0*JY*dt);
    apply_rxx_fused(sv, ns, q1, q2, 2.0*JX*dt);
}

// Computes the overlap S(k) and Hamiltonian H(k) matrix elements for one
// Krylov step using the Hadamard test, just like the GPU version.
// The statevector is a heap-allocated std::vector shared across OpenMP threads.

void compute_sk_hk_rank(int nQ, int excBit, int nSteps,
                        double JX, double JY, double JZ,
                        double dt_total,
                        Complex &outS, Complex &outH) {

    long long ns     = 1LL << (nQ + 1);
    long long ns_sys = 1LL << nQ;

    auto make_sv = [&]() {
        std::vector<double2> sv(ns, {0.0, 0.0});
        return sv;
    };

    auto copy_sv = [&](const std::vector<double2> &src) {
        return src;
    };

    auto run_trotter = [&]() {
        auto sv = make_sv();
        sv[0] = {1.0, 0.0};

        apply_hadamard(sv.data(), ns, 0);
        apply_cx(sv.data(), ns, 0, excBit + 1);

        double dt_step = dt_total / nSteps;

        std::vector<double2> sv_sys(ns_sys, {0.0, 0.0});
        extract_odd_sector(sv.data(), sv_sys.data(), ns_sys);

        for (int j = 0; j < nSteps; j++) {
            if (j % 2 == 0) {
                for (int q = 0; q < nQ-1; q += 2)
                    apply_trotter_bond(sv_sys.data(), ns_sys, q, q+1, JX, JY, JZ, dt_step);
                for (int q = 1; q < nQ-1; q += 2)
                    apply_trotter_bond(sv_sys.data(), ns_sys, q, q+1, JX, JY, JZ, dt_step);
            } else {
                for (int q = 1; q < nQ-1; q += 2)
                    apply_trotter_bond_reverse(sv_sys.data(), ns_sys, q, q+1, JX, JY, JZ, dt_step);
                for (int q = 0; q < nQ-1; q += 2)
                    apply_trotter_bond_reverse(sv_sys.data(), ns_sys, q, q+1, JX, JY, JZ, dt_step);
            }
        }

        insert_odd_sector(sv_sys.data(), sv.data(), ns_sys);
        return sv;
    };

    // Measure the overlap matrix element S(k).
    {
        auto sv_s    = run_trotter();
        auto sv_s_im = copy_sv(sv_s);
        // Re(S)
        apply_x_gate(sv_s.data(), ns, 0);
        apply_cx(sv_s.data(), ns, 0, excBit+1);
        apply_x_gate(sv_s.data(), ns, 0);
        apply_hadamard(sv_s.data(), ns, 0);
        double re_s = measure_x(sv_s.data(), ns);   // ← was 2.0*sv_s[0].x - 1.0

        // Im(S)
        apply_x_gate(sv_s_im.data(), ns, 0);
        apply_cx(sv_s_im.data(), ns, 0, excBit+1);
        apply_x_gate(sv_s_im.data(), ns, 0);
        apply_sdg_gate(sv_s_im.data(), ns, 0);
        apply_hadamard(sv_s_im.data(), ns, 0);
        double im_s = measure_x(sv_s_im.data(), ns); // ← was 2.0*sv_s_im[0].x - 1.0

        outS = Complex(re_s, im_s);
    }

    // Measure the Hamiltonian matrix element H(k).
    outH = Complex(0.0, 0.0);
    double coeffs[3] = {JX, JY, JZ};

    auto sv_base = run_trotter();

    for (int b = 0; b < nQ-1; b++) {
        int q1_sys = b;
        int q2_sys = b + 1;

        for (int pauli = 0; pauli < 3; pauli++) {

            // Real part of H for this bond and Pauli.
            auto sv_re = copy_sv(sv_base);
            {
                std::vector<double2> sv_sys_p(ns_sys);
                extract_odd_sector(sv_re.data(), sv_sys_p.data(), ns_sys);

                if      (pauli == 0) { apply_x_gate(sv_sys_p.data(), ns_sys, q1_sys);
                                       apply_x_gate(sv_sys_p.data(), ns_sys, q2_sys); }
                else if (pauli == 1) { apply_y_gate(sv_sys_p.data(), ns_sys, q1_sys);
                                       apply_y_gate(sv_sys_p.data(), ns_sys, q2_sys); }
                else                 { apply_z_gate(sv_sys_p.data(), ns_sys, q1_sys);
                                       apply_z_gate(sv_sys_p.data(), ns_sys, q2_sys); }

                insert_odd_sector(sv_sys_p.data(), sv_re.data(), ns_sys);
            }
            // Re(H)  (inside the bond/pauli loop)
            apply_x_gate(sv_re.data(), ns, 0);
            apply_cx(sv_re.data(), ns, 0, excBit+1);
            apply_x_gate(sv_re.data(), ns, 0);
            apply_hadamard(sv_re.data(), ns, 0);
            double re_h = measure_x(sv_re.data(), ns);   // ← was 2.0*sv_re[0].x - 1.0

            // Imaginary part of H for this bond and Pauli.
            auto sv_im = copy_sv(sv_base);
            {
                std::vector<double2> sv_sys_p(ns_sys);
                extract_odd_sector(sv_im.data(), sv_sys_p.data(), ns_sys);

                if      (pauli == 0) { apply_x_gate(sv_sys_p.data(), ns_sys, q1_sys);
                                       apply_x_gate(sv_sys_p.data(), ns_sys, q2_sys); }
                else if (pauli == 1) { apply_y_gate(sv_sys_p.data(), ns_sys, q1_sys);
                                       apply_y_gate(sv_sys_p.data(), ns_sys, q2_sys); }
                else                 { apply_z_gate(sv_sys_p.data(), ns_sys, q1_sys);
                                       apply_z_gate(sv_sys_p.data(), ns_sys, q2_sys); }

                insert_odd_sector(sv_sys_p.data(), sv_im.data(), ns_sys);
            }
            // Im(H)
            apply_x_gate(sv_im.data(), ns, 0);
            apply_cx(sv_im.data(), ns, 0, excBit+1);
            apply_x_gate(sv_im.data(), ns, 0);
            apply_sdg_gate(sv_im.data(), ns, 0);
            apply_hadamard(sv_im.data(), ns, 0);
            double im_h = measure_x(sv_im.data(), ns);   // ← was 2.0*sv_im[0].x - 1.0

            outH += coeffs[pauli] * Complex(re_h, im_h);
        }
    }
}

// Main entry point.
int main(int argc, char **argv) {

    // MPI: provide_thread level = MPI_THREAD_FUNNELED
    // (only the main thread calls MPI; OpenMP runs inside each rank)
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        std::cerr << "MPI does not support MPI_THREAD_FUNNELED\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    if (nranks != 3) {
        if (rank == 0) std::cerr << "Need exactly 3 MPI ranks\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Read the Hamiltonian parameters and system size from input.txt.
    double JX, JY, JZ;
    int nQ;
    {
        std::ifstream f("input.txt");
        if (!f.is_open()) { std::cerr << "Cannot open input.txt\n"; MPI_Abort(MPI_COMM_WORLD,1); }
        f >> JX >> JY >> JZ >> nQ;
    }

    if (rank == 0) {
        std::cout << "=== KQD OpenMP CPU Simulator ===\n";
        std::cout << "JX=" << JX << " JY=" << JY << " JZ=" << JZ << " nQ=" << nQ << "\n";
        std::cout << "MPI ranks=" << nranks
                  << "  OMP threads/rank=" << omp_get_max_threads() << "\n";
        std::cout << "Statevector size per run: "
                  << ((1LL << (nQ+1)) * 16) / (1024*1024) << " MB\n\n";
    }

    int excBit = (nQ / 2) + 1;

    // Build the single-particle projected Hamiltonian (same as GPU version).
    CMat H_proj(nQ, std::vector<Complex>(nQ, 0));
    for (int i = 0; i < nQ; i++) {
        int bonds_touching   = (i > 0 ? 1 : 0) + (i < nQ-1 ? 1 : 0);
        int bonds_not_touching = (nQ-1) - bonds_touching;
        H_proj[i][i] = Complex(bonds_not_touching*JZ - bonds_touching*JZ, 0);
        if (i < nQ-1) {
            H_proj[i][i+1] = Complex(JX+JY, 0);
            H_proj[i+1][i] = Complex(JX+JY, 0);
        }
    }

    CMat S_proj(nQ, std::vector<Complex>(nQ, 0));
    for (int i = 0; i < nQ; i++) S_proj[i][i] = 1;

    double E0_exact = solve_gen_eig(H_proj, S_proj, 1e-12);

    if (rank == 0)
        std::cout << "Classical exact E0 (single-particle sector): "
                  << std::fixed << std::setprecision(8) << E0_exact << "\n\n";

    double dt          = Compute_SpectralNorm(H_proj);
    int krylov_dim     = 4;
    int trotter_steps  = 4;

    // k=0 is analytical
    Complex S0 = Complex(1.0, 0.0);
    Complex H0 = H_proj[excBit][excBit];

    // Rank → k assignment (identical to GPU version)
    auto k_for = [](int r) -> int { return r == 0 ? 3 : r; };
    int my_k = k_for(rank);

    MPI_Barrier(MPI_COMM_WORLD);
    auto t0 = std::chrono::high_resolution_clock::now();

    Complex my_S(0,0), my_H(0,0);
    compute_sk_hk_rank(nQ, excBit, trotter_steps,
                       JX, JY, JZ, my_k * dt,
                       my_S, my_H);

    auto t1 = std::chrono::high_resolution_clock::now();
    long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();

    // Print each rank's results one at a time to avoid interleaving output.
    for (int r = 0; r < nranks; r++) {
        if (rank == r) {
            std::cout << "  k=" << my_k
                      << "  S(k)=(" << std::fixed << std::setprecision(8)
                      << my_S.real() << ", " << my_S.imag() << ")"
                      << "  H(k)=(" << my_H.real() << ", " << my_H.imag() << ")"
                      << "  threads=" << omp_get_max_threads()
                      << "  time=" << ms << " ms\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Gather all results to rank 0.
    double send4[4]  = { my_S.real(), my_S.imag(), my_H.real(), my_H.imag() };
    double recv16[16] = {};
    MPI_Gather(send4, 4, MPI_DOUBLE, recv16, 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // On rank 0, assemble the Toeplitz Krylov matrices and solve.
    if (rank == 0) {
        std::cout << "\n";

        Complex SR[4] = {}, HR[4] = {};
        SR[0] = S0;  HR[0] = H0;
        // rank 0 sent k=3
        SR[3] = Complex(recv16[0], recv16[1]);
        HR[3] = Complex(recv16[2], recv16[3]);
        // ranks 1,2 sent k=1,2
        for (int src = 1; src <= 2; src++) {
            int kk = k_for(src);
            SR[kk] = Complex(recv16[src*4+0], recv16[src*4+1]);
            HR[kk] = Complex(recv16[src*4+2], recv16[src*4+3]);
        }

        std::cout << "--- Krylov convergence ---\n";
        for (int K = 1; K <= krylov_dim; K++) {
            CMat Sk(K, std::vector<Complex>(K));
            CMat Hk(K, std::vector<Complex>(K));
            for (int i = 0; i < K; i++) {
                for (int j = 0; j < K; j++) {
                    int d = std::abs(i - j);
                    Sk[i][j] = (i <= j) ? SR[d] : std::conj(SR[d]);
                    Hk[i][j] = (i <= j) ? HR[d] : std::conj(HR[d]);
                }
                Sk[i][i] += Complex(1e-8, 0);  // regularization
            }
            double e0 = solve_gen_eig(Hk, Sk, 1e-3);
            std::cout << "  K=" << K << "  E0=" << std::fixed << std::setprecision(8)
                      << e0 << "  |err|=" << std::abs(e0 - E0_exact) << "\n";
        }
        std::cout << "\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}