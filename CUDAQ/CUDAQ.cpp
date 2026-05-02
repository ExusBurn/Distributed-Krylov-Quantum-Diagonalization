// Krylov Quantum Diagonalization for a 10-qubit Heisenberg chain.
// This is a CUDA-Q accelerated implementation that follows IBM's
// "Efficient Hadamard Test" algorithm.

#include <iostream>
#include <vector>
#include <complex>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <fstream>
#include <sstream>
#include <chrono>

#include <cudaq.h>
#include <cudaq/algorithm.h>
#include <cudaq/builder.h>

using Complex = std::complex<double>;
using CMat = std::vector<std::vector<Complex>>;

static constexpr double PI = 3.14159265358979323846;
const Complex I1(0.0, 1.0);

// Linear algebra helper functions used throughout the program.
CMat cmat_eye(int n) {
    CMat M(n, std::vector<Complex>(n, 0));
    for (int i = 0; i < n; ++i) M[i][i] = 1;
    return M;
}

CMat cmat_mul(const CMat &A, const CMat &B) {
    int r = A.size(), k = B.size(), c = B[0].size();
    CMat C(r, std::vector<Complex>(c, 0));
    for (int i = 0; i < r; i++) for (int l = 0; l < k; l++)
        if (A[i][l] != Complex(0))
            for (int j = 0; j < c; j++) C[i][j] += A[i][l] * B[l][j];
    return C;
}

// Jacobi eigendecomposition
std::pair<std::vector<double>, CMat> jacobi_eigh(CMat A) {
    int n = A.size();
    CMat V = cmat_eye(n);
    for (int iter = 0; iter < 3000; ++iter) {
        double maxOff = 0; int p = 0, q = 1;
        for (int i = 0; i < n; i++) for (int j = i+1; j < n; j++)
            if (std::abs(A[i][j]) > maxOff) { maxOff = std::abs(A[i][j]); p = i; q = j; }
        if (maxOff < 1e-14) break;
        double app = A[p][p].real(), aqq = A[q][q].real();
        Complex apq = A[p][q];
        double tau = (aqq - app) / (2.0 * std::abs(apq));
        double t = (tau >= 0 ? 1.0 : -1.0) / (std::abs(tau) + std::sqrt(1 + tau*tau));
        double c = 1.0 / std::sqrt(1 + t*t), s = t * c;
        Complex ph = apq / std::abs(apq);
        Complex cs_c(c), cs_s = Complex(s) * std::conj(ph);
        for (int r = 0; r < n; r++) {
            if (r == p || r == q) continue;
            Complex arp = A[r][p], arq = A[r][q];
            A[r][p] = cs_c*arp + cs_s*arq;           A[p][r] = std::conj(A[r][p]);
            A[r][q] = -std::conj(cs_s)*arp + cs_c*arq; A[q][r] = std::conj(A[r][q]);
        }
        A[p][p] = app - t*std::abs(apq); A[q][q] = aqq + t*std::abs(apq);
        A[p][q] = A[q][p] = 0;
        for (int r = 0; r < n; r++) {
            Complex vrp = V[r][p], vrq = V[r][q];
            V[r][p] = cs_c*vrp + cs_s*vrq;
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

double solve_gen_eig(const CMat &H, const CMat &S, double threshold = 0.1) {
    int d = H.size();
    if (d == 1) return H[0][0].real();
    auto [sv, svecs] = jacobi_eigh(S);
    std::vector<std::vector<Complex>> good;
    for (int i = 0; i < d; i++) if (sv[i] > threshold) {
        std::vector<Complex> v(d);
        for (int r = 0; r < d; r++) v[r] = svecs[r][i];
        good.push_back(v);
    }
    int m = good.size();
    if (m == 0) return 1e10;
    
    CMat Hr(m, std::vector<Complex>(m, 0)), Sr(m, std::vector<Complex>(m, 0));
    for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) {
        for (int r = 0; r < d; r++) for (int cc = 0; cc < d; cc++) {
            Hr[i][j] += std::conj(good[i][r]) * H[r][cc] * good[j][cc];
            Sr[i][j] += std::conj(good[i][r]) * S[r][cc] * good[j][cc];
        }
    }
    auto [srv, srvecs] = jacobi_eigh(Sr);
    CMat Sinv(m, std::vector<Complex>(m, 0));
    for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) {
        for (int k = 0; k < m; k++) {
            Sinv[i][j] += srvecs[i][k] * (1.0/std::sqrt(std::max(srv[k], 1e-20))) * std::conj(srvecs[j][k]);
        }
    }
    CMat Ht = cmat_mul(cmat_mul(Sinv, Hr), Sinv);
    auto [evals, _] = jacobi_eigh(Ht);
    return evals[0];
}

// Build the Heisenberg Hamiltonian as a CUDA-Q spin operator.
cudaq::spin_op buildHeisenbergHamiltonian(int nQ, double JX, double JY, double JZ, int offset = 0) {
    cudaq::spin_op H;
    for (int i = 0; i < nQ - 1; i++) {
        H += JX * cudaq::spin::x(i + offset) * cudaq::spin::x(i + 1 + offset);
        H += JY * cudaq::spin::y(i + offset) * cudaq::spin::y(i + 1 + offset);
        H += JZ * cudaq::spin::z(i + offset) * cudaq::spin::z(i + 1 + offset);
    }
    return H;
}

// Single-particle reference for dt calculation matching exact classical baseline
double getSpectralNormReferenceClassical(int n, double JX, double JY, double JZ) {
    CMat H(n, std::vector<Complex>(n, 0));
    for (int i = 0; i < n; i++) {
        if (i < n - 1) {
            H[i][i+1] += Complex(JX + JY, 0);
        }
    }
    for (int i=0; i<n; i++) for (int j=0; j<i; j++) H[i][j] = std::conj(H[j][i]);
    for (int i=0; i<n; i++) {
        double z_val = 0;
        for (int b=0; b<n-1; b++) {
            if (b == i || b+1 == i) z_val -= JZ; 
            else z_val += JZ; 
        }
        H[i][i] += Complex(z_val, 0);
    }
    
    auto [ev, _] = jacobi_eigh(H);
    return std::max(std::abs(ev.front()), std::abs(ev.back()));
}

double singleParticleGzClassical(int n, double JX, double JY, double JZ) {
    CMat H(n+1, std::vector<Complex>(n+1, 0));
    for (int i = 0; i < n; i++) {
        if (i < n - 1) {
            H[i][i+1] += Complex(JX + JY, 0);
        }
    }
    for (int i=0; i<n; i++) for (int j=0; j<i; j++) H[i][j] = std::conj(H[j][i]);
    for (int i=0; i<n; i++) {
        double z_val = 0;
        for (int b=0; b<n-1; b++) {
            if (b == i || b+1 == i) z_val -= JZ; 
            else z_val += JZ; 
        }
        H[i][i] += Complex(z_val, 0);
    }
    double z_vac = (n-1)*JZ;
    H[n][n] += Complex(z_vac, 0);
    
    auto [ev, _] = jacobi_eigh(H);
    return ev[0];
}

// Trotter evolution kernels implemented with CUDA-Q gate primitives.

struct TrotterBondKernel {
    void operator()(cudaq::qubit& qi, cudaq::qubit& qi1, double JX, double JY, double JZ, double dt_circ) __qpu__ {
        // RXX (IBM manual decomposition)
        cudaq::h(qi); cudaq::h(qi1);
        cudaq::cx(qi, qi1);
        cudaq::rz(2.0 * JX * dt_circ, qi1);
        cudaq::cx(qi, qi1);
        cudaq::h(qi); cudaq::h(qi1);
        
        // RYY (IBM manual decomposition)
        cudaq::s(qi); cudaq::s(qi1);
        cudaq::h(qi); cudaq::h(qi1);
        cudaq::cx(qi, qi1);
        cudaq::rz(2.0 * JY * dt_circ, qi1);
        cudaq::cx(qi, qi1);
        cudaq::h(qi); cudaq::h(qi1);
        cudaq::sdg(qi); cudaq::sdg(qi1);

        // RZZ (IBM manual decomposition)
        cudaq::cx(qi, qi1);
        cudaq::rz(2.0 * JZ * dt_circ, qi1);
        cudaq::cx(qi, qi1);
    }
};

// Efficient Hadamard test kernel that measures overlap between quantum states.
struct EfficientHadamardTestKernel {
    void operator()(int nQ, int excQubit, int num_trotter_steps, double JX, double JY, double JZ, double dt_circ_k) __qpu__ {
        // q[0] is the ancilla, q[1]..q[nQ] is the system
        cudaq::qvector q(nQ + 1);
        cudaq::qubit& anc = q[0];

        // Prepare ancilla in superposition
        cudaq::h(anc);

        // Controlled state prep (CX from ancilla to excitation qubit)
        cudaq::cx(anc, q[excQubit + 1]);

        // Uncontrolled U_d evolution on the system qubits
        for (int step = 0; step < num_trotter_steps; step++) {
            if (step % 2 == 0) {
                // Even bonds
                for (int i = 0; i < nQ - 1; i += 2) {
                    TrotterBondKernel{}(q[i+1], q[i+2], JX, JY, JZ, dt_circ_k);
                }
                for (int i = 1; i < nQ - 1; i += 2) {
                    TrotterBondKernel{}(q[i+1], q[i+2], JX, JY, JZ, dt_circ_k);
                }
            } else {
                // Odd bonds reversed
                for (int i = (nQ % 2 == 0 ? nQ - 3 : nQ - 2); i >= 1; i -= 2) {
                    TrotterBondKernel{}(q[i+1], q[i+2], JX, JY, JZ, dt_circ_k);
                }
                for (int i = (nQ % 2 == 0 ? nQ - 2 : nQ - 3); i >= 0; i -= 2) {
                    TrotterBondKernel{}(q[i+1], q[i+2], JX, JY, JZ, dt_circ_k);
                }
            }
        }

        // Anti-controlled state prep inverse block
        cudaq::x(anc);
        cudaq::cx(anc, q[excQubit + 1]);
        cudaq::x(anc);
    }
};

// Holds the parameters for a single test case.
struct TestCase {
    double JX, JY, JZ;
    int    nQ;
};

// Parse input.txt: skip blank lines and '#' comments, read JX JY JZ nQ
std::vector<TestCase> readTestCases(const std::string& filename) {
    std::vector<TestCase> cases;
    std::ifstream fin(filename);
    if (!fin) {
        std::cerr << "ERROR: Cannot open input file '" << filename << "'\n";
        return cases;
    }
    std::string line;
    while (std::getline(fin, line)) {
        // Strip leading whitespace
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        if (line[start] == '#') continue;

        std::istringstream iss(line);
        TestCase tc;
        if (iss >> tc.JX >> tc.JY >> tc.JZ >> tc.nQ) {
            cases.push_back(tc);
        }
    }
    return cases;
}

// Runs a single KQD test case and returns the best ground-state energy estimate
// at the maximum Krylov dimension.
double runKQDTestCase(const TestCase& tc,
                      int krylov_dim,
                      int num_trotter_steps,
                      std::ostream& out) {

    const int  nQ = tc.nQ;
    const double JX = tc.JX, JY = tc.JY, JZ = tc.JZ;

    // Compute the time step from the spectral norm of the Hamiltonian.
    double specNorm = getSpectralNormReferenceClassical(nQ, JX, JY, JZ);
    double dt       = PI / specNorm;
    double dt_circ  = dt / num_trotter_steps;

    out << std::fixed << std::setprecision(6);
    out << "  nQ=" << nQ
        << "  JX=" << JX << "  JY=" << JY << "  JZ=" << JZ << "\n";
    out << "  specNorm=" << specNorm
        << "  dt=" << dt << "  dt_circ=" << dt_circ << "\n";

    // Set up the observables. The ancilla is qubit 0, system qubits are 1 through nQ.
    int excQubit = (nQ / 2) + 1;

    cudaq::spin_op S_real = cudaq::spin::x(0);
    cudaq::spin_op S_imag = cudaq::spin::y(0);

    cudaq::spin_op H_sys  = buildHeisenbergHamiltonian(nQ, JX, JY, JZ, 1);
    cudaq::spin_op H_real = H_sys * cudaq::spin::x(0);
    cudaq::spin_op H_imag = H_sys * cudaq::spin::y(0);

    double sum_zz = (nQ - 1) * JZ;

    // Compute the first-row elements of the Krylov overlap and Hamiltonian matrices.
    std::vector<Complex> S_first_row(krylov_dim, Complex(0.0));
    std::vector<Complex> H_first_row(krylov_dim, Complex(0.0));


    for (int k = 0; k < krylov_dim; k++) {
        double dt_circ_k = k * dt_circ;

        // Measure the overlap matrix element S(k) via ancilla X and Y expectations.
        cudaq::spin_op Ox = cudaq::spin::x(0);
        cudaq::spin_op Oy = cudaq::spin::y(0);
        
        // Measure the Hamiltonian matrix element H(k), split into real and imaginary parts.
        cudaq::spin_op H_real_part, H_imag_part;
        for (int i = 1; i < nQ; i++) {
            H_real_part += JX * cudaq::spin::x(0) * cudaq::spin::x(i) * cudaq::spin::x(i + 1);
            H_real_part += JY * cudaq::spin::x(0) * cudaq::spin::y(i) * cudaq::spin::y(i + 1);
            H_real_part += JZ * cudaq::spin::x(0) * cudaq::spin::z(i) * cudaq::spin::z(i + 1);

            H_imag_part += JX * cudaq::spin::y(0) * cudaq::spin::x(i) * cudaq::spin::x(i + 1);
            H_imag_part += JY * cudaq::spin::y(0) * cudaq::spin::y(i) * cudaq::spin::y(i + 1);
            H_imag_part += JZ * cudaq::spin::y(0) * cudaq::spin::z(i) * cudaq::spin::z(i + 1);
        }

        double exp_S_real = cudaq::observe(EfficientHadamardTestKernel{}, Ox, nQ, excQubit, num_trotter_steps, JX, JY, JZ, dt_circ_k).expectation();
        double exp_S_imag = cudaq::observe(EfficientHadamardTestKernel{}, Oy, nQ, excQubit, num_trotter_steps, JX, JY, JZ, dt_circ_k).expectation();
        double exp_H_real = cudaq::observe(EfficientHadamardTestKernel{}, H_real_part, nQ, excQubit, num_trotter_steps, JX, JY, JZ, dt_circ_k).expectation();
        double exp_H_imag = cudaq::observe(EfficientHadamardTestKernel{}, H_imag_part, nQ, excQubit, num_trotter_steps, JX, JY, JZ, dt_circ_k).expectation();

        Complex prefactor = std::polar(1.0, -1.0 * sum_zz * (k * dt));
        S_first_row[k] = prefactor * Complex(exp_S_real, exp_S_imag);
        H_first_row[k] = prefactor * Complex(exp_H_real, exp_H_imag);

        out << "    k=" << k
            << "  S=(" << S_first_row[k].real() << "+" << S_first_row[k].imag() << "j)"
            << "  H=(" << H_first_row[k].real() << "+" << H_first_row[k].imag() << "j)\n";
    }

    // Build the Toeplitz-structured Krylov overlap (S) and Hamiltonian (H) matrices.
    CMat Sfull(krylov_dim, std::vector<Complex>(krylov_dim, 0));
    CMat Hfull(krylov_dim, std::vector<Complex>(krylov_dim, 0));
    for (int j = 0; j < krylov_dim; j++) {
        for (int k = 0; k < krylov_dim; k++) {
            if (j <= k) {
                Sfull[j][k] = S_first_row[k - j];
                Hfull[j][k] = H_first_row[k - j];
            } else {
                Sfull[j][k] = std::conj(S_first_row[j - k]);
                Hfull[j][k] = std::conj(H_first_row[j - k]);
            }
        }
    }

    // Sweep over increasing Krylov subspace sizes to check convergence.
    double best_gnd = 1e10;
    for (int K = 1; K <= krylov_dim; K++) {
        CMat Hs(K, std::vector<Complex>(K)), Ss(K, std::vector<Complex>(K));
        for (int i = 0; i < K; i++)
            for (int j = 0; j < K; j++) {
                Hs[i][j] = Hfull[i][j];
                Ss[i][j] = Sfull[i][j];
            }
        double gnd = solve_gen_eig(Hs, Ss, 0.1);
        out << "    K=" << K
            << "  E0=" << std::fixed << std::setprecision(8) << gnd << "\n";
        if (K == krylov_dim) best_gnd = gnd;
    }
    return best_gnd;
}

// Main entry point for the KQD test harness.
int main() {


    std::ofstream out("kqd_results.txt");
    if (!out) {
        std::cerr << "Failed to open output file." << std::endl;
        return 1;
    }
    auto t_start_total = std::chrono::high_resolution_clock::now();

    const int krylov_dim       = 4;
    const int num_trotter_steps= 4;
    const double tolerance     = 0.5;   // |E0_kqd - E0_exact| < tolerance → PASS

    // Read test cases from input.txt
    std::vector<TestCase> tests = readTestCases("input.txt");
    if (tests.empty()) {
        std::cerr << "No test cases found in input.txt — aborting.\n";
        return 1;
    }

    const int nTests = static_cast<int>(tests.size());
    out  << "=========================================================\n";
    out  << " KQD Single-Particle Test Harness\n";
    out  << " krylov_dim=" << krylov_dim
         << "  num_trotter_steps=" << num_trotter_steps
         << "  tolerance=" << tolerance << "\n";
    out  << " Test cases loaded from input.txt: " << nTests << "\n";
    out  << "=========================================================\n\n";

    std::cout << "Running " << nTests << " test case(s)...\n";

    int nPass = 0, nFail = 0;

    for (int t = 0; t < nTests; t++) {
        const TestCase& tc = tests[t];
        if (tc.JX == 0.0 || tc.JY == 0.0 || tc.JZ == 0.0) {
            out << "\n  Error: JX, JY, or JZ is zero. Skipping test case..." << "\n";
            std::cout << "  [" << (t+1) << "/" << nTests << "] Error: JX, JY, or JZ is zero. Skipping...\n";
            continue;
        }
        auto tc_start = std::chrono::high_resolution_clock::now();

        out << "─────────────────────────────────────────────────────────\n";
        out << " TEST CASE " << (t + 1) << " / " << nTests << "\n";
        out << "─────────────────────────────────────────────────────────\n";

        // Exact single-particle ground state (classical diagonalization)
        double E0_exact = singleParticleGzClassical(tc.nQ, tc.JX, tc.JY, tc.JZ);

        // KQD estimate
        double E0_kqd   = runKQDTestCase(tc, krylov_dim, num_trotter_steps, out);

        double error    = std::abs(E0_kqd - E0_exact);
        bool   passed   = (error < tolerance);
        if (passed) ++nPass; else ++nFail;

        auto tc_end = std::chrono::high_resolution_clock::now();
        double tc_time = std::chrono::duration<double>(tc_end - tc_start).count();

        // Print a summary for this test case.
        out  << "\n  JX=" << tc.JX << "  JY=" << tc.JY
             << "  JZ=" << tc.JZ << "  nQ=" << tc.nQ << "\n";
        out  << "  Exact (classical):  "
             << std::fixed << std::setprecision(8) << E0_exact << "\n";
        out  << "  KQD   (K="  << krylov_dim << "):      "
             << std::fixed << std::setprecision(8) << E0_kqd   << "\n";
        out  << "  |Error|:            "
             << std::fixed << std::setprecision(8) << error      << "\n";
        out  << "  Tolerance:          " << tolerance << "\n";
        out  << "  Result:             " << (passed ? "PASS ✓" : "FAIL ✗")
             << "  [wall time: " << std::fixed << std::setprecision(4) << tc_time << " s]\n\n";

        // Also print a compact summary to the console.
        std::cout << "  [" << (t+1) << "/" << nTests << "]"
                  << " JX=" << tc.JX << " JY=" << tc.JY << " JZ=" << tc.JZ
                  << " nQ=" << tc.nQ
                  << "  exact=" << std::setprecision(6) << E0_exact
                  << "  kqd="   << E0_kqd
                  << "  err="   << std::setprecision(4) << error
                  << "  " << (passed ? "PASS" : "FAIL") << "\n";
    }

    // Print the final pass/fail summary and total wall time.
    auto t_end_total = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(t_end_total - t_start_total).count();

    out  << "=========================================================\n";
    out  << " SUMMARY:  " << nPass << " PASSED,  " << nFail << " FAILED"
         << "  (out of " << nTests << ")\n";
    out  << " Total wall time: "
         << std::fixed << std::setprecision(4) << total_time << " seconds\n";
    out  << "=========================================================\n";

    std::cout << "\n-----------------------------------------\n";
    std::cout << " SUMMARY: " << nPass << " PASSED,  " << nFail << " FAILED\n";
    std::cout << " Total time: " << total_time << " s\n";
    std::cout << "-----------------------------------------\n";
    std::cout << "Results saved to kqd_results.txt\n";
    return (nFail == 0) ? 0 : 1;
}
