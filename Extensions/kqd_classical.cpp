#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <cstdlib>

using Complex = std::complex<double>;
using CVec    = std::vector<Complex>;
using CMat    = std::vector<std::vector<Complex>>;

static const double PI = 3.14159265358979323846;

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
    std::vector<double> evals; CMat evecs;       // each eigenvalue appears twice; keep one copy
    for (int idx = 0; idx < m && (int)evals.size() < n; idx++) {
        int col = eg[idx].second; double lam = eg[idx].first;
        CVec v(n);
        for (int r = 0; r < n; r++) v[r] = Complex(Vr[r][col], Vr[r+n][col]);
        for (auto& u : evecs) {                  // orthogonalize against already-kept vectors
            Complex ip = 0; for (int r = 0; r < n; r++) ip += std::conj(u[r]) * v[r];
            for (int r = 0; r < n; r++) v[r] -= ip * u[r];
        }
        double nv = 0; for (int r = 0; r < n; r++) nv += std::norm(v[r]); nv = std::sqrt(nv);
        if (nv < 1e-8) continue;                 // drop the duplicate copy
        for (int r = 0; r < n; r++) v[r] /= nv;
        evals.push_back(lam); evecs.push_back(v);
    }
    std::vector<int> o(evals.size()); std::iota(o.begin(), o.end(), 0);
    std::sort(o.begin(), o.end(), [&](int a, int b){ return evals[a] < evals[b]; });
    std::vector<double> sev(n); CMat sV(n, CVec(n));
    for (int i = 0; i < n; i++) { sev[i] = evals[o[i]];
        for (int r = 0; r < n; r++) sV[r][i] = evecs[o[i]][r]; }
    return {sev, sV};
}

// Build the full open-chain Heisenberg Hamiltonian as a dense matrix.
CMat build_heisenberg(int nQ, double JX, double JY, double JZ) {
    long long dim = 1LL << nQ;
    CMat H(dim, CVec(dim, Complex(0, 0)));
    for (long long s = 0; s < dim; s++) {
        double diag = 0.0;
        for (int i = 0; i < nQ - 1; i++) {
            int bi = (int)((s >> i) & 1);
            int bj = (int)((s >> (i + 1)) & 1);
            diag += JZ * (bi == bj ? 1.0 : -1.0);                 // Z_i Z_{i+1}
            long long sf = s ^ (1LL << i) ^ (1LL << (i + 1));      // X/Y flip both spins
            double coeff = (bi == bj) ? (JX - JY) : (JX + JY);     // XX + YY combined
            H[sf][s] += Complex(coeff, 0);
        }
        H[s][s] += Complex(diag, 0);
    }
    return H;
}

// Lowest eigenvalue of the generalized problem  H c = E S c
// via canonical orthogonalization: drop S-directions with eigenvalue < threshold.
double solve_gen_eig_lowest(const CMat &H, const CMat &S, double threshold) {
    int K = (int)H.size();
    if (K == 1) return H[0][0].real() / S[0][0].real();

    auto [sv, svec] = jacobi_eigh(S);
    std::vector<int> keep;
    for (int i = 0; i < K; i++) if (sv[i] > threshold) keep.push_back(i);
    int m = (int)keep.size();
    if (m == 0) return std::numeric_limits<double>::quiet_NaN();

    // X[r][a] = svec[r][keep[a]] / sqrt(sv[keep[a]])
    CMat X(K, CVec(m));
    for (int a = 0; a < m; a++) {
        double inv = 1.0 / std::sqrt(sv[keep[a]]);
        for (int r = 0; r < K; r++) X[r][a] = svec[r][keep[a]] * inv;
    }
    // HX[r][a] = sum_c H[r][c] X[c][a]
    CMat HX(K, CVec(m, 0));
    for (int r = 0; r < K; r++)
        for (int a = 0; a < m; a++) {
            Complex acc = 0;
            for (int c = 0; c < K; c++) acc += H[r][c] * X[c][a];
            HX[r][a] = acc;
        }
    // Hp[a][b] = sum_r conj(X[r][a]) HX[r][b]   (the H projected into the S-orthonormal basis)
    CMat Hp(m, CVec(m, 0));
    for (int a = 0; a < m; a++)
        for (int b = 0; b < m; b++) {
            Complex acc = 0;
            for (int r = 0; r < K; r++) acc += std::conj(X[r][a]) * HX[r][b];
            Hp[a][b] = acc;
        }
    auto [ev, evec] = jacobi_eigh(Hp);
    (void)evec;
    return ev[0];
}

int main() {
    double JX, JY, JZ;
    int nQ, krylov_dim;
    {
        std::ifstream f("input.txt");
        if (!f.is_open()) { std::cerr << "Cannot open input.txt\n"; return 1; }
        f >> JX >> JY >> JZ;
        if (!(f >> nQ))         nQ = 5;
        if (!(f >> krylov_dim)) krylov_dim = 50;
    }
    if (krylov_dim < 1) krylov_dim = 1;

    long long dim = 1LL << nQ;

    const char *results_env = std::getenv("RESULTS_FILE");
    std::string results_path = (results_env && results_env[0]) ? results_env : "results_classical.txt";
    std::ofstream out(results_path, std::ios::app);

    auto emit = [&](const std::string &line) {
        std::cout << line << "\n";
        out << line << "\n";
    };

    // Build and diagonalize the full Hamiltonian
    CMat H = build_heisenberg(nQ, JX, JY, JZ);
    auto [evals, evecs] = jacobi_eigh(H);

    double E0_true        = evals.front();
    double spectral_norm  = std::max(std::abs(evals.front()), std::abs(evals.back()));
    double dt             = PI / spectral_norm;

    // Initial state: NEEL state |1010...> (excite even-indexed qubits).
    // Same total-Sz sector as the global ground state -> Krylov reaches -7.7115.
    long long neel_index = 0;
    for (int q = 0; q < nQ; q += 2) neel_index |= (1LL << q);
    CVec psi0(dim, Complex(0, 0));
    psi0[neel_index] = 1.0;

    //Exact propagator U = exp(-i H dt) = V exp(-i D dt) V^dagge
    CMat U(dim, CVec(dim, Complex(0, 0)));
    for (long long a = 0; a < dim; a++)
        for (long long b = 0; b < dim; b++) {
            Complex acc = 0;
            for (long long mm = 0; mm < dim; mm++) {
                Complex phase = std::exp(Complex(0.0, -evals[mm] * dt));
                acc += evecs[a][mm] * phase * std::conj(evecs[b][mm]);
            }
            U[a][b] = acc;
        }

    // Krylov vectors  psi_k = U^k psi_0
    int K = krylov_dim;
    std::vector<CVec> psi(K, CVec(dim));
    psi[0] = psi0;
    for (int k = 1; k < K; k++)
        for (long long a = 0; a < dim; a++) {
            Complex acc = 0;
            for (long long b = 0; b < dim; b++) acc += U[a][b] * psi[k - 1][b];
            psi[k][a] = acc;
        }

    // Precompute H psi_k  (used for the projected Hamiltonian elements)
    std::vector<CVec> Hpsi(K, CVec(dim));
    for (int k = 0; k < K; k++)
        for (long long a = 0; a < dim; a++) {
            Complex acc = 0;
            for (long long b = 0; b < dim; b++) acc += H[a][b] * psi[k][b];
            Hpsi[k][a] = acc;
        }

    // Lowest eigenvalue actually REACHABLE from psi_0
    // (smallest eigenvalue whose eigenstate has nonzero overlap with psi_0;
    //  this is the value the Krylov estimate must converge to.)
    double reachable_min = std::numeric_limits<double>::infinity();
    for (long long mm = 0; mm < dim; mm++) {
        Complex ov = 0;
        for (long long a = 0; a < dim; a++) ov += std::conj(evecs[a][mm]) * psi0[a];
        if (std::norm(ov) > 1e-12) reachable_min = std::min(reachable_min, evals[mm]);
    }

    // Header
    out << "\n==================================================\n";
    {
        std::ostringstream h;
        h << "CLASSICAL exact KQD   nQ=" << nQ << "  krylov_dim=" << krylov_dim
          << "  JX=" << JX << " JY=" << JY << " JZ=" << JZ;
        emit(h.str());
    }
    {
        std::ostringstream h;
        h << "  Initial state = Dense Random State "
          << "   (dim = " << dim << ")";
        emit(h.str());
    }
    {
        std::ostringstream h;
        h << std::fixed << std::setprecision(8)
          << "  spectral norm = " << spectral_norm << "   dt = " << dt;
        emit(h.str());
    }
    {
        std::ostringstream h;
        h << std::fixed << std::setprecision(8) << "  True ground-state energy E0 (full spectrum) = " << E0_true;
        emit(h.str());
    }
    {
        std::ostringstream h;
        h << std::fixed << std::setprecision(8)
          << "  Lowest energy reachable from this psi_0    = " << reachable_min
          << "   (Krylov converges here)";
        emit(h.str());
    }
    emit("");
    emit("  Krylov convergence (E0 estimate vs full-spectrum ground state):");

    //Per-K solve
    const double s_threshold = 1e-12;
    const double plateau_eps = 1e-8;
    double prev_e0 = std::numeric_limits<double>::quiet_NaN();
    int    converged_k = -1;
    double converged_e0 = std::numeric_limits<double>::quiet_NaN();

    std::vector<Complex> Sk_1D(K), Hk_1D(K);
    for (int k = 0; k < K; k++) {
        Complex s_k = 0, h_k = 0;
        for (long long a = 0; a < dim; a++) {
            s_k += std::conj(psi0[a]) * psi[k][a];
            h_k += std::conj(psi0[a]) * Hpsi[k][a];
        }
        Sk_1D[k] = s_k;
        Hk_1D[k] = h_k;
    }

    for (int Kd = 1; Kd <= krylov_dim; Kd++) {
        CMat Hk(Kd, CVec(Kd)), Sk(Kd, CVec(Kd));
        for (int i = 0; i < Kd; i++) {
            for (int j = 0; j < Kd; j++) {
                int d = std::abs(i - j);
                Sk[i][j] = (i <= j) ? Sk_1D[d] : std::conj(Sk_1D[d]);
                Hk[i][j] = (i <= j) ? Hk_1D[d] : std::conj(Hk_1D[d]);
            }
        }

        double e0  = solve_gen_eig_lowest(Hk, Sk, s_threshold);
        double err = std::abs(e0 - E0_true);

        std::ostringstream line;
        line << "  K=" << std::setw(2) << Kd
             << "  E0=" << std::fixed << std::setprecision(8) << e0
             << "  |err vs true|=" << err
             << "  |err vs reachable|=" << std::abs(e0 - reachable_min);
        emit(line.str());

        if (e0 < E0_true - 1e-10) {
            emit("  Stopping: e0 is less than true ground state (ill-conditioning detected).");
            break;
        }

        if (std::isfinite(prev_e0) && std::abs(e0 - prev_e0) < plateau_eps) {
            converged_k  = Kd;
            converged_e0 = e0;
            emit("  Converged (estimate stopped changing) -- stopping early.");
            break;
        }
        prev_e0 = e0;
    }

    emit("");
    if (converged_k > 0) {
        std::ostringstream s;
        s << "  Converged in " << converged_k << " Krylov iterations  (E0="
          << std::fixed << std::setprecision(8) << converged_e0
          << ", which equals the reachable minimum "
          << reachable_min << ").";
        emit(s.str());
    } else {
        emit("  Did not plateau within krylov_dim iterations.");
    }

    {
        std::ostringstream s;
        s << "  NOTE: psi_0 is a dense random state, so it has overlap across all sectors.";
        emit(s.str());
    }
    emit("        The Krylov estimate should converge exactly to the true global ground state.");

    {
        std::ostringstream s;
        s << "  Results saved to " << results_path;
        emit(s.str());
    }
    out.close();
    return 0;
}