// Clean benchmark for triangular solves (serial and level-scheduled) without MKL dependency.
// Matrix: data/thermal2.mtx (assumed MatrixMarket, zero-based after read)
// Benchmarks:
//  - SerialForward: TriangularSolve<L>
//  - SerialBackward: TriangularSolve<U>
//  - LevelForward: LevelScheduleTriangularSubstitution<L>
//  - LevelBackward: LevelScheduleTriangularSubstitution<U>

#include "../config.h"
#include "io.hpp"
#include "matrix_utils.hpp"
#include "triangle_solve.hpp"
#include "precond.hpp" // for ILULevelSymbolic / ILULevelNumeric
#include "utils.h"
#include <benchmark/benchmark.h>
#include <omp.h>
#include <fstream>
#include <vector>
#include <memory>
#include <iostream>

namespace {
struct FactorSet {
    int k{0};
    matrix_utils::CSRMatrix<int,int,double> L, U; // L and U factors
    std::vector<double> D; // diagonal (non-unit)
    matrix_utils::LevelScheduleTriangularSubstitution<matrix_utils::TriangularMatrix::L,int,int,double> levelL;
    matrix_utils::LevelScheduleTriangularSubstitution<matrix_utils::TriangularMatrix::U,int,int,double> levelU;
    bool ok{false};
};

struct BenchmarkData {
    // Original matrix in CSR
    std::vector<int> ai, aj;
    std::vector<double> av;
    int n = 0; // dimension
    std::vector<FactorSet> factors; // ILU(k) for k=0,1,2
};

BenchmarkData& getData() {
    static BenchmarkData data; // initialized once
    static bool initialized = false;
    if (!initialized) {
        std::cout << "Initializing benchmark data...\n";
        std::ifstream f("data/thermal2.mtx");
        if (!f.good()) {
            throw std::runtime_error("Cannot open data/thermal2.mtx");
        }
        matrix_utils::readMatrixMarket(f, data.ai, data.aj, data.av); // produces ai size n+1
        data.n = static_cast<int>(data.ai.size()) - 1;
        // Build ILU(k) factors for k in {0,1,2}
        const std::vector<int> klevels = {0,1,2};
        data.factors.resize(klevels.size());
        for(size_t idx=0; idx<klevels.size(); ++idx) {
            int k = klevels[idx];
            data.factors[idx].k = k;
            matrix_utils::CSRMatrix<int,int,double> ilu_mat;
            matrix_utils::ILULevelSymbolic<decltype(ilu_mat)> sym;
            bool ok_sym = sym(data.n, data.ai.data(), data.aj.data(), k, ilu_mat);
            bool ok_num = false;
            if(ok_sym) ok_num = matrix_utils::ILULevelNumeric(data.n, data.ai.data(), data.aj.data(), data.av.data(), k, ilu_mat);
            if(ok_sym && ok_num) {
                matrix_utils::SplitLDU(ilu_mat.rows, ilu_mat.Base(), ilu_mat.AI(), ilu_mat.AJ(), ilu_mat.AV(), data.factors[idx].L, data.factors[idx].D, data.factors[idx].U);
                data.factors[idx].levelL.analysis(data.factors[idx].L.rows, data.factors[idx].L.ai.get(), data.factors[idx].L.aj.get(), data.factors[idx].L.av.get(), nullptr);
                data.factors[idx].levelU.analysis(data.factors[idx].U.rows, data.factors[idx].U.ai.get(), data.factors[idx].U.aj.get(), data.factors[idx].U.av.get(), data.factors[idx].D.data());
                data.factors[idx].ok = true;
                std::cout << "ILU("<<k<<") OK: levels(L)=" << data.factors[idx].levelL._levels << " levels(U)=" << data.factors[idx].levelU._levels << "\n";
            } else {
                std::cout << "ILU("<<k<<") failed\n";
            }
        }
        std::cout << "Loaded matrix data/thermal2.mtx, n=" << data.n << "\n";
        initialized = true;
    }
    return data;
}
} // namespace

// No global thread variable; thread control is per benchmark via LevelScheduleTriangularSubstitution::set_num_threads

// Helper to generate RHS and solution buffers
static inline void prepare_vectors(int n, std::vector<double>& b, std::vector<double>& x) {
    b.assign(n, 1.0);
    x.assign(n, 0.0);
}

// Serial forward substitution benchmark (single call per iteration, no Args)
// Serial forward substitution for factor set index kidx
static void SerialForward(benchmark::State& state, size_t kidx) {
    auto& d = getData();
    if(kidx >= d.factors.size() || !d.factors[kidx].ok) return; // nothing to benchmark
    auto &F = d.factors[kidx];
    std::vector<double> b, x;
    for (auto _ : state) {
        prepare_vectors(d.n, b, x);
        matrix_utils::TriangularSolve<matrix_utils::TriangularMatrix::L, int, int, double>(
            F.L.rows, F.L.ai.get(), F.L.aj.get(), F.L.av.get(), static_cast<const double*>(nullptr), b.data(), x.data());
    }
}

// Serial backward substitution for factor set index kidx
static void SerialBackward(benchmark::State& state, size_t kidx) {
    auto& d = getData();
    if(kidx >= d.factors.size() || !d.factors[kidx].ok) return;
    auto &F = d.factors[kidx];
    std::vector<double> b, x;
    for (auto _ : state) {
        prepare_vectors(d.n, b, x);
        matrix_utils::TriangularSolve<matrix_utils::TriangularMatrix::U, int, int, double>(
            F.U.rows, F.U.ai.get(), F.U.aj.get(), F.U.av.get(), F.D.data(), b.data(), x.data());
    }
}

// Parallel level-scheduled forward substitution benchmark (parameter = thread count power-of-two)
static void LevelForward(benchmark::State& state, size_t kidx) {
    auto threads = static_cast<int>(state.range(0));
    auto& d = getData();
    if(kidx >= d.factors.size() || !d.factors[kidx].ok) return;
    auto &F = d.factors[kidx];
    F.levelL.set_num_threads(threads);
    std::vector<double> b, x;
    for (auto _ : state) {
        prepare_vectors(d.n, b, x);
        F.levelL(b.data(), x.data());
    }
    state.counters["threads"] = threads;
    state.counters["k"] = F.k;
}

static void LevelBackward(benchmark::State& state, size_t kidx) {
    auto threads = static_cast<int>(state.range(0));
    auto& d = getData();
    if(kidx >= d.factors.size() || !d.factors[kidx].ok) return;
    auto &F = d.factors[kidx];
    F.levelU.set_num_threads(threads);
    std::vector<double> b, x;
    for (auto _ : state) {
        prepare_vectors(d.n, b, x);
        F.levelU(b.data(), x.data());
    }
    state.counters["threads"] = threads;
    state.counters["k"] = F.k;
}

// Dynamically register powers-of-two thread counts up to (and excluding) max threads if not exact power-of-two
static void RegisterBenchmarks() {
    auto &d = getData();
    // Register serial benchmarks per k
    for(size_t kidx=0; kidx<d.factors.size(); ++kidx) {
        if(!d.factors[kidx].ok) continue;
        benchmark::RegisterBenchmark((std::string("SerialForward/ILU=") + std::to_string(d.factors[kidx].k)).c_str(),
            [kidx](benchmark::State& st){ SerialForward(st, kidx); });
        benchmark::RegisterBenchmark((std::string("SerialBackward/ILU=") + std::to_string(d.factors[kidx].k)).c_str(),
            [kidx](benchmark::State& st){ SerialBackward(st, kidx); });
    }
    // Parallel benchmarks: thread power-of-two args
    int maxThreads = omp_get_num_procs();
    if (const char* env = std::getenv("MAX_BENCH_THREADS")) {
        int val = std::atoi(env);
        if (val > 0) maxThreads = std::min(maxThreads, val);
    }
    std::vector<int> threadArgs;
    for(int t=1; t<maxThreads; t<<=1) {
        threadArgs.push_back(t);
        if((t<<1) >= maxThreads) break;
    }
    for(size_t kidx=0; kidx<d.factors.size(); ++kidx) {
        if(!d.factors[kidx].ok) continue;
        auto nameF = std::string("LevelForward/ILU=") + std::to_string(d.factors[kidx].k);
        auto nameB = std::string("LevelBackward/ILU=") + std::to_string(d.factors[kidx].k);
        auto* fwdReg = benchmark::RegisterBenchmark(nameF.c_str(), [kidx](benchmark::State& st){ LevelForward(st, kidx); });
        auto* bwdReg = benchmark::RegisterBenchmark(nameB.c_str(), [kidx](benchmark::State& st){ LevelBackward(st, kidx); });
        for(int arg : threadArgs) { fwdReg->Arg(arg); bwdReg->Arg(arg); }
    }
}
namespace {
struct BenchInit { BenchInit() { RegisterBenchmarks(); } };
static BenchInit _benchInitializer;
}

BENCHMARK_MAIN();
