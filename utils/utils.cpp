#include "utils.h"
#include <Eigen/Sparse>
#include <omp.h>
#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

namespace utils {
std::pair<int32_t, int32_t>
ReadFromBinaryEigen(const std::string &filename,
                    std::vector<Eigen::Triplet<double, int32_t>> &coo) {

  std::ifstream file(filename, std::ios::binary);
  int64_t size;
  int32_t m = 0, n = 0;
  std::tuple<int32_t, int32_t, double> tmp;
  file.read(reinterpret_cast<char *>(&size), sizeof size);
  coo.clear();
  for (int64_t i = 0; i < size; i++) {
    file.read(reinterpret_cast<char *>(&tmp), sizeof tmp);
    coo.emplace_back(Eigen::Triplet<double, int32_t>(
        std::get<0>(tmp) - 1, std::get<1>(tmp) - 1, std::get<2>(tmp)));
    m = std::max(m, coo[i].row());
    n = std::max(n, coo[i].col());
  }
  return std::make_pair(m + 1, n + 1);
}

void printProgress(double percentage) {
  int val = (int)(percentage * 100);
  int lpad = (int)(percentage * PBWIDTH);
  int rpad = PBWIDTH - lpad;
  printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
  fflush(stdout);
}

void ReadFromBinaryVec(const std::string &filename, std::vector<double> &vec) {
  std::ifstream file(filename, std::ios::binary);
  int64_t size;
  double tmp;
  file.read(reinterpret_cast<char *>(&size), sizeof size);
  vec.resize(size);
  for (int64_t i = 0; i < size; i++) {
    file.read(reinterpret_cast<char *>(&tmp), sizeof tmp);
    vec[i] = tmp;
  }
}

// std::pair<int32_t, int32_t> ReadFromBinaryCSR(const std::string &filename,
//                                               std::vector<int32_t> &ai,
//                                               std::vector<int32_t> &aj,
//                                               std::vector<double> &av) {

//   std::vector<int32_t> rows;
//   auto &cols = aj;
//   auto &vals = av;
//   auto res = ReadFromBinaryCOO(filename, rows, cols, vals);
//   ai = std::vector<int32_t>(res.first + 1, 0);

//   size_t nnz = cols.size();
//   std::vector<size_t> index(nnz);
//   for (size_t i = 0; i < index.size(); i++) {
//     index[i] = i;
//   }
//   std::sort(index.begin(), index.end(), [&rows, &cols](size_t a, size_t b) {
//     if (rows[a] == rows[b])
//       return cols[a] < cols[b];
//     return rows[a] < rows[b];
//   });
//   for (size_t i = 0; i != nnz; i++) {
//     size_t current = i;
//     while (i != index[current]) {
//       size_t next = index[current];
//       std::swap(rows[current], rows[next]);
//       std::swap(cols[current], cols[next]);
//       std::swap(vals[current], vals[next]);
//       index[current] = current;
//       current = next;
//     }
//     index[current] = current;
//   }

//   for (size_t i = 0; i < nnz; i++) {
//     ai[rows[i] + 1]++;
//   }
//   for (size_t i = 0; i < res.first; i++) {
//     ai[i + 1] += ai[i];
//   }
// return res;
// }

std::vector<MKL_INT> randomPermute(const MKL_INT n, const MKL_INT base) {
  std::vector<MKL_INT> perm(n);
  std::iota(perm.begin(), perm.end(), base);

  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(perm.begin(), perm.end(), g);
  return perm;
}

std::vector<MKL_INT> inversePermute(const std::vector<MKL_INT> &perm,
                                    const MKL_INT base) {
  std::vector<MKL_INT> inv_perm(perm.size());
#pragma parallel for
  for (MKL_INT i = 0; i < perm.size(); i++) {
    inv_perm[perm[i] - base] = i + base;
  }
  return inv_perm;
}
bool isPermutation(const std::vector<MKL_INT> &perm, const MKL_INT base) {
  std::vector<MKL_INT> inv_perm(perm.size(), -1);
#pragma parallel for
  for (MKL_INT i = 0; i < perm.size(); i++) {
    inv_perm[perm[i] - base] = i + base;
  }
  // Compute the logical OR of all elements in the array
  bool all_true = true;
#pragma omp parallel for reduction(&& : all_true)
  for (MKL_INT i = 0; i < inv_perm.size(); i++) {
    all_true &= (inv_perm[i] != -1);
  }
  return all_true;
}
} // namespace utils