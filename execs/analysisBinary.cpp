
#include "../utils/utils.h"
#include <Eigen/Sparse>
#include <fstream>
#include <iostream>

int main() {
  std::vector<Eigen::Triplet<double, int32_t>> tuples;
  auto mn = utils::ReadFromBinary(
      "/u/dimiao/dimiao/work/vinay/mopt_tcase5/n92_M_40_16", tuples);
  std::cout << "m: " << mn.first << " n: " << mn.second << std::endl;

  Eigen::SparseMatrix<double> mat(mn.first, mn.second);
  mat.setFromTriplets(tuples.begin(), tuples.end());
  std::cout << mat.block(0, 0, 10, 10) << std::endl;
  return 0;
}