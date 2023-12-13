#include "utils.h"
#include <fstream>
namespace utils {
std::pair<int32_t, int32_t>
ReadFromBinary(const std::string &filename,
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
        std::get<0>(tmp), std::get<1>(tmp), std::get<2>(tmp)));
    m = std::max(m, coo[i].row());
    n = std::max(n, coo[i].col());
  }
  return std::make_pair(m + 1, n + 1);
}

} // namespace utils