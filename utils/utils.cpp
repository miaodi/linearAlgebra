#include "utils.h"
#include <Eigen/Sparse>
#include <omp.h>

#ifdef USE_BOOST_LIB
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#endif

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

template <typename COLTYPE>
std::vector<COLTYPE> randomPermute(const COLTYPE n, const COLTYPE base) {
  std::vector<COLTYPE> perm(n);
  std::iota(perm.begin(), perm.end(), base);

  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(perm.begin(), perm.end(), g);
  return perm;
}

template <typename COLTYPE>
void inversePermute(std::vector<COLTYPE> &iperm,
                    const std::vector<COLTYPE> &perm, const COLTYPE base) {
  iperm.resize(perm.size());
#pragma omp parallel for
  for (size_t i = 0; i < perm.size(); i++) {
    iperm[perm[i] - base] = i + base;
  }
}

#ifdef USE_BOOST_LIB
template <typename COLTYPE>
void printEliminationTree(const COLTYPE size, const COLTYPE base,
                          COLTYPE *const parent, const std::string &filename) {
  // Define the graph type with a string name property
  using graph_type =
      boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS,
                            boost::property<boost::vertex_name_t, std::string>>;

  // Get the vertex property map type
  using VertexNameMap =
      boost::property_map<graph_type, boost::vertex_name_t>::type;

  graph_type graph;
  VertexNameMap name_map = boost::get(boost::vertex_name, graph);

  // Add vertices and set their names
  for (COLTYPE i = 0; i < size; i++) {
    auto v = boost::add_vertex(graph);
    name_map[v] = std::to_string(i + base);
  }

  // Add edges based on parent array
  for (COLTYPE i = 0; i < size; i++) {
    if (parent[i] != i + base) { // If not root
      boost::add_edge(boost::vertex(i, graph),
                      boost::vertex(parent[i] - base, graph), graph);
    }
  }

  // Write to dot file
  std::ofstream dot(filename);
  boost::write_graphviz(
      dot, graph,
      boost::make_label_writer(boost::get(boost::vertex_name, graph)));
}

template void printEliminationTree(const std::int32_t size,
                                   const std::int32_t base,
                                   std::int32_t *const parent,
                                   const std::string &filename);
template void printEliminationTree(const std::int64_t size,
                                   const std::int64_t base,
                                   std::int64_t *const parent,
                                   const std::string &filename);
#endif

#define INSTANTIATE(T)                                                         \
  template std::vector<T> randomPermute(const T n, const T base);              \
  template void inversePermute(std::vector<T> &iperm,                          \
                               const std::vector<T> &perm, const T base);

INSTANTIATE(std::int32_t)
INSTANTIATE(std::int64_t)

} // namespace utils