#include "utils.h"
#include <Eigen/Sparse>
#include <omp.h>
#include <numeric>
#include <type_traits>

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
  if (val == 100)
    printf("\n");
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

template <class InputIt, class OutputIt>
OutputIt ParallelPrefixSum(const int nthreads, InputIt first, InputIt last, OutputIt d_first)
{
    using ValueType = typename std::iterator_traits<InputIt>::value_type;
    const auto total_size = std::distance(first, last);
    std::vector<ValueType> offsets(nthreads + 1, 0);
    offsets[0] = *d_first;
#pragma omp parallel num_threads(nthreads)
    {
        const int tid = omp_get_thread_num();

        // Step 1: Each thread computes local prefix sum
        auto [local_first, local_last] = utils::LoadBalancedPartitionPos(total_size, tid, nthreads);
        
        ValueType local_sum = 0;
        auto input_it = first + local_first;
        auto output_it = d_first + 1 + local_first;
        const auto input_end = first + local_last;
        
        for (; input_it != input_end; ++input_it, ++output_it)
        {
            local_sum += *input_it;
            *output_it = local_sum;
        }
        offsets[tid + 1] = local_sum;

// Step 2: Compute offsets
#pragma omp barrier
#pragma omp single
        {
            std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
        }

        // Step 3: Add offsets to local results
        const ValueType offset = offsets[tid];
        auto add_begin = d_first + 1 + local_first;
        auto add_end = d_first + 1 + local_last;

#if defined(_OPENMP) && defined(__cpp_if_constexpr) && __cpp_if_constexpr >= 201606
        if constexpr ( std::is_pointer_v<OutputIt> )
        {
#pragma omp simd
            for ( auto* out = add_begin; out < add_end; ++out )
            {
                *out += offset;
            }
        }
        else
#endif
        {
            for ( auto out = add_begin; out != add_end; ++out )
            {
                *out += offset;
            }
        }
    }
    return d_first + total_size + 1;
}

template <class Iter>
Iter ParallelPrefixSumInplace(const int nthreads, Iter first, Iter last)
{
    using ValueType = typename std::iterator_traits<Iter>::value_type;
    const auto total_size = std::distance( first, last );
    if ( total_size <= 0 )
        return last;

    std::vector<ValueType> offsets( nthreads + 1, 0 );
#pragma omp parallel num_threads( nthreads )
    {
        const int tid = omp_get_thread_num();
        auto [local_first, local_last] = utils::LoadBalancedPartitionPos( total_size, tid, nthreads );

        ValueType local_sum = 0;
        auto it = first + local_first;
        const auto it_end = first + local_last;
        for ( ; it != it_end; ++it )
        {
            local_sum += *it;
            *it = local_sum;
        }
        offsets[tid + 1] = local_sum;

#pragma omp barrier
#pragma omp single
        { std::partial_sum( offsets.begin(), offsets.end(), offsets.begin() ); }

        const ValueType offset = offsets[tid];
        auto add_begin = first + local_first;
        auto add_end = first + local_last;

#if defined(_OPENMP) && defined(__cpp_if_constexpr) && __cpp_if_constexpr >= 201606
        if constexpr ( std::is_pointer_v<Iter> )
        {
#pragma omp simd
            for ( auto* out = add_begin; out < add_end; ++out )
            {
                *out += offset;
            }
        }
        else
#endif
        {
            for ( auto out = add_begin; out != add_end; ++out )
            {
                *out += offset;
            }
        }
    }
    return last;
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

template <typename ROWTYPE, typename COLTYPE>
void writeAdjacencyGraphDOT(const COLTYPE rows, ROWTYPE const* ai, COLTYPE const* aj, 
                           const std::string& filename, const std::string& title) {
  std::ofstream dot(filename);
  if (!dot.is_open()) {
    std::cerr << "Error: Cannot open file " << filename << " for writing" << std::endl;
    return;
  }
  
  const int base = ai[0];
  
  // Create a valid DOT identifier by replacing spaces and special chars with underscores
  std::string dot_id = title;
  for (char& c : dot_id) {
    if (!std::isalnum(c)) {
      c = '_';
    }
  }
  
  // Write DOT format header
  dot << "digraph " << dot_id << " {\n";
  dot << "  label=\"" << title << "\";\n";  // Use label for the full title with spaces
  dot << "  rankdir=TB;\n";  // Top to bottom layout
  dot << "  node [shape=circle, style=filled, fillcolor=lightblue];\n";
  dot << "  edge [color=darkblue];\n\n";
  
  // Write vertices with labels
  for (COLTYPE i = 0; i < rows; i++) {
    dot << "  " << (i + base) << " [label=\"" << (i + base) << "\"];\n";
  }
  dot << "\n";
  
  // Write edges (skip self-loops for clarity)
  for (COLTYPE i = 0; i < rows; i++) {
    for (ROWTYPE j = ai[i] - base; j < ai[i + 1] - base; j++) {
      COLTYPE neighbor = aj[j] - base;
      if (neighbor != i) { // Skip self-loops for clarity
        dot << "  " << (i + base) << " -> " << (neighbor + base) << ";\n";
      }
    }
  }
  
  dot << "}\n";
  dot.close();
}

// Explicit template instantiations
template void writeAdjacencyGraphDOT<std::int32_t, std::int32_t>(const std::int32_t rows, 
                                              std::int32_t const* ai, std::int32_t const* aj, 
                                              const std::string& filename, const std::string& title);
template void writeAdjacencyGraphDOT<std::int64_t, std::int64_t>(const std::int64_t rows, 
                                              std::int64_t const* ai, std::int64_t const* aj, 
                                              const std::string& filename, const std::string& title);

// Enhanced version with partition support
template <typename ROWTYPE, typename COLTYPE>
void writeAdjacencyGraphDOT(const COLTYPE rows, 
                           ROWTYPE const* ai, 
                           COLTYPE const* aj,
                           COLTYPE const* partition,
                           COLTYPE num_partitions,
                           const std::string& filename, 
                           const std::string& title)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << " for writing." << std::endl;
        return;
    }
    
    const int base = (rows > 0) ? ai[0] : 0;
    
    // Create a valid graph name by replacing spaces and special characters
    std::string graph_name = title;
    for (char& c : graph_name) {
        if (!std::isalnum(c)) {
            c = '_';
        }
    }
    
    // Color palette for different partitions
    std::vector<std::string> colors = {
        "lightblue", "lightgreen", "lightcoral", "lightyellow", "lightpink",
        "lightgray", "lightcyan", "wheat", "lavender", "mistyrose",
        "palegreen", "peachpuff", "plum", "powderblue", "rosybrown",
        "sandybrown", "silver", "skyblue", "tan", "thistle"
    };
    
    file << "digraph " << graph_name << " {\n";
    file << "  label=\"" << title << "\";\n";
    file << "  labelloc=\"t\";\n";
    file << "  rankdir=TB;\n";
    file << "  node [shape=circle, style=filled];\n";
    file << "  edge [color=gray];\n\n";
    
    // If partitioning is provided, create subgraphs for each partition
    if (partition != nullptr && num_partitions > 1) {
        // Create subgraphs for each partition
        for (COLTYPE p = 0; p < num_partitions; ++p) {
            file << "  subgraph cluster_partition_" << p << " {\n";
            file << "    label=\"Partition " << p << "\";\n";
            file << "    style=\"rounded,filled\";\n";
            file << "    color=black;\n";
            file << "    bgcolor=\"" << colors[p % colors.size()] << "30\";\n"; // Semi-transparent background
            file << "    penwidth=2;\n";
            
            // Add nodes in this partition
            for (COLTYPE i = 0; i < rows; ++i) {
                if (partition[i] == p) {
                    file << "    " << (i + base) << " [label=\"" << (i + base) << "\", ";
                    file << "fillcolor=\"" << colors[p % colors.size()] << "\"];\n";
                }
            }
            file << "  }\n\n";
        }
    } else {
        // Single partition case - just add all nodes with default styling
        file << "  // All nodes in single partition\n";
        for (COLTYPE i = 0; i < rows; ++i) {
            file << "  " << (i + base) << " [label=\"" << (i + base) << "\", fillcolor=\"lightblue\"];\n";
        }
        file << "\n";
    }
    
    // Add edges with special styling for cross-partition edges
    file << "  // Edges\n";
    for (COLTYPE i = 0; i < rows; ++i) {
        for (ROWTYPE j = ai[i] - base; j < ai[i + 1] - base; ++j) {
            COLTYPE target = aj[j] - base;
            if (target < rows && target != i) { // Skip self-loops and invalid targets
                // Color edges based on whether they cross partitions
                if (partition != nullptr && num_partitions > 1) {
                    if (partition[i] == partition[target]) {
                        // Same partition - blue edges
                        file << "  " << (i + base) << " -> " << (target + base) << " [color=blue, penwidth=1];\n";
                    } else {
                        // Cross partition - red edges with thicker line
                        file << "  " << (i + base) << " -> " << (target + base) << " [color=red, penwidth=3, style=bold];\n";
                    }
                } else {
                    // Default single partition
                    file << "  " << (i + base) << " -> " << (target + base) << ";\n";
                }
            }
        }
    }
    
    // Add legend if partitioned
    if (partition != nullptr && num_partitions > 1) {
        file << "\n  // Legend\n";
        file << "  subgraph cluster_legend {\n";
        file << "    label=\"Legend\";\n";
        file << "    style=rounded;\n";
        file << "    color=black;\n";
        file << "    bgcolor=white;\n";
        file << "    legend_intra [label=\"Intra-partition\\nedge\", shape=plaintext, color=blue];\n";
        file << "    legend_inter [label=\"Inter-partition\\nedge\", shape=plaintext, color=red];\n";
        file << "    legend_intra -> legend_inter [color=blue, label=\"Same partition\"];\n";
        file << "    legend_inter -> legend_intra [color=red, style=bold, penwidth=3, label=\"Cross partition\"];\n";
        file << "  }\n";
    }
    
    file << "}\n";
    file.close();
    
    std::cout << "Adjacency graph written to: " << filename << std::endl;
    if (partition != nullptr && num_partitions > 1) {
        std::cout << "Graph shows " << num_partitions << " partitions with colored clustering" << std::endl;
        std::cout << "Blue edges: intra-partition, Red edges: inter-partition" << std::endl;
    }
    std::cout << "Generate visualization: dot -Tpng " << filename << " -o graph.png" << std::endl;
}

// Template instantiations for the enhanced version
template void writeAdjacencyGraphDOT<std::int32_t, std::int32_t>(const std::int32_t rows, 
                                              std::int32_t const* ai, 
                                              std::int32_t const* aj,
                                              std::int32_t const* partition,
                                              std::int32_t num_partitions,
                                              const std::string& filename, 
                                              const std::string& title);

template void writeAdjacencyGraphDOT<std::int64_t, std::int64_t>(const std::int64_t rows, 
                                                                std::int64_t const* ai, 
                                                                std::int64_t const* aj,
                                                                std::int64_t const* partition,
                                                                std::int64_t num_partitions,
                                                                const std::string& filename, 
                                                                const std::string& title);
#endif

#define INSTANTIATE_UTILS(T)                                                   \
    template std::vector<T> randomPermute( const T n, const T base );         \
    template void inversePermute( std::vector<T>& iperm,                      \
                                  const std::vector<T>& perm,                 \
                                  const T base );                             \
    template T* ParallelPrefixSum( const int nthreads, T* first, T* last, T* d_first ); \
    template T* ParallelPrefixSumInplace( const int nthreads, T* first, T* last );      \
    template typename std::vector<T>::iterator                                \
    ParallelPrefixSum<typename std::vector<T>::iterator,                      \
                      typename std::vector<T>::iterator>(                     \
        const int nthreads,                                                   \
        typename std::vector<T>::iterator first,                              \
        typename std::vector<T>::iterator last,                               \
        typename std::vector<T>::iterator d_first );                          \
    template typename std::vector<T>::iterator                                \
    ParallelPrefixSumInplace<typename std::vector<T>::iterator>(              \
        const int nthreads,                                                   \
        typename std::vector<T>::iterator first,                              \
        typename std::vector<T>::iterator last );

INSTANTIATE_UTILS( std::int32_t )
INSTANTIATE_UTILS( std::int64_t )

#undef INSTANTIATE_UTILS

} // namespace utils
