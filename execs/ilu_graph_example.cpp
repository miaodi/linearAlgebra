#include "config.h"
#include "graph_algs.hpp"
#include "io.hpp"
#include "matrix_utils.hpp"
#include "precond.hpp" // for ILULevelSymbolic / ILULevelNumeric / SplitLDU
#include "utils.h"
#include <cxxopts.hpp>
#include <fstream>
#include <iostream>
#include <string>

// Helper function to write adjacency list in text format
template<typename ROWTYPE, typename COLTYPE>
void writeAdjacencyListText(const COLTYPE rows, ROWTYPE const* ai, COLTYPE const* aj, 
                           const std::string& filename) {
  std::ofstream out(filename);
  const int base = ai[0];
  
  out << "Adjacency List (Base " << base << "):\n";
  out << "=================================\n";
  
  for (COLTYPE i = 0; i < rows; i++) {
    out << "Node " << (i + base) << ": ";
    for (ROWTYPE j = ai[i] - base; j < ai[i + 1] - base; j++) {
      out << (aj[j]) << " ";
    }
    out << "\n";
  }
  
  out << "\nGraph Statistics:\n";
  out << "Total nodes: " << rows << "\n";
  out << "Total edges: " << (ai[rows] - ai[0]) << "\n";
  out << "Average degree: " << (double)(ai[rows] - ai[0]) / rows << "\n";
}

int main(int argc, char **argv) {
  cxxopts::Options options("ILU Graph Example", 
                          "Example of ILU factorization followed by graph analysis");
  options.add_options()
    ("f,filename", "Matrix Market file to read",
     cxxopts::value<std::string>()->default_value("../data/ex5.mtx"))
    ("l,level", "Level of ILU factorization",
     cxxopts::value<int>()->default_value("0"))
    ("h,help", "Print usage");

  auto result = options.parse(argc, argv);
  
  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  std::string filename = result["filename"].as<std::string>();
  int level = result["level"].as<int>();

  std::cout << "=== ILU Graph Analysis Example ===" << std::endl;
  std::cout << "Input file: " << filename << std::endl;
  std::cout << "ILU level: " << level << std::endl << std::endl;

  // 1. Read matrix from Matrix Market format
  std::cout << "1. Reading matrix from file..." << std::endl;
  std::ifstream f(filename);
  if (!f.is_open()) {
    std::cerr << "Error: Cannot open file " << filename << std::endl;
    return 1;
  }
  
  std::vector<int> rows;
  std::vector<int> cols;
  std::vector<double> vals;
  matrix_utils::readMatrixMarket(f, rows, cols, vals);
  f.close();
  
  matrix_utils::CSRMatrix<int, int, double> A;
  A.rows = rows.size() - 1;
  A.cols = A.rows;
  A.ResizeAI(rows.size());
  A.ResizeAJ(vals.size());
  A.ResizeAV(vals.size());
  std::copy(rows.begin(), rows.end(), A.AI());
  std::copy(cols.begin(), cols.end(), A.AJ());
  std::copy(vals.begin(), vals.end(), A.AV());
  
  std::cout << "   Matrix size: " << A.rows << " x " << A.cols << std::endl;
  std::cout << "   Non-zeros: " << A.NNZ() << std::endl;
  std::cout << "   Base indexing: " << A.Base() << std::endl << std::endl;

  // 2. Perform ILU factorization using matrix_utils
  std::cout << "2. Performing ILU(" << level << ") factorization..." << std::endl;
  
  // Build ILU (symbolic+numeric) for factor splitting
  matrix_utils::CSRMatrix<int, int, double> ilu;
  matrix_utils::ILULevelSymbolic<matrix_utils::CSRMatrix<int, int, double>> iluSym;
  if (!iluSym(A.rows, A.AI(), A.AJ(), level, ilu)) {
    std::cerr << "Error: ILU symbolic factorization failed" << std::endl;
    return 1;
  }
  
  if (!matrix_utils::ILULevelNumeric(A.rows, A.AI(), A.AJ(), A.AV(), level, ilu)) {
    std::cerr << "Error: ILU numeric factorization failed" << std::endl;
    return 1;
  }
  
  std::cout << "   ILU factorization completed successfully" << std::endl;
  std::cout << "   ILU factor size: " << ilu.rows << " x " << ilu.cols << std::endl;
  std::cout << "   ILU factor non-zeros: " << ilu.NNZ() << std::endl << std::endl;
  
  // 3. Split ILU into L, D, U factors
  std::cout << "3. Splitting ILU into L, D, U factors..." << std::endl;
  
  matrix_utils::CSRMatrix<int, int, double> L, U;
  std::vector<double> D;
  matrix_utils::SplitLDU(ilu.rows, ilu.Base(), ilu.AI(), ilu.AJ(), ilu.AV(), L, D, U);
  
  std::cout << "   L factor size: " << L.rows << " x " << L.cols << std::endl;
  std::cout << "   L factor non-zeros: " << L.NNZ() << std::endl;
  std::cout << "   U factor size: " << U.rows << " x " << U.cols << std::endl;
  std::cout << "   U factor non-zeros: " << U.NNZ() << std::endl << std::endl;

  // 4. Check if L is a DAG (should be true for lower triangular matrix)
  std::cout << "4. Checking if L is a Directed Acyclic Graph (DAG)..." << std::endl;
  
  bool is_dag = graph::IsDAG<int, int>(L.rows, L.AI(), L.AJ());
  std::cout << "   L is a DAG: " << (is_dag ? "YES" : "NO") << std::endl;
  
  if (!is_dag) {
    std::cout << "   Warning: L factor contains cycles, which is unexpected for a triangular matrix" << std::endl;
  }
  std::cout << std::endl;

  // 5. Transpose L to get L^T
  std::cout << "5. Computing transpose of L..." << std::endl;
  
  // Allocate memory for L^T
  auto Lt_csr = matrix_utils::AllocateCSRData<int, int, double>(L.rows, L.NNZ());
  auto Lt_ai = std::get<0>(Lt_csr).get();
  auto Lt_aj = std::get<1>(Lt_csr).get();
  auto Lt_av = std::get<2>(Lt_csr).get();
  
  // Compute transpose using parallel transpose
  matrix_utils::ParallelTranspose2<int, int, double>(
    L.rows, L.cols,
    L.AI(), L.AJ(), L.AV(),
    Lt_ai, Lt_aj, Lt_av
  );
  
  std::cout << "   L^T computed successfully" << std::endl;
  std::cout << "   L^T size: " << L.rows << " x " << L.cols << std::endl;
  std::cout << "   L^T non-zeros: " << L.NNZ() << std::endl << std::endl;

  // 6. Check if L^T is also a DAG
  std::cout << "6. Checking if L^T is a DAG..." << std::endl;
  
  bool Lt_is_dag = graph::IsDAG<int, int>(L.rows, Lt_ai, Lt_aj);
  std::cout << "   L^T is a DAG: " << (Lt_is_dag ? "YES" : "NO") << std::endl << std::endl;

  // 7. Compute transitive reduction of L^T (if it's a DAG)
  std::cout << "7. Computing transitive reduction of L^T..." << std::endl;
  
  if (!Lt_is_dag) {
    std::cout << "   Skipping transitive reduction since L^T is not a DAG" << std::endl;
  } else {
    // Allocate memory for transitive reduction result
    auto tr_csr = matrix_utils::AllocateCSRData<int, int, double>(L.rows, L.NNZ());
    auto tr_ai = std::get<0>(tr_csr).get();
    auto tr_aj = std::get<1>(tr_csr).get();
    
    // Compute transitive reduction
    graph::TransitiveReduction<int, int> tr_solver;
    tr_solver(L.rows, Lt_ai, Lt_aj, tr_ai, tr_aj, false); // Assume no self-loops
    
    // Count non-zeros in result
    int tr_nnz = tr_ai[L.rows] - tr_ai[0];
    
    std::cout << "   Transitive reduction completed successfully" << std::endl;
    std::cout << "   Original L^T non-zeros: " << L.NNZ() << std::endl;
    std::cout << "   Reduced graph non-zeros: " << tr_nnz << std::endl;
    std::cout << "   Reduction ratio: " << (double)tr_nnz / L.NNZ() * 100.0 << "%" << std::endl;
    
    // Count edges removed
    int edges_removed = L.NNZ() - tr_nnz;
    std::cout << "   Transitive edges removed: " << edges_removed << std::endl << std::endl;
    
    // 8. Optional: Write results to files for visualization
    std::cout << "8. Writing results to files for visualization..." << std::endl;
    
    // Write original matrix as SVG
    std::ofstream svg_orig("original_matrix.svg");
    if (svg_orig.is_open()) {
      matrix_utils::writeSVG(A.rows, A.cols, 
                           A.AI(), A.AJ(), svg_orig);
      svg_orig.close();
      std::cout << "   Original matrix written to: original_matrix.svg" << std::endl;
    }
    
    // Write L factor as SVG
    std::ofstream svg_L("L_factor.svg");
    if (svg_L.is_open()) {
      matrix_utils::writeSVG(L.rows, L.cols,
                           L.AI(), L.AJ(), svg_L);
      svg_L.close();
      std::cout << "   L factor written to: L_factor.svg" << std::endl;
    }
    
    // Write L^T as SVG
    std::ofstream svg_Lt("Lt_transpose.svg");
    if (svg_Lt.is_open()) {
      matrix_utils::writeSVG(L.rows, L.cols,
                           Lt_ai, Lt_aj, svg_Lt);
      svg_Lt.close();
      std::cout << "   L^T written to: Lt_transpose.svg" << std::endl;
    }
    
    // Write transitive reduction as SVG
    std::ofstream svg_tr("transitive_reduction.svg");
    if (svg_tr.is_open()) {
      matrix_utils::writeSVG(L.rows, L.cols,
                           tr_ai, tr_aj, svg_tr);
      svg_tr.close();
      std::cout << "   Transitive reduction written to: transitive_reduction.svg" << std::endl;
    }
    
    // 9. Write adjacency graphs in DOT format for GraphViz
    std::cout << "9. Writing adjacency graphs as DOT files..." << std::endl;
    
#ifdef USE_BOOST_LIB
    // Write L^T as adjacency graph (DOT format)
    utils::writeAdjacencyGraphDOT(L.rows, Lt_ai, Lt_aj, 
                                 "Lt_adjacency_graph.dot", "Lt_Graph");
    std::cout << "   L^T adjacency graph written to: Lt_adjacency_graph.dot" << std::endl;
    
    // Write transitive reduction as adjacency graph (DOT format)
    utils::writeAdjacencyGraphDOT(L.rows, tr_ai, tr_aj, 
                                 "transitive_reduction_graph.dot", "TR_Graph");
    std::cout << "   Transitive reduction graph written to: transitive_reduction_graph.dot" << std::endl;
#else
    std::cout << "   DOT file generation disabled (USE_BOOST_LIB not enabled)" << std::endl;
#endif
    
    // 10. Write adjacency lists in text format
    std::cout << "10. Writing adjacency lists in text format..." << std::endl;
    
    writeAdjacencyListText<int, int>(L.rows, Lt_ai, Lt_aj, "Lt_adjacency_list.txt");
    std::cout << "   L^T adjacency list written to: Lt_adjacency_list.txt" << std::endl;
    
    writeAdjacencyListText<int, int>(L.rows, tr_ai, tr_aj, "transitive_reduction_list.txt");
    std::cout << "   Transitive reduction list written to: transitive_reduction_list.txt" << std::endl;
    
    // 11. Instructions for visualization
    std::cout << "\n11. Visualization Instructions:" << std::endl;
    std::cout << "   - SVG files: Open directly in web browser or vector graphics editor" << std::endl;
    std::cout << "   - DOT files: Convert to images using GraphViz:" << std::endl;
    std::cout << "     > dot -Tpng Lt_adjacency_graph.dot -o Lt_graph.png" << std::endl;
    std::cout << "     > dot -Tsvg transitive_reduction_graph.dot -o tr_graph.svg" << std::endl;
    std::cout << "     > dot -Tpdf transitive_reduction_graph.dot -o tr_graph.pdf" << std::endl;
    std::cout << "   - Text files: View adjacency lists in any text editor" << std::endl;
  }

  std::cout << std::endl << "=== Analysis Complete ===" << std::endl;
  
  return 0;
}
