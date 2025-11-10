#include "io.hpp"
#include "sp_ops.hpp"
#include "utils.h"
#include <cxxopts.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include "graph_algs.hpp"
#include "permutation.hpp"
#include "ilum.hpp"

using namespace matrix_utils;

int main(int argc, char** argv) {
    cxxopts::Options options("test_ilum", "Test ILUMLevel and visualize all matrices");
    
    options.add_options()
        ("f,file", "Input matrix file (Matrix Market format)", 
         cxxopts::value<std::string>()->default_value("../data/ex5.mtx"))
        ("o,output", "Output file prefix (default: matrix)", 
         cxxopts::value<std::string>()->default_value("matrix"))
        ("s,size", "Maximum display size for SVG (default: 2000)", 
         cxxopts::value<int>()->default_value("2000"))
        ("t,threads", "Number of threads for A+A^T computation (default: 4)", 
         cxxopts::value<int>()->default_value("4"))
        ("tau", "Drop tolerance for ILUM (default: 0, no dropping)", 
         cxxopts::value<double>()->default_value("0.0"))
        ("h,help", "Print help");

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        std::cout << "\nExample usage:\n";
        std::cout << "  " << argv[0] << " -f ../data/ex5.mtx\n";
        std::cout << "  " << argv[0] << " -f ../data/ex5.mtx -o output -s 3000 -t 8\n";
        std::cout << "  " << argv[0] << " -f ../data/ex5.mtx --tau 0.01\n";
        std::cout << "  " << argv[0] << " (uses default: ../data/ex5.mtx)\n";
        return 0;
    }

    std::string matrix_file = result["file"].as<std::string>();
    std::string output_prefix = result["output"].as<std::string>();
    int max_display_size = result["size"].as<int>();
    int num_threads = result["threads"].as<int>();
    double tau = result["tau"].as<double>();

    std::cout << "Reading matrix from: " << matrix_file << std::endl;

    // Read the matrix
    std::ifstream f(matrix_file);
    if (!f.good()) {
        std::cerr << "Error: Cannot open file " << matrix_file << std::endl;
        return 1;
    }

    std::vector<int> ai, aj;
    std::vector<double> av;
    utils::read_matrix_market_csr(f, ai, aj, av);
    f.close();

    if (ai.size() == 0) {
        std::cerr << "Error: Could not read matrix" << std::endl;
        return 1;
    }

    const int size = ai.size() - 1;
    const int base = ai[0];
    const int nnz = ai[size] - base;

    std::cout << "Matrix properties:" << std::endl;
    std::cout << "  Size: " << size << " x " << size << std::endl;
    std::cout << "  NNZ: " << nnz << std::endl;
    std::cout << "  Base: " << base << std::endl;
    std::cout << "  Density: " << (100.0 * nnz / (size * size)) << "%" << std::endl;
    std::cout << "  Threads: " << num_threads << std::endl;
    std::cout << "  Max display size: " << max_display_size << std::endl;
    std::cout << "  Drop tolerance (tau): " << tau << std::endl;

    // Generate SVG for original matrix
    std::string original_svg = output_prefix + "_original.svg";
    std::cout << "\nGenerating original sparsity pattern: " << original_svg << std::endl;
    {
        std::ofstream svg_out(original_svg);
        if (!svg_out.good()) {
            std::cerr << "Error: Cannot create file " << original_svg << std::endl;
            return 1;
        }
        writeSVG(size, size, ai.data(), aj.data(), svg_out, max_display_size);
        svg_out.close();
    }
    std::cout << "  Written to: " << original_svg << std::endl;

    // Create and execute ILUMLevel
    std::cout << "\n=== Testing ILUMLevel ===" << std::endl;
    preconditioner::ILUMLevel<CSRMatrixVec<int, int, double>> ilum_level(num_threads, tau);
    
    std::cout << "\nExecuting ILUMLevel (reorder, split, compute Schur complement)..." << std::endl;
    ilum_level(size, ai.data(), aj.data(), av.data());
    
    std::cout << "\nILUMLevel Results:" << std::endl;
    std::cout << "  Split row: " << ilum_level._split_row << std::endl;
    std::cout << "  PAPT matrix: " << ilum_level._PAPT.rows << "x" << ilum_level._PAPT.cols 
              << ", NNZ: " << ilum_level._PAPT.NNZ() << std::endl;
    std::cout << "  D  (diagonal):     " << ilum_level._D.rows << "x" << ilum_level._D.cols 
              << ", NNZ: " << ilum_level._D.NNZ() << std::endl;
    std::cout << "  F  (top-right):    " << ilum_level._F.rows << "x" << ilum_level._F.cols 
              << ", NNZ: " << ilum_level._F.NNZ() << std::endl;
    std::cout << "  EDinv (E*D^-1):    " << ilum_level._EDinv.rows << "x" << ilum_level._EDinv.cols 
              << ", NNZ: " << ilum_level._EDinv.NNZ() << std::endl;
    std::cout << "  C  (bottom-right): " << ilum_level._C.rows << "x" << ilum_level._C.cols 
              << ", NNZ: " << ilum_level._C.NNZ() << std::endl;
    std::cout << "  ANext (Schur):     " << ilum_level._ANext.rows << "x" << ilum_level._ANext.cols 
              << ", NNZ: " << ilum_level._ANext.NNZ() << std::endl;
        std::cout<<"hello"<<std::endl;
        if (ilum_level._ANextDropped.rows > 0)
        {
            std::cout << "  ANextDropped:      " << ilum_level._ANextDropped.rows << "x"
                      << ilum_level._ANextDropped.cols
                      << ", NNZ: " << ilum_level._ANextDropped.NNZ() << " (reduced by "
                      << (100.0 * (ilum_level._ANext.NNZ() - ilum_level._ANextDropped.NNZ()) /
                          ilum_level._ANext.NNZ())
                      << "%)" << std::endl;
        }

    // Generate SVG files for all matrices
    std::cout << "\n=== Generating SVG Visualizations ===" << std::endl;
    
    // PAPT
    {
        std::string svg_file = output_prefix + "_PAPT.svg";
        std::cout << "\nGenerating PAPT (permuted matrix): " << svg_file << std::endl;
        std::ofstream svg_out(svg_file);
        if (svg_out.good()) {
            writeSVG(ilum_level._PAPT.rows, ilum_level._PAPT.cols, 
                    ilum_level._PAPT.AI(), ilum_level._PAPT.AJ(), svg_out, max_display_size);
            svg_out.close();
            std::cout << "  Written to: " << svg_file << std::endl;
        }
    }
    
    // D (diagonal block)
    {
        std::string svg_file = output_prefix + "_D.svg";
        std::cout << "\nGenerating D (diagonal block): " << svg_file << std::endl;
        std::ofstream svg_out(svg_file);
        if (svg_out.good()) {
            writeSVG(ilum_level._D.rows, ilum_level._D.cols, 
                    ilum_level._D.AI(), ilum_level._D.AJ(), svg_out, max_display_size);
            svg_out.close();
            std::cout << "  Written to: " << svg_file << std::endl;
        }
    }
    
    // F (top-right block)
    {
        std::string svg_file = output_prefix + "_F.svg";
        std::cout << "\nGenerating F (top-right block): " << svg_file << std::endl;
        std::ofstream svg_out(svg_file);
        if (svg_out.good()) {
            writeSVG(ilum_level._F.rows, ilum_level._F.cols, 
                    ilum_level._F.AI(), ilum_level._F.AJ(), svg_out, max_display_size);
            svg_out.close();
            std::cout << "  Written to: " << svg_file << std::endl;
        }
    }
    
    // EDinv (E*D^{-1}, bottom-left block)
    {
        std::string svg_file = output_prefix + "_EDinv.svg";
        std::cout << "\nGenerating EDinv (E*D^{-1}, bottom-left block): " << svg_file << std::endl;
        std::ofstream svg_out(svg_file);
        if (svg_out.good()) {
            writeSVG(ilum_level._EDinv.rows, ilum_level._EDinv.cols, 
                    ilum_level._EDinv.AI(), ilum_level._EDinv.AJ(), svg_out, max_display_size);
            svg_out.close();
            std::cout << "  Written to: " << svg_file << std::endl;
        }
    }
    
    // C (bottom-right block)
    {
        std::string svg_file = output_prefix + "_C.svg";
        std::cout << "\nGenerating C (bottom-right block): " << svg_file << std::endl;
        std::ofstream svg_out(svg_file);
        if (svg_out.good()) {
            writeSVG(ilum_level._C.rows, ilum_level._C.cols, 
                    ilum_level._C.AI(), ilum_level._C.AJ(), svg_out, max_display_size);
            svg_out.close();
            std::cout << "  Written to: " << svg_file << std::endl;
        }
    }
    
    // ANext (Schur complement)
    {
        std::string svg_file = output_prefix + "_ANext.svg";
        std::cout << "\nGenerating ANext (Schur complement): " << svg_file << std::endl;
        std::ofstream svg_out(svg_file);
        if (svg_out.good()) {
            writeSVG(ilum_level._ANext.rows, ilum_level._ANext.cols, 
                    ilum_level._ANext.AI(), ilum_level._ANext.AJ(), svg_out, max_display_size);
            svg_out.close();
            std::cout << "  Written to: " << svg_file << std::endl;
        }
    }
    
    // ANextDropped (after applying drop tolerance)
    if (ilum_level._ANextDropped.rows > 0)
    {
        std::string svg_file = output_prefix + "_ANextDropped.svg";
        std::cout << "\nGenerating ANextDropped (after drop tolerance): " << svg_file << std::endl;
        std::ofstream svg_out(svg_file);
        if (svg_out.good()) {
            writeSVG(ilum_level._ANextDropped.rows, ilum_level._ANextDropped.cols, 
                    ilum_level._ANextDropped.AI(), ilum_level._ANextDropped.AJ(), svg_out, max_display_size);
            svg_out.close();
            std::cout << "  Written to: " << svg_file << std::endl;
        }
    }

    // Apply second level with ANext as input
    std::cout << "\n=== Testing ILUMLevel - Second Level ===" << std::endl;
    preconditioner::ILUMLevel<CSRMatrixVec<int, int, double>> ilum_level2(num_threads, tau);
    
    // Use dropped matrix if available, otherwise use ANext
    bool use_dropped = (ilum_level._ANextDropped.rows > 0);
    auto& input_matrix = use_dropped ? ilum_level._ANextDropped : ilum_level._ANext;
    
    if (use_dropped) {
        std::cout << "\nExecuting ILUMLevel on ANextDropped (second level)..." << std::endl;
    } else {
        std::cout << "\nExecuting ILUMLevel on ANext (second level)..." << std::endl;
    }
    
    ilum_level2(input_matrix.rows, input_matrix.AI(), input_matrix.AJ(), input_matrix.AV());
    
    std::cout << "\nILUMLevel Level 2 Results:" << std::endl;
    std::cout << "  Split row: " << ilum_level2._split_row << std::endl;
    std::cout << "  PAPT matrix: " << ilum_level2._PAPT.rows << "x" << ilum_level2._PAPT.cols 
              << ", NNZ: " << ilum_level2._PAPT.NNZ() << std::endl;
    std::cout << "  D  (diagonal):     " << ilum_level2._D.rows << "x" << ilum_level2._D.cols 
              << ", NNZ: " << ilum_level2._D.NNZ() << std::endl;
    std::cout << "  F  (top-right):    " << ilum_level2._F.rows << "x" << ilum_level2._F.cols 
              << ", NNZ: " << ilum_level2._F.NNZ() << std::endl;
    std::cout << "  EDinv (E*D^-1):    " << ilum_level2._EDinv.rows << "x" << ilum_level2._EDinv.cols 
              << ", NNZ: " << ilum_level2._EDinv.NNZ() << std::endl;
    std::cout << "  C  (bottom-right): " << ilum_level2._C.rows << "x" << ilum_level2._C.cols 
              << ", NNZ: " << ilum_level2._C.NNZ() << std::endl;
    std::cout << "  ANext (Schur):     " << ilum_level2._ANext.rows << "x" << ilum_level2._ANext.cols 
              << ", NNZ: " << ilum_level2._ANext.NNZ() << std::endl;
    if (ilum_level2._ANextDropped.rows > 0) {
        std::cout << "  ANextDropped:      " << ilum_level2._ANextDropped.rows << "x" << ilum_level2._ANextDropped.cols 
                  << ", NNZ: " << ilum_level2._ANextDropped.NNZ() 
                  << " (reduced by " << (100.0 * (ilum_level2._ANext.NNZ() - ilum_level2._ANextDropped.NNZ()) / ilum_level2._ANext.NNZ()) << "%)" << std::endl;
    }

    // Generate SVG files for second level matrices
    std::cout << "\n=== Generating Level 2 SVG Visualizations ===" << std::endl;
    
    // Level 2 PAPT
    {
        std::string svg_file = output_prefix + "_L2_PAPT.svg";
        std::cout << "\nGenerating L2 PAPT: " << svg_file << std::endl;
        std::ofstream svg_out(svg_file);
        if (svg_out.good()) {
            writeSVG(ilum_level2._PAPT.rows, ilum_level2._PAPT.cols, 
                    ilum_level2._PAPT.AI(), ilum_level2._PAPT.AJ(), svg_out, max_display_size);
            svg_out.close();
            std::cout << "  Written to: " << svg_file << std::endl;
        }
    }
    
    // Level 2 D
    {
        std::string svg_file = output_prefix + "_L2_D.svg";
        std::cout << "\nGenerating L2 D: " << svg_file << std::endl;
        std::ofstream svg_out(svg_file);
        if (svg_out.good()) {
            writeSVG(ilum_level2._D.rows, ilum_level2._D.cols, 
                    ilum_level2._D.AI(), ilum_level2._D.AJ(), svg_out, max_display_size);
            svg_out.close();
            std::cout << "  Written to: " << svg_file << std::endl;
        }
    }
    
    // Level 2 F
    {
        std::string svg_file = output_prefix + "_L2_F.svg";
        std::cout << "\nGenerating L2 F: " << svg_file << std::endl;
        std::ofstream svg_out(svg_file);
        if (svg_out.good()) {
            writeSVG(ilum_level2._F.rows, ilum_level2._F.cols, 
                    ilum_level2._F.AI(), ilum_level2._F.AJ(), svg_out, max_display_size);
            svg_out.close();
            std::cout << "  Written to: " << svg_file << std::endl;
        }
    }
    
    // Level 2 EDinv
    {
        std::string svg_file = output_prefix + "_L2_EDinv.svg";
        std::cout << "\nGenerating L2 EDinv: " << svg_file << std::endl;
        std::ofstream svg_out(svg_file);
        if (svg_out.good()) {
            writeSVG(ilum_level2._EDinv.rows, ilum_level2._EDinv.cols, 
                    ilum_level2._EDinv.AI(), ilum_level2._EDinv.AJ(), svg_out, max_display_size);
            svg_out.close();
            std::cout << "  Written to: " << svg_file << std::endl;
        }
    }
    
    // Level 2 C
    {
        std::string svg_file = output_prefix + "_L2_C.svg";
        std::cout << "\nGenerating L2 C: " << svg_file << std::endl;
        std::ofstream svg_out(svg_file);
        if (svg_out.good()) {
            writeSVG(ilum_level2._C.rows, ilum_level2._C.cols, 
                    ilum_level2._C.AI(), ilum_level2._C.AJ(), svg_out, max_display_size);
            svg_out.close();
            std::cout << "  Written to: " << svg_file << std::endl;
        }
    }
    
    // Level 2 ANext
    {
        std::string svg_file = output_prefix + "_L2_ANext.svg";
        std::cout << "\nGenerating L2 ANext: " << svg_file << std::endl;
        std::ofstream svg_out(svg_file);
        if (svg_out.good()) {
            writeSVG(ilum_level2._ANext.rows, ilum_level2._ANext.cols, 
                    ilum_level2._ANext.AI(), ilum_level2._ANext.AJ(), svg_out, max_display_size);
            svg_out.close();
            std::cout << "  Written to: " << svg_file << std::endl;
        }
    }
    
    // Level 2 ANextDropped
    if (ilum_level2._ANextDropped.rows > 0) {
        std::string svg_file = output_prefix + "_L2_ANextDropped.svg";
        std::cout << "\nGenerating L2 ANextDropped: " << svg_file << std::endl;
        std::ofstream svg_out(svg_file);
        if (svg_out.good()) {
            writeSVG(ilum_level2._ANextDropped.rows, ilum_level2._ANextDropped.cols, 
                    ilum_level2._ANextDropped.AI(), ilum_level2._ANextDropped.AJ(), svg_out, max_display_size);
            svg_out.close();
            std::cout << "  Written to: " << svg_file << std::endl;
        }
    }

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Generated SVG files:" << std::endl;
    std::cout << "\nLevel 1:" << std::endl;
    std::cout << "  1. " << output_prefix << "_original.svg (original matrix A)" << std::endl;
    std::cout << "  2. " << output_prefix << "_PAPT.svg (permuted matrix P*A*P^T)" << std::endl;
    std::cout << "  3. " << output_prefix << "_D.svg (diagonal block)" << std::endl;
    std::cout << "  4. " << output_prefix << "_F.svg (top-right block)" << std::endl;
    std::cout << "  5. " << output_prefix << "_EDinv.svg (E*D^{-1}, bottom-left)" << std::endl;
    std::cout << "  6. " << output_prefix << "_C.svg (bottom-right block)" << std::endl;
    std::cout << "  7. " << output_prefix << "_ANext.svg (Schur complement C-EDinv*F)" << std::endl;
    std::cout << "  8. " << output_prefix << "_ANextDropped.svg (after drop tolerance, if tau > 0)" << std::endl;
    std::cout << "\nLevel 2:" << std::endl;
    std::cout << "  9. " << output_prefix << "_L2_PAPT.svg (permuted ANext)" << std::endl;
    std::cout << " 10. " << output_prefix << "_L2_D.svg (diagonal block)" << std::endl;
    std::cout << " 11. " << output_prefix << "_L2_F.svg (top-right block)" << std::endl;
    std::cout << " 12. " << output_prefix << "_L2_EDinv.svg (E*D^{-1}, bottom-left)" << std::endl;
    std::cout << " 13. " << output_prefix << "_L2_C.svg (bottom-right block)" << std::endl;
    std::cout << " 14. " << output_prefix << "_L2_ANext.svg (Schur complement)" << std::endl;
    std::cout << " 15. " << output_prefix << "_L2_ANextDropped.svg (after drop tolerance, if tau > 0)" << std::endl;
    std::cout << "\nDisplay resolution: " << max_display_size << "x" << max_display_size << std::endl;
    std::cout << "Open the SVG files in a web browser to view the sparsity patterns." << std::endl;

    return 0;
}
