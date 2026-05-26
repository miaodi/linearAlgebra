#include "io.hpp"
#include "matrix_utils.hpp"
#include "sparse_mat_traits.hpp"
#include "graph_algs.hpp"
#include <cxxopts.hpp>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <numeric>

int main( int argc, char* argv[] )
{
    cxxopts::Options options( "print_matrix_svg", "Read a matrix and print it as SVG" );

    // clang-format off
    options.add_options()
        ( "f,filename", "Matrix Market file to read", cxxopts::value<std::string>()->default_value( "../tests/data/ex5.mtx" ) )
        ( "o,output", "Output SVG file path", cxxopts::value<std::string>()->default_value( "matrix.svg" ) )
        ( "s,size", "Maximum display size (pixels)", cxxopts::value<int>()->default_value( "2000" ) )
        ( "t,threshold", "Pruning threshold (absolute value)", cxxopts::value<double>()->default_value( "0.0" ) )
        ( "scc", "Find and display strongly connected components", cxxopts::value<bool>()->default_value( "false" ) )
        ( "h,help", "Print usage" );
    // clang-format on

    auto result = options.parse( argc, argv );

    if ( result.count( "help" ) )
    {
        std::cout << options.help() << std::endl;
        return 0;
    }

    std::string filename = result["filename"].as<std::string>();
    std::string output_file = result["output"].as<std::string>();
    int max_display_size = result["size"].as<int>();
    double threshold = result["threshold"].as<double>();
    bool find_scc = result["scc"].as<bool>();

    std::cout << "Options:" << std::endl;
    std::cout << "  filename: " << filename << std::endl;
    std::cout << "  output: " << output_file << std::endl;
    std::cout << "  max_display_size: " << max_display_size << std::endl;
    std::cout << "  threshold: " << threshold << std::endl;
    std::cout << "  find_scc: " << ( find_scc ? "true" : "false" ) << std::endl;

    // Validate max_display_size
    if ( max_display_size <= 0 )
    {
        std::cerr << "Invalid max_display_size: " << max_display_size
                  << ". Must be a positive integer." << std::endl;
        return -1;
    }
    omp_set_num_threads( 8 );
    std::ifstream f( filename );
    if ( !f.is_open() )
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return -1;
    }

    f.clear();
    f.seekg( 0, std::ios::beg );
    matrix_utils::CSRMatrix<int, int, double> csr_matrix;
    matrix_utils::readMatrixMarket( f, csr_matrix );
    f.close();

    std::cout << "Matrix: " << csr_matrix.rows << " x " << csr_matrix.cols
              << ", NNZ: " << csr_matrix.NNZ() << std::endl;

    std::cout << "\nPruning matrix with thresholds (a_ii * a_jj * " << threshold << ")..." << std::endl;
    int original_nnz = csr_matrix.NNZ();
    matrix_utils::DiagonalScaledPrune( csr_matrix.rows, csr_matrix.AI(), csr_matrix.AJ(),
                                       csr_matrix.AV(), threshold );
    int pruned_nnz = csr_matrix.NNZ();
    std::cout << "Original NNZ: " << original_nnz << std::endl;
    std::cout << "Pruned NNZ: " << pruned_nnz << std::endl;
    std::cout << "Removed entries: " << ( original_nnz - pruned_nnz ) << std::endl;
    if ( original_nnz > 0 )
    {
        std::cout << "Retention rate: " << ( static_cast<double>( pruned_nnz ) / original_nnz * 100.0 )
                  << "%" << std::endl;
    }

    // Write SVG to file
    std::ofstream out( output_file );
    if ( !out.is_open() )
    {
        std::cerr << "Failed to create output file: " << output_file << std::endl;
        return -1;
    }

    matrix_utils::writeSVG( csr_matrix.rows, csr_matrix.cols, csr_matrix.AI(), csr_matrix.AJ(), out, max_display_size );
    out.close();

    std::cout << "SVG written to: " << output_file << std::endl;

    // Find strongly connected components if requested
    if ( find_scc )
    {
        std::cout << "\nFinding strongly connected components..." << std::endl;

        const int base = csr_matrix.AI()[0];
        const int rows = csr_matrix.rows;

        std::vector<int> scc_prefix( rows + 1 );
        std::vector<int> scc_to_node( rows );
        std::vector<int> node_to_scc( rows );

        int num_sccs = graph::FindStronglyConnectedComponents( rows, csr_matrix.AI(), csr_matrix.AJ(),
                                                               scc_prefix.data(), scc_to_node.data(),
                                                               node_to_scc.data() );

        std::cout << "Number of strongly connected components: " << num_sccs << std::endl;

        // Compute SCC sizes
        std::vector<int> scc_sizes( num_sccs );
        for ( int scc_id = 0; scc_id < num_sccs; ++scc_id )
        {
            scc_sizes[scc_id] = scc_prefix[scc_id + 1] - scc_prefix[scc_id];
            // std::cout << "  SCC " << scc_id + base
            //           << " size: " << scc_sizes[scc_id] << std::endl;
        }

        // Find largest SCC
        int max_size = 0;
        int max_scc_id = 0;
        for ( int scc_id = 0; scc_id < num_sccs; ++scc_id )
        {
            if ( scc_sizes[scc_id] > max_size )
            {
                max_size = scc_sizes[scc_id];
                max_scc_id = scc_id;
            }
        }

        // Count trivial SCCs (size 1)
        int trivial_count = 0;
        for ( int size : scc_sizes )
        {
            if ( size == 1 )
                ++trivial_count;
        }

        std::cout << "Trivial SCCs (size 1): " << trivial_count << std::endl;
        std::cout << "Non-trivial SCCs: " << ( num_sccs - trivial_count ) << std::endl;
        std::cout << "Largest SCC: SCC " << max_scc_id + base << " with " << max_size << " nodes" << std::endl;

        // Print size distribution statistics
        std::cout << "\nSCC Size Distribution:" << std::endl;
        std::map<int, int> size_histogram;
        for ( int size : scc_sizes )
        {
            size_histogram[size]++;
        }
        for ( const auto& [size, count] : size_histogram )
        {
            std::cout << "  Size " << size << ": " << count << " SCC(s)" << std::endl;
        }

        // Compute statistics
        double mean_size = static_cast<double>( rows ) / num_sccs;
        double variance = 0.0;
        for ( int size : scc_sizes )
        {
            double diff = size - mean_size;
            variance += diff * diff;
        }
        variance /= num_sccs;
        double std_dev = std::sqrt( variance );

        std::cout << "\nSCC Size Statistics:" << std::endl;
        std::cout << "  Mean: " << mean_size << std::endl;
        std::cout << "  Std Dev: " << std_dev << std::endl;
        std::cout << "  Min: " << *std::min_element( scc_sizes.begin(), scc_sizes.end() ) << std::endl;
        std::cout << "  Max: " << max_size << std::endl;

        // Project the original graph to SCC graph using ProjectGraphToTaskGraph
        std::cout << "\nProjecting graph to SCC graph..." << std::endl;

        graph::ProjectGraphToTaskGraph<int, int, true> projector( 8 ); // Single-threaded for now

        // Allocate arrays for the projected SCC graph
        std::vector<int> scc_ai( num_sccs + 1 );
        // Worst case: every edge becomes a task edge, but typically much less
        std::vector<int> scc_aj( csr_matrix.NNZ() );

        int scc_edges = projector( rows,               // work_graph_rows (original nodes)
                                   csr_matrix.AI(),    // work_ai (original graph row pointers)
                                   csr_matrix.AJ(),    // work_aj (original graph column indices)
                                   num_sccs,           // num_tasks (number of SCCs)
                                   scc_prefix.data(),  // task_prefix (SCC-to-node mapping)
                                   scc_to_node.data(), // task_to_node (nodes in each SCC)
                                   node_to_scc.data(), // node_to_task (node-to-SCC mapping)
                                   scc_ai.data(),      // task_ai (output: SCC graph row pointers)
                                   scc_aj.data()       // task_aj (output: SCC graph column indices)
        );

        std::cout << "SCC graph edges: " << scc_edges << std::endl;
        std::cout << "Original graph edges: " << csr_matrix.NNZ() << std::endl;
        std::cout << "Reduction ratio: "
                  << ( csr_matrix.NNZ() > 0 ? static_cast<double>( scc_edges ) / csr_matrix.NNZ() : 0.0 )
                  << std::endl;

        // Topologically sort the SCC condensation graph to get level sets
        const int scc_base = scc_ai[0];
        bool is_lower_triangular = true;
        for ( int row = 0; row < num_sccs && is_lower_triangular; ++row )
        {
            for ( int j = scc_ai[row] - scc_base; j < scc_ai[row + 1] - scc_base; ++j )
            {
                if ( scc_aj[j] > row + scc_base )
                {
                    is_lower_triangular = false;
                    break;
                }
            }
        }
        // Check if SCC graph has all diagonal terms
        std::cout << "\nChecking for diagonal terms in SCC graph..." << std::endl;
        int diagonal_count = 0;
        for ( int scc_id = 0; scc_id < num_sccs; ++scc_id )
        {
            for ( int j = scc_ai[scc_id] - scc_base; j < scc_ai[scc_id + 1] - scc_base; ++j )
            {
                if ( scc_aj[j] == scc_id + scc_base )
                {
                    ++diagonal_count;
                    break;
                }
            }
        }
        std::cout << "SCCs with self-loops: " << diagonal_count << " / " << num_sccs << std::endl;
        if ( diagonal_count == num_sccs )
        {
            std::cout << "All SCCs have diagonal terms (self-loops)" << std::endl;
        }
        else
        {
            std::cout << "Missing diagonal terms in " << ( num_sccs - diagonal_count ) << " SCC(s)"
                      << std::endl;
        }

        std::vector<int> scc_perm( num_sccs );
        std::vector<int> scc_level_prefix( num_sccs + 1 );
        int scc_levels = 0;

        if ( is_lower_triangular )
        {
            graph::TopologicalSort2<int, int, matrix_utils::TriangularMatrix::L> topo;
            scc_levels =
                topo( num_sccs, scc_ai.data(), scc_aj.data(), scc_perm.data(), scc_level_prefix.data() );

            // matrix_utils::KahnParallel<int, int> kahn(8);
            // scc_levels = kahn(num_sccs, scc_ai.data(), scc_aj.data(), scc_perm.data(),
            //                   scc_level_prefix.data());
        }
        else
        {
            std::cout << "SCC graph not lower triangular; falling back to Kahn topological order." << std::endl;
            graph::KahnParallel<int, int> kahn_parallel( omp_get_max_threads() );
            scc_levels = kahn_parallel( num_sccs, scc_ai.data(), scc_aj.data(), scc_perm.data(),
                                        scc_level_prefix.data() );
        }
        std::cout << "SCC graph levels: " << scc_levels << std::endl;
        for ( int lvl = 0; lvl < scc_levels; ++lvl )
        {
            int sz = scc_level_prefix[lvl + 1] - scc_level_prefix[lvl];
            int nodes_in_level = 0;
            for ( int idx = scc_level_prefix[lvl] - scc_ai[0];
                  idx < scc_level_prefix[lvl + 1] - scc_ai[0]; ++idx )
            {
                const int scc_id = scc_perm[idx] - scc_ai[0];
                nodes_in_level += scc_sizes[scc_id];
            }
            std::cout << "  Level " << lvl << ": " << sz << " SCC(s), " << nodes_in_level
                      << " node(s)" << std::endl;
        }

        std::vector<int> node_perm( rows );
        std::vector<int> node_iperm( rows );
        graph::BuildPermutationFromSccLevels( num_sccs, scc_prefix.data(), scc_to_node.data(),
                                              scc_perm.data(), scc_level_prefix.data(), scc_levels,
                                              node_perm.data(), node_iperm.data() );

        std::cout << "\nBuilt node permutation grouped by SCC level order." << std::endl;

        // Build permuted matrix (rows and columns permuted by SCC/level order)
        matrix_utils::CSRMatrix<int, int, double> permuted;
        permuted.rows = rows;
        permuted.cols = csr_matrix.cols;
        permuted.ResizeAI( rows + 1 );
        permuted.ResizeAJ( pruned_nnz );
        permuted.ResizeAV( pruned_nnz );

        matrix_utils::permuteMat( rows, csr_matrix.cols,
                                  node_perm.data(),  // iperm: new row -> old row
                                  node_iperm.data(), // perm: old col -> new col
                                  csr_matrix.AI(), csr_matrix.AJ(), csr_matrix.AV(), permuted.AI(),
                                  permuted.AJ(), permuted.AV() );

        // Write permuted matrix to SVG
        std::string perm_svg_file = output_file;
        size_t dot_pos = perm_svg_file.rfind( '.' );
        if ( dot_pos != std::string::npos )
        {
            perm_svg_file.insert( dot_pos, "_perm" );
        }
        else
        {
            perm_svg_file += "_perm";
        }

        std::ofstream perm_out( perm_svg_file );
        if ( perm_out.is_open() )
        {
            matrix_utils::writeSVG( permuted.rows, permuted.cols, permuted.AI(), permuted.AJ(),
                                    perm_out, max_display_size );
            perm_out.close();
            std::cout << "Permuted matrix SVG written to: " << perm_svg_file << std::endl;
        }
        else
        {
            std::cerr << "Failed to create permuted SVG output file: " << perm_svg_file << std::endl;
        }

        // Write SCC graph to SVG
        std::string scc_svg_file = output_file;
        dot_pos = scc_svg_file.rfind( '.' );
        if ( dot_pos != std::string::npos )
        {
            scc_svg_file.insert( dot_pos, "_scc" );
        }
        else
        {
            scc_svg_file += "_scc";
        }

        std::ofstream scc_out( scc_svg_file );
        if ( scc_out.is_open() )
        {
            matrix_utils::writeSVG( num_sccs, num_sccs, scc_ai.data(), scc_aj.data(), scc_out, max_display_size );
            scc_out.close();
            std::cout << "SCC graph SVG written to: " << scc_svg_file << std::endl;
        }
        else
        {
            std::cerr << "Failed to create SCC SVG output file: " << scc_svg_file << std::endl;
        }
    }

    return 0;
}
