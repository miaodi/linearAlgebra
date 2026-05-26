#include "../utils/utils.h"
#include <gtest/gtest.h>
#include <iomanip>
#include <memory>
#include <vector>
#include "../utils/variadic_sort.hpp"

TEST( Utils, knuth_s )
{
    std::random_device dev;
    std::mt19937 rng( dev() );
    std::uniform_int_distribution<std::mt19937::result_type> dist1( 0, 100000000 ); // distribution in range [1, 100000000]
    size_t size = dist1( rng );
    size_t lower_bound = dist1( rng );
    size_t upper_bound = std::max( size + lower_bound, dist1( rng ) );

    std::vector<int> randVec( size, 0 );

    utils::knuth_s rand;
    rand( size, lower_bound, upper_bound, randVec.begin() );
    for ( auto i : randVec )
    {
        EXPECT_GE( i, lower_bound );
        EXPECT_LT( i, upper_bound );
    }
}

TEST( Utils, MaxHeap )
{
    auto compMax = []( const int v1, const int v2 ) { return v1 > v2; };

    utils::MaxHeap<int, decltype( compMax )> max_heap( compMax );

    for ( int i = 10; i >= 0; i-- )
    {
        max_heap.push( i );
        EXPECT_EQ( i, *max_heap.top() );
    }
    max_heap.clear();

    for ( int i = 10; i >= 0; i-- )
    {
        max_heap.push( i );
        if ( max_heap.size() > 2 )
        {
            max_heap.pop();
            EXPECT_EQ( 9, *max_heap.top() );
        }
    }
}

TEST( sort, insertion_sort )
{
    int size = 100;
    std::random_device rd;
    std::mt19937 gen( rd() );
    std::uniform_int_distribution<int> dis( std::numeric_limits<int>::min(), std::numeric_limits<int>::max() );
    std::vector<int> vec( size );
    std::generate( vec.begin(), vec.end(), [&] { return dis( gen ); } );

    utils::variadic_insertion_sort( 0, size, vec.data() );
    EXPECT_TRUE( std::is_sorted( vec.begin(), vec.end() ) );
}

TEST( sort, partition )
{
    int size = 100;
    std::random_device rd;
    std::mt19937 gen( rd() );
    std::uniform_int_distribution<int> dis( std::numeric_limits<int>::min(), std::numeric_limits<int>::max() );
    std::vector<int> vec( size );
    std::generate( vec.begin(), vec.end(), [&] { return dis( gen ); } );
    int pivot_val = vec[size - 1];
    auto pivot = utils::variadic_partition( 0, size, vec.data() );
    for ( int i = 0; i < pivot; i++ )
    {
        EXPECT_LE( vec[i], pivot_val );
    }
    for ( int i = pivot + 1; i < size; i++ )
    {
        EXPECT_GE( vec[i], pivot_val );
    }
}

TEST( sort, quicksort )
{
    int size = std::rand() % 1000;
    std::random_device rd;
    std::mt19937 gen( rd() );
    std::uniform_int_distribution<int> dis( std::numeric_limits<int>::min(), std::numeric_limits<int>::max() );
    std::vector<int> vec( size );
    std::generate( vec.begin(), vec.end(), [&] { return dis( gen ); } );

    utils::variadic_quick_sort( 0, size, vec.data() );
    EXPECT_TRUE( std::is_sorted( vec.begin(), vec.end() ) );
}

TEST( sort, quicksort_2 )
{
    int size = 10000;
    std::random_device rd;
    std::mt19937 gen( rd() );
    std::uniform_int_distribution<int> dis( std::numeric_limits<int>::min(), std::numeric_limits<int>::max() );
    std::vector<int> vec( size );
    std::generate( vec.begin(), vec.end(), [&] { return dis( gen ); } );
    std::vector<int> val( vec );

    utils::variadic_quick_sort( 0, size, vec.data(), val.data() );
    EXPECT_TRUE( vec == val );
}

TEST( Utils, ParallelPrefixSum )
{
    const int size = 1000;
    std::vector<int32_t> input( size );
    std::iota( input.begin(), input.end(), 1 ); // 1, 2, 3, ..., 1000

    std::vector<int32_t> output( size + 1 );

    std::vector<int> bases = { 0, 1 };
    std::vector<int> thread_counts = { 1, 2, 4, 8 };

    for ( int base : bases )
    {
        for ( int nthreads : thread_counts )
        {
            std::fill( output.begin() + 1, output.end(), 0 );
            output[0] = base;

            utils::ParallelPrefixSum( nthreads, input.data(), input.data() + size, output.data() );

            // Verify prefix sum: output[i] should be base + sum of input[0..i-1]
            int32_t expected_sum = base;
            EXPECT_EQ( output[0], base );
            for ( int i = 0; i < size; ++i )
            {
                expected_sum += input[i];
                EXPECT_EQ( output[i + 1], expected_sum )
                    << "Failed at index " << i + 1 << " with base=" << base << " and " << nthreads
                    << " threads";
            }
        }
    }
}

TEST( Utils, ParallelPrefixSum_Large )
{
    const int size = 10000;
    std::vector<int64_t> input( size );
    std::iota( input.begin(), input.end(), 1 );

    std::vector<int64_t> output( size + 1 );

    std::vector<int> bases = { 0, 1 };
    std::vector<int> thread_counts = { 1, 2, 3, 4, 6, 8, 12 };

    for ( int base : bases )
    {
        for ( int nthreads : thread_counts )
        {
            std::fill( output.begin() + 1, output.end(), 0 );
            output[0] = base;

            utils::ParallelPrefixSum( nthreads, input.data(), input.data() + size, output.data() );

            // Verify correctness
            int64_t expected_sum = base;
            for ( int i = 0; i < size; ++i )
            {
                expected_sum += input[i];
                EXPECT_EQ( output[i + 1], expected_sum )
                    << "Failed at index " << i + 1 << " with base=" << base << " and " << nthreads
                    << " threads";
            }
        }
    }
}

TEST( Utils, ParallelPrefixSum_RandomData )
{
    const int size = 5000;
    std::random_device rd;
    std::mt19937 gen( rd() );
    std::uniform_int_distribution<int32_t> dis( 1, 100 );

    std::vector<int32_t> input( size );
    std::generate( input.begin(), input.end(), [&] { return dis( gen ); } );

    std::vector<int> bases = { 0, 1 };
    std::vector<int> thread_counts = { 1, 2, 4, 8 };

    for ( int base : bases )
    {
        for ( int nthreads : thread_counts )
        {
            std::vector<int32_t> output( size + 1 );
            output[0] = base;

            utils::ParallelPrefixSum( nthreads, input.data(), input.data() + size, output.data() );

            // Verify correctness
            int32_t expected_sum = base;
            EXPECT_EQ( output[0], base );
            for ( int i = 0; i < size; ++i )
            {
                expected_sum += input[i];
                EXPECT_EQ( output[i + 1], expected_sum )
                    << "Failed at index " << i + 1 << " with base=" << base << " and " << nthreads
                    << " threads";
            }
        }
    }
}

TEST( Utils, ParallelPrefixSumInplace )
{
    const int size = 256;
    std::vector<int32_t> input( size );
    std::iota( input.begin(), input.end(), 1 ); // 1..256

    std::vector<int> thread_counts = { 1, 2, 4, 8 };
    for ( int nthreads : thread_counts )
    {
        auto data = input;
        utils::ParallelPrefixSumInplace( nthreads, data.data(), data.data() + data.size() );

        int32_t running = 0;
        for ( int i = 0; i < size; ++i )
        {
            running += input[i];
            EXPECT_EQ( data[i], running ) << "nthreads=" << nthreads << " idx=" << i;
        }
    }
}

TEST( Utils, ParallelPrefixSumInplace_Random )
{
    const int size = 513;
    std::mt19937 gen( 42 );
    std::uniform_int_distribution<int32_t> dis( 0, 10 );
    std::vector<int32_t> input( size );
    std::generate( input.begin(), input.end(), [&] { return dis( gen ); } );

    for ( int nthreads : { 1, 3, 4 } )
    {
        auto data = input;
        utils::ParallelPrefixSumInplace( nthreads, data.begin(), data.end() );

        int32_t ref = 0;
        for ( int i = 0; i < size; ++i )
        {
            ref += input[i];
            EXPECT_EQ( data[i], ref ) << "nthreads=" << nthreads << " idx=" << i;
        }
    }
}

TEST( Utils, ParallelPrefixSum_EdgeCases )
{
    // Test with size 1
    {
        std::vector<int32_t> input = { 42 };
        std::vector<int32_t> output( 2 );
        output[0] = 0;

        utils::ParallelPrefixSum( 4, input.data(), input.data() + 1, output.data() );
        EXPECT_EQ( output[0], 0 );
        EXPECT_EQ( output[1], 42 );
    }

    // Test with base 1
    {
        std::vector<int32_t> input = { 42 };
        std::vector<int32_t> output( 2 );
        output[0] = 1;

        utils::ParallelPrefixSum( 4, input.data(), input.data() + 1, output.data() );
        EXPECT_EQ( output[0], 1 );
        EXPECT_EQ( output[1], 43 );
    }

    // Test with all zeros
    {
        std::vector<int32_t> input( 100, 0 );
        std::vector<int32_t> output( 101 );
        output[0] = 0;

        utils::ParallelPrefixSum( 4, input.data(), input.data() + 100, output.data() );

        for ( int i = 0; i <= 100; ++i )
        {
            EXPECT_EQ( output[i], 0 );
        }
    }
}
