#include "circularbuffer.hpp"
#include "utils.h"
#include <algorithm>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

// Demonstrate some basic assertions.
TEST( circular_buffer, BasicAssertions )
{
    // Expect two strings not to be equal.
    EXPECT_STRNE( "hello", "world" );
}

// Demonstrate some basic assertions.
TEST( circular_buffer, unshift )
{
    utils::CircularBuffer<int> cb( 5 );
    EXPECT_TRUE( cb.push_front_overwrite( 1 ) );
    EXPECT_EQ( cb.size(), 1 );
    EXPECT_EQ( cb.available(), 4 );
    EXPECT_TRUE( cb.push_front_overwrite( 2 ) );
    EXPECT_EQ( cb.size(), 2 );
    EXPECT_EQ( cb.available(), 3 );
    EXPECT_TRUE( cb.push_front_overwrite( 3 ) );
    EXPECT_EQ( cb.size(), 3 );
    EXPECT_EQ( cb.available(), 2 );
    EXPECT_TRUE( cb.push_front_overwrite( 4 ) );
    EXPECT_EQ( cb.size(), 4 );
    EXPECT_EQ( cb.available(), 1 );
    EXPECT_TRUE( cb.push_front_overwrite( 5 ) );
    EXPECT_EQ( cb.size(), 5 );
    EXPECT_EQ( cb.available(), 0 );
    EXPECT_FALSE( cb.push_front_overwrite( 6 ) );
    EXPECT_EQ( cb.size(), 5 );
    EXPECT_EQ( cb.available(), 0 );
}

// Demonstrate some basic assertions.
TEST( circular_buffer, push )
{
    utils::CircularBuffer<int> cb( 5 );
    EXPECT_TRUE( cb.push_back_overwrite( 1 ) );
    EXPECT_EQ( cb.size(), 1 );
    EXPECT_EQ( cb.available(), 4 );
    EXPECT_TRUE( cb.push_back_overwrite( 2 ) );
    EXPECT_EQ( cb.size(), 2 );
    EXPECT_EQ( cb.available(), 3 );
    EXPECT_TRUE( cb.push_back_overwrite( 3 ) );
    EXPECT_EQ( cb.size(), 3 );
    EXPECT_EQ( cb.available(), 2 );
    EXPECT_TRUE( cb.push_back_overwrite( 4 ) );
    EXPECT_EQ( cb.size(), 4 );
    EXPECT_EQ( cb.available(), 1 );
    EXPECT_TRUE( cb.push_back_overwrite( 5 ) );
    EXPECT_EQ( cb.size(), 5 );
    EXPECT_EQ( cb.available(), 0 );
    EXPECT_FALSE( cb.push_back_overwrite( 6 ) );
    EXPECT_EQ( cb.size(), 5 );
    EXPECT_EQ( cb.available(), 0 );
}

TEST( circular_buffer, random_access )
{
    utils::CircularBuffer<int> cb( 5 );
    cb.push_front_overwrite( 5 );
    cb.push_front_overwrite( 4 );
    cb.push_front_overwrite( 3 );
    EXPECT_EQ( cb[0], 3 );
    EXPECT_EQ( cb[1], 4 );
    EXPECT_EQ( cb[2], 5 );
    EXPECT_EQ( cb.first(), 3 );
    EXPECT_EQ( cb.last(), 5 );
    EXPECT_THROW( cb[3], std::out_of_range );

    cb.push_back_overwrite( 6 );
    EXPECT_EQ( cb.first(), 3 );
    EXPECT_EQ( cb.last(), 6 );
    cb.push_back_overwrite( 7 );
    EXPECT_EQ( cb.first(), 3 );
    EXPECT_EQ( cb.last(), 7 );
    cb.push_back_overwrite( 8 );
    EXPECT_EQ( cb.first(), 4 );
    EXPECT_EQ( cb.last(), 8 );
}

TEST( circular_buffer, copy )
{
    utils::CircularBuffer<int> cb( 5 );
    cb.push_front_overwrite( 5 );
    cb.push_front_overwrite( 4 );
    cb.push_front_overwrite( 3 );
    cb.push_back_overwrite( 6 );
    cb.push_back_overwrite( 7 );
    cb.push_back_overwrite( 8 );
    std::vector<int> cp( 5 );
    cb.dump_to_vector( cp );
    ASSERT_THAT( cp, testing::ElementsAre( 4, 5, 6, 7, 8 ) );
    cb.push_front_overwrite( 2 );
    cb.dump_to_vector( cp );
    ASSERT_THAT( cp, testing::ElementsAre( 2, 4, 5, 6, 7 ) );
}

TEST( circular_buffer, resize )
{
    utils::CircularBuffer<int> cb( 5 );
    cb.push_front_overwrite( 5 );
    cb.push_front_overwrite( 4 );
    cb.push_front_overwrite( 3 );
    cb.push_back_overwrite( 6 );
    cb.push_back_overwrite( 7 );
    cb.push_back_overwrite( 8 );
    cb.resize( 6 );
    std::vector<int> cp( 6, -1 );
    cb.dump_to_vector( cp );
    // std::copy(cp.begin(), cp.end(), std::ostream_iterator<int>(std::cout, "
    // "));
    ASSERT_THAT( cp, testing::ElementsAre( 4, 5, 6, 7, 8, -1 ) );
    // cb.unshift(2);
    // cb.dump_to_vector(cp);
    // ASSERT_THAT(cp, testing::ElementsAre(2, 4, 5, 6, 7));
}

TEST( circular_buffer, auto_resize_push_front )
{
    utils::CircularBuffer<int> cb( 2 );
    cb.push_front( 1 );
    EXPECT_EQ( cb.size(), 1 );
    cb.push_front( 2 );
    EXPECT_EQ( cb.size(), 2 );
    // Should auto-resize from 2 to 4
    cb.push_front( 3 );
    EXPECT_EQ( cb.size(), 3 );
    EXPECT_GE( cb.capacity(), 3 );
    EXPECT_EQ( cb[0], 3 );
    EXPECT_EQ( cb[1], 2 );
    EXPECT_EQ( cb[2], 1 );
}

TEST( circular_buffer, auto_resize_push_back )
{
    utils::CircularBuffer<int> cb( 2 );
    cb.push_back( 1 );
    EXPECT_EQ( cb.size(), 1 );
    cb.push_back( 2 );
    EXPECT_EQ( cb.size(), 2 );
    // Should auto-resize from 2 to 4
    cb.push_back( 3 );
    EXPECT_EQ( cb.size(), 3 );
    EXPECT_GE( cb.capacity(), 3 );
    EXPECT_EQ( cb[0], 1 );
    EXPECT_EQ( cb[1], 2 );
    EXPECT_EQ( cb[2], 3 );
}

TEST( circular_buffer, pop_front )
{
    utils::CircularBuffer<int> cb( 5 );
    cb.push_back_overwrite( 1 );
    cb.push_back_overwrite( 2 );
    cb.push_back_overwrite( 3 );
    EXPECT_EQ( cb.size(), 3 );

    EXPECT_EQ( cb.pop_front(), 1 );
    EXPECT_EQ( cb.size(), 2 );
    EXPECT_EQ( cb.pop_front(), 2 );
    EXPECT_EQ( cb.size(), 1 );
    EXPECT_EQ( cb.pop_front(), 3 );
    EXPECT_EQ( cb.size(), 0 );
}

TEST( circular_buffer, pop_back )
{
    utils::CircularBuffer<int> cb( 5 );
    cb.push_back_overwrite( 1 );
    cb.push_back_overwrite( 2 );
    cb.push_back_overwrite( 3 );
    EXPECT_EQ( cb.size(), 3 );

    EXPECT_EQ( cb.pop_back(), 3 );
    EXPECT_EQ( cb.size(), 2 );
    EXPECT_EQ( cb.pop_back(), 2 );
    EXPECT_EQ( cb.size(), 1 );
    EXPECT_EQ( cb.pop_back(), 1 );
    EXPECT_EQ( cb.size(), 0 );
}

TEST( circular_buffer, empty_and_full )
{
    utils::CircularBuffer<int> cb( 3 );
    EXPECT_TRUE( cb.empty() );
    EXPECT_FALSE( cb.full() );

    cb.push_back_overwrite( 1 );
    EXPECT_FALSE( cb.empty() );
    EXPECT_FALSE( cb.full() );

    cb.push_back_overwrite( 2 );
    cb.push_back_overwrite( 3 );
    EXPECT_FALSE( cb.empty() );
    EXPECT_TRUE( cb.full() );

    cb.pop_front();
    EXPECT_FALSE( cb.empty() );
    EXPECT_FALSE( cb.full() );

    cb.pop_front();
    cb.pop_front();
    EXPECT_TRUE( cb.empty() );
    EXPECT_FALSE( cb.full() );
}

TEST( circular_buffer, clear )
{
    utils::CircularBuffer<int> cb( 5 );
    cb.push_back_overwrite( 1 );
    cb.push_back_overwrite( 2 );
    cb.push_back_overwrite( 3 );
    EXPECT_EQ( cb.size(), 3 );
    EXPECT_FALSE( cb.empty() );

    cb.clear();
    EXPECT_EQ( cb.size(), 0 );
    EXPECT_TRUE( cb.empty() );
    EXPECT_EQ( cb.available(), 5 );
}

TEST( circular_buffer, reserve )
{
    utils::CircularBuffer<int> cb( 3 );
    cb.push_back_overwrite( 1 );
    cb.push_back_overwrite( 2 );
    EXPECT_EQ( cb.size(), 2 );
    EXPECT_EQ( cb.capacity(), 3 );

    cb.reserve( 10 );
    EXPECT_EQ( cb.capacity(), 10 );
    // Reserve clears the buffer
    EXPECT_EQ( cb.size(), 0 );

    // Reserve with smaller capacity is a no-op
    cb.push_back_overwrite( 5 );
    cb.reserve( 5 );
    EXPECT_EQ( cb.capacity(), 10 );
    EXPECT_EQ( cb.size(), 1 );
}

TEST( circular_buffer, shrink_to_fit )
{
    utils::CircularBuffer<int> cb( 10 );
    cb.push_back_overwrite( 1 );
    cb.push_back_overwrite( 2 );
    cb.push_back_overwrite( 3 );
    EXPECT_EQ( cb.size(), 3 );
    EXPECT_EQ( cb.capacity(), 10 );

    cb.shrink_to_fit();
    EXPECT_EQ( cb.capacity(), 3 );
    EXPECT_EQ( cb.size(), 3 );
    EXPECT_EQ( cb[0], 1 );
    EXPECT_EQ( cb[1], 2 );
    EXPECT_EQ( cb[2], 3 );
}

TEST( circular_buffer, move_semantics )
{
    utils::CircularBuffer<int> cb( 5 );
    int a = 42;
    int b = 99;

    cb.push_back( std::move( a ) );
    cb.push_front( std::move( b ) );

    EXPECT_EQ( cb.size(), 2 );
    EXPECT_EQ( cb[0], 99 );
    EXPECT_EQ( cb[1], 42 );
}

TEST( circular_buffer, dump_to_vector_insufficient_space )
{
    utils::CircularBuffer<int> cb( 5 );
    cb.push_back_overwrite( 1 );
    cb.push_back_overwrite( 2 );
    cb.push_back_overwrite( 3 );

    std::vector<int> small( 2 );
    EXPECT_FALSE( cb.dump_to_vector( small ) );

    std::vector<int> exact( 3 );
    EXPECT_TRUE( cb.dump_to_vector( exact ) );
    ASSERT_THAT( exact, testing::ElementsAre( 1, 2, 3 ) );
}

TEST( circular_buffer, resize_preserve_elements )
{
    utils::CircularBuffer<int> cb( 5 );
    for ( int i = 1; i <= 5; ++i )
    {
        cb.push_back_overwrite( i );
    }

    EXPECT_TRUE( cb.resize( 8 ) );
    EXPECT_EQ( cb.size(), 5 );
    EXPECT_EQ( cb.capacity(), 8 );

    std::vector<int> result( 8, -1 );
    cb.dump_to_vector( result );
    ASSERT_THAT( result, testing::ElementsAre( 1, 2, 3, 4, 5, -1, -1, -1 ) );
}

TEST( circular_buffer, resize_fail_conditions )
{
    utils::CircularBuffer<int> cb( 5 );
    cb.push_back_overwrite( 1 );
    cb.push_back_overwrite( 2 );
    cb.push_back_overwrite( 3 );

    // Cannot resize smaller than current element count
    EXPECT_FALSE( cb.resize( 2 ) );

    // Cannot resize to 0
    EXPECT_FALSE( cb.resize( 0 ) );

    // Can resize to equal size
    EXPECT_TRUE( cb.resize( 3 ) );
}

TEST( circular_buffer, capacity )
{
    utils::CircularBuffer<int> cb( 10 );
    EXPECT_EQ( cb.capacity(), 10 );

    cb.resize( 20 );
    EXPECT_EQ( cb.capacity(), 20 );

    for ( int i = 0; i < 5; ++i )
    {
        cb.push_back_overwrite( i );
    }
    cb.shrink_to_fit();
    EXPECT_EQ( cb.capacity(), 5 );
}

TEST( circular_buffer, wrapped_state )
{
    utils::CircularBuffer<int> cb( 5 );
    // Fill the buffer
    for ( int i = 1; i <= 5; ++i )
    {
        cb.push_back_overwrite( i );
    }

    // Overwrite to create wrapped state
    cb.push_back_overwrite( 6 ); // Overwrites 1
    cb.push_back_overwrite( 7 ); // Overwrites 2

    EXPECT_EQ( cb.size(), 5 );
    EXPECT_EQ( cb[0], 3 );
    EXPECT_EQ( cb[1], 4 );
    EXPECT_EQ( cb[2], 5 );
    EXPECT_EQ( cb[3], 6 );
    EXPECT_EQ( cb[4], 7 );

    std::vector<int> result( 5 );
    cb.dump_to_vector( result );
    ASSERT_THAT( result, testing::ElementsAre( 3, 4, 5, 6, 7 ) );
}

TEST( circular_buffer, zero_capacity )
{
    utils::CircularBuffer<int> cb( 0 );
    EXPECT_EQ( cb.size(), 0 );
    EXPECT_EQ( cb.capacity(), 0 );
    EXPECT_TRUE( cb.empty() );

    // Auto-resize from zero
    cb.push_back( 1 );
    EXPECT_EQ( cb.size(), 1 );
    EXPECT_GE( cb.capacity(), 1 );
}

TEST( circular_buffer, first_and_last )
{
    utils::CircularBuffer<int> cb( 5 );
    cb.push_back_overwrite( 10 );
    EXPECT_EQ( cb.first(), 10 );
    EXPECT_EQ( cb.last(), 10 );

    cb.push_back_overwrite( 20 );
    EXPECT_EQ( cb.first(), 10 );
    EXPECT_EQ( cb.last(), 20 );

    cb.push_front_overwrite( 5 );
    EXPECT_EQ( cb.first(), 5 );
    EXPECT_EQ( cb.last(), 20 );
}
