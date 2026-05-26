#pragma once

#include <algorithm>
#include <iterator>
#include <random>
#include <type_traits>
#include <utility>

namespace utils
{

template <typename Numeric, typename Generator = std::mt19937>
Numeric random( Numeric from, Numeric to )
{
    thread_local static Generator gen( std::random_device{}() );

    using dist_type =
        typename std::conditional<std::is_integral<Numeric>::value, std::uniform_int_distribution<Numeric>,
                                  std::uniform_real_distribution<Numeric>>::type;

    thread_local static dist_type dist;

    return dist( gen, typename dist_type::param_type{ from, to } );
}

template <typename T>
class singleton
{
private:
    singleton();

public:
    singleton( const singleton& ) = delete;
    singleton& operator=( const singleton& ) = delete;
    singleton( singleton&& ) = delete;
    singleton& operator=( singleton&& ) = delete;

    static T& instance()
    {
        static T inst;
        return inst;
    }
};

template <typename Iter>
std::pair<Iter, Iter> LoadBalancedPartition( Iter begin, Iter end, int tid, int nthreads )
{
    const size_t total_work = std::distance( begin, end );
    const size_t work_per_thread = total_work / nthreads;
    const size_t resid = total_work % nthreads;
    return static_cast<size_t>( tid ) >= resid
               ? std::make_pair( begin + tid * work_per_thread + resid, begin + ( tid + 1 ) * work_per_thread + resid )
               : std::make_pair( begin + tid * ( work_per_thread + 1 ),
                                 begin + ( tid + 1 ) * ( work_per_thread + 1 ) );
}

template <typename T>
std::pair<T, T> LoadBalancedPartitionPos( T total_work, int tid, int nthreads )
{
    const T work_per_thread = total_work / nthreads;
    const T resid = total_work % nthreads;
    return static_cast<size_t>( tid ) >= resid
               ? std::make_pair( tid * work_per_thread + resid, ( tid + 1 ) * work_per_thread + resid )
               : std::make_pair( tid * ( work_per_thread + 1 ), ( tid + 1 ) * ( work_per_thread + 1 ) );
}

template <typename Iter>
std::pair<Iter, Iter> LoadPrefixBalancedPartition( Iter begin, Iter end, int tid, int nthreads )
{
    const int total_work = *end - *begin;
    const int work_per_thread = total_work / nthreads;
    const int resid = total_work % nthreads;

    Iter lb = tid == 0 ? begin
                       : std::lower_bound( begin, end,
                                           ( tid >= resid ? ( tid * work_per_thread + resid )
                                                          : ( tid * ( work_per_thread + 1 ) ) ) +
                                               *begin );
    Iter le = tid == nthreads - 1
                  ? end
                  : std::lower_bound( begin, end,
                                      ( tid >= resid ? ( ( tid + 1 ) * work_per_thread + resid )
                                                     : ( ( tid + 1 ) * ( work_per_thread + 1 ) ) ) +
                                          *begin );

    return std::make_pair( lb, le );
}

template <typename Iter>
std::pair<typename std::iterator_traits<Iter>::difference_type, typename std::iterator_traits<Iter>::difference_type> LoadPrefixBalancedPartitionPos(
    Iter begin,
    Iter end,
    int tid,
    int nthreads )
{
    auto [sIter, eIter] = LoadPrefixBalancedPartition( begin, end, tid, nthreads );
    return std::make_pair( std::distance( begin, sIter ), std::distance( begin, eIter ) );
}

class knuth_s
{
public:
    knuth_s() : eng( rd() ) {}

    template <typename T, typename Iter>
    void operator()( T M, T start, T end, Iter dest ) const
    {
        double select = std::min( M, end - start ), remaining = end - start;
        for ( T i = start; i < end; ++i )
        {
            if ( dist( eng ) < select / remaining )
            {
                *dest++ = i;
                --select;
            }
            --remaining;
        }
    }

private:
    std::random_device rd;
    mutable std::mt19937 eng;
    mutable std::uniform_real_distribution<double> dist;
};

} // namespace utils
