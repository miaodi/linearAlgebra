#include <iostream>
#include <vector>
#include <ranges>

int main(int argc, char** argv) {
    std::vector<std::vector<int>> discounnected_sets{{1,2,3}, {4}, {5,6}, {7,8}, {9,10,11,12, 13}};

    // Using join_view to flatten the nested vectors
    auto joined = discounnected_sets | std::views::join;
    
    // Get the size of joined view
    auto joined_size = std::ranges::distance(joined);
    std::cout << "Joined size: " << joined_size << std::endl;
    
    std::cout << "Joined elements: ";
    for (int val : joined) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    // Partition into 3 pieces evenly using take and drop
    auto third = joined_size / 3;
    auto two_thirds = 2 * third;
    
    auto first_piece = joined | std::views::drop(0) | std::views::take(third);
    auto second_piece = joined | std::views::drop(third) | std::views::take(third);
    auto third_piece = joined | std::views::drop(two_thirds) | std::views::take(joined_size - two_thirds);
    
    std::cout << "First piece: ";
    for (int val : first_piece) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Second piece: ";
    for (int val : second_piece) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Third piece: ";
    for (int val : third_piece) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
