#include <pstl/algorithm>
#include <pstl/execution>
#include <pstl/iterators.h>

#include <iostream>
#include <vector>
#include <random>
#include <functional>

int main(void){
    const std::size_t len = 1024;
    std::vector<int> data(len);
    std::random_device dev;
    std::mt19937 mgen(dev());
    std::uniform_int_distribution<int> dist(-1024, 1024);
    const auto generator = std::bind(dist, mgen);
    std::generate(data.begin(), data.end(), generator);
    for(auto it = data.begin(); it != std::prev(data.end()); it++)
        if(*it > *std::next(it))
            std::cout << "1: error" << std::endl;
    std::sort(std::execution::par_unseq, data.begin(), data.end());
    for(auto it = data.begin(); it != std::prev(data.end()); it++)
        if(*it > *std::next(it))
            std::cout << "2: error" << std::endl;
    return 0;
};