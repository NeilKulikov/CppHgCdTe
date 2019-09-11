#include <iostream>

#include "../ext_deps/matrix.hpp"

int main(void){
    std::vector<double> data = 
        {
            0.123, 3.54, 345., -127., 1e-5, 1e-2,
            0.789, 4.59, 10.1, 7.891, 678., 1.e8,
            5.291, 5.78, 0.02, 7.101, 8.87, 12.4
        };
    std::size_t rsiz = data.size() / 2;
    std::cout << "msiz: " << rsiz << std::endl;
    std::size_t msiz = matrix::sqrt(rsiz);
    std::cout << "msiz: " << msiz << std::endl;
    auto mat = matrix::cmat(data);
    std::cout << "size: " << mat.size() << std::endl;
    for(int i = 0; i < mat.size(); i++){
        for(int j = 0; j < mat.size(); j++){
            std::cout << '(' << i << ',' << j << "): ";
            auto cur = mat.at(i, j);
            std::cout << cur.real() << '+' << cur.imag() 
                << "im" << std::endl;
        }
    }
    return 0;
}