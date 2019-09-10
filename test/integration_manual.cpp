#include <iostream>
#include <cmath>

#include "../ext_deps/integrate.hpp"

int main(void){
    std::function<double(double)> func = [](double x){ return (x > 0.) ? 1.: -1.; };
    //auto ptr = func.target<double(*)(double)>();
    //std::cout << (**ptr)(123.);
    auto rv = staff::integrate_qag(func, std::pair<double, double>(-5., 11.));
    std::cout << rv.first << ' ' << rv.second << std::endl;
    return 0;
}