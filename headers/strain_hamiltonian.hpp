#ifndef STRAIN_HAMILTONIAN
#define STRAIN_HAMILTONIAN

#include <map>
#include <list>
#include <string>
#include <utility>
#include <complex>
#include <functional>
#include <iostream>

#ifdef USE_PSTL
    #include <pstl/algorithm>
    #include <pstl/execution>
#else
    #include <algorithm>
    //#include <execution>
#endif

#include <strain_ model.hpp>
#include <matrix.hpp>
#include <vector.hpp>
#include <constants.hpp>
#include <operators.hpp>

const static std::complex<double> cn = {0., 0.};
const static std::complex<double> co = {1., 0.};
const static std::complex<double> ci = {0., 1.};

namespace strain{
namespace hamiltonian{

    class hcore{
            public:
                double accuracy = 1.e-6;
                std::size_t integ_space = 16384;
            protected:
                double len = 0;
                std::pair<int, int> blims = {0, 1};
                std::map< std::string, std::shared_ptr<matrix::herm>& > mapping =
                    {
                        {"A", A },
                        {"B", B },
                        {"D", D },
                        {"Ac", Ac },
                        {"Av", Av },
                        {"C11", C11 },
                        {"C12", C12 },
                        {"C44", C44 }
                    };
                
    };

};
};

#endif