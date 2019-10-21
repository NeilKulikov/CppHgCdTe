#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE StrainHamiltonian
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <complex>
#include <cmath>


#include <matrix.hpp>
#include <strain_model.hpp>
#include <operators.hpp>
#include <strain_hamiltonian.hpp>

BOOST_AUTO_TEST_SUITE(HamiltonianTests)

BOOST_AUTO_TEST_CASE(model_test)
{
    std::vector<double> xs = 
        { 0.0, 9.999, 10.001, 14.999, 15.001, 20. };
    std::vector<double> ys = 
        { 0.7,   0.7,    0.1,    0.1,    0.7, 0.7 };
    strain::materials::heterostruct hs(xs, ys);
    BOOST_CHECK_CLOSE(hs.composition(12.5), 0.1, 1.e-2);
}

BOOST_AUTO_TEST_CASE(hcore_test)
{
    std::vector<double> xs = 
        { 0.0, 9.999, 10.001, 14.999, 15.001, 20. };
    std::vector<double> ys = 
        { 0.7,   0.7,    0.1,    0.1,    0.7, 0.7 };
    strain::materials::strhtr sh(xs, ys);
    strain::hamiltonian::hcore hs(sh);
    BOOST_CHECK(0==0);
}


BOOST_AUTO_TEST_SUITE_END()