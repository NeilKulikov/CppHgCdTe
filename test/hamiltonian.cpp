#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Hamiltonian
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <complex>
#include <cmath>
#include <hamiltonian.hpp>
#include <operators.hpp>
#include <matrix.hpp>
#include <model.hpp>

BOOST_AUTO_TEST_SUITE(HamiltonianTests)

BOOST_AUTO_TEST_CASE(model_test)
{
    std::vector<double> xs = 
        { 0.0, 9.999, 10.001, 14.999, 15.001, 20. };
    std::vector<double> ys = 
        { 0.7,   0.7,    0.1,    0.1,    0.7, 0.7 };
    materials::heterostruct hs(xs, ys);
    BOOST_CHECK_CLOSE(hs.composition(12.5), 0.1, 1.e-2);
}

BOOST_AUTO_TEST_CASE(spectr_test)
{
    std::vector<double> xs = 
        { 0.0, 10., 10.0001, 14.9999, 15.0, 25.0 };
    std::vector<double> ys = 
        { 1.0, 1.0,  0.00000, 0.00000, 1.0,  1.0 };
    materials::heterostruct hs(xs, ys);
    hamiltonian::hcore hc(hs, 61);
    auto    spec0 = hc.full_h({0., 0.}).diagonalize(),
            spec1 = hc.full_h({1., 0.}).diagonalize();
    BOOST_CHECK_CLOSE(spec0[368] - spec0[364], 0.449509, 1.e-1);
    BOOST_CHECK_CLOSE(spec0[366] - spec0[364], 0.055627, 1.e-1);
    BOOST_CHECK_CLOSE(spec0[362] - spec0[364], -0.09348, 1.e-1);
    BOOST_CHECK_CLOSE(spec0[360] - spec0[364], -0.13150, 1.e-1);
    BOOST_CHECK_CLOSE(spec1[368] - spec0[364], 0.717170, 1.e-1);
    BOOST_CHECK_CLOSE(spec1[366] - spec0[364], 0.583921, 1.e-1);
    BOOST_CHECK_CLOSE(spec1[364] - spec0[364], -0.09533, 1.e-1);
    BOOST_CHECK_CLOSE(spec1[362] - spec0[364], -0.15505, 1.e-1);
    BOOST_CHECK_CLOSE(spec1[360] - spec0[364], -0.26302, 1.e-1);
}

BOOST_AUTO_TEST_CASE(transl_test)
{
    std::vector<double> xs = 
        { 0.0, 5., 5.0001, 10., 10.001, 25.0 };
    std::vector<double> ys = 
        { 1.0, 1.0,  0.00000, 0.00000, 1.0,  1.0 };
    materials::heterostruct hs(xs, ys);
    hamiltonian::hcore hc(hs, 61);
    auto    spec0 = hc.full_h({0., 0.}).diagonalize(),
            spec1 = hc.full_h({1., 0.}).diagonalize();
    BOOST_CHECK_CLOSE(spec0[368] - spec0[364], 0.449509, 1.e-1);
    BOOST_CHECK_CLOSE(spec0[366] - spec0[364], 0.055627, 1.e-1);
    BOOST_CHECK_CLOSE(spec0[362] - spec0[364], -0.09348, 1.e-1);
    BOOST_CHECK_CLOSE(spec0[360] - spec0[364], -0.13150, 1.e-1);
    BOOST_CHECK_CLOSE(spec1[368] - spec0[364], 0.717170, 1.e-1);
    BOOST_CHECK_CLOSE(spec1[366] - spec0[364], 0.583921, 1.e-1);
    BOOST_CHECK_CLOSE(spec1[364] - spec0[364], -0.09533, 1.e-1);
    BOOST_CHECK_CLOSE(spec1[362] - spec0[364], -0.15505, 1.e-1);
    BOOST_CHECK_CLOSE(spec1[360] - spec0[364], -0.26302, 1.e-1);
}

BOOST_AUTO_TEST_CASE(impure_test)
{
    std::vector<double> xs = 
        { 0.0, 5., 5.0001, 8., 8.001, 25.0 };
    std::vector<double> ys = 
        { 0.7, 0.7,  0.1, 0.1, 0.7,  0.7 };
    materials::heterostruct hs(xs, ys);
    hamiltonian::hcore hc(hs, 61);
    auto    spec0 = hc.full_h({0., 0.}).diagonalize(),
            spec1 = hc.full_h({1., 0.}).diagonalize();
    BOOST_CHECK_CLOSE(spec0[368] - spec0[364], 0.73246024, 1.e-1);
    BOOST_CHECK_CLOSE(spec0[366] - spec0[364], 0.28199164, 1.e-1);
    BOOST_CHECK_CLOSE(spec0[362] - spec0[364], -0.12547854, 1.e-1);
    BOOST_CHECK_CLOSE(spec0[360] - spec0[364], -0.16914711, 1.e-1);
    BOOST_CHECK_CLOSE(spec1[368] - spec0[364], 1.03442234, 1.e-1);
    BOOST_CHECK_CLOSE(spec1[366] - spec0[364], 0.76566906, 1.e-1);
    BOOST_CHECK_CLOSE(spec1[364] - spec0[364], -0.0987744, 1.e-1);
    BOOST_CHECK_CLOSE(spec1[362] - spec0[364], -0.19366342, 1.e-1);
    BOOST_CHECK_CLOSE(spec1[360] - spec0[364], -0.31618791, 1.e-1);
}



BOOST_AUTO_TEST_SUITE_END()