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

BOOST_AUTO_TEST_CASE(matrix_elem)
{
    std::vector<double> xs = 
        { 0.0, 9.999, 10.001, 14.999, 15.001, 20. };
    std::vector<double> ys = 
        { 0.7,   0.7,    0.1,    0.1,    0.7, 0.7 };
    materials::heterostruct hs(xs, ys);
    BOOST_CHECK_CLOSE(hs.composition(12.5), 0.1, 1.e-2);
}

BOOST_AUTO_TEST_CASE(hcore_elem)
{
    std::vector<double> xs = 
        { 0.0, 9.5, 10.1, 14.9, 15.5, 20. };
    std::vector<double> ys = 
        { 0.7,   0.7,    0.1,    0.1,    0.7, 0.7 };
    materials::heterostruct hs(xs, ys);
    hamiltonian::hcore hc(hs);
    BOOST_CHECK_CLOSE(hs.composition(12.5), 0.1, 1.e-2);
}

BOOST_AUTO_TEST_CASE(hcore_compl)
{
    const int len = 101;
    const int bsi = 11;
    std::vector<double> xs(len), ys(len);
    for(int i = 0; i < len; i++){
        xs[i] = 0.1 * i;
        ys[i] = 0.5 + 0.25 * std::sin(xs[i]);
    }
    materials::heterostruct hs(xs, ys);
    BOOST_CHECK_CLOSE(hs.length(), 10., 1.e-2);
    hamiltonian::hcore hc(hs, bsi);
    BOOST_CHECK_CLOSE(hc.Eg->at(0, 0).real(), 0.710105, 1.);
    BOOST_CHECK_CLOSE(hc.Eg->at(0, 1).real(), 0.144693, 1.);
    BOOST_CHECK_CLOSE(hc.Eg->at(0, 1).imag(), -0.026889, 1.);
    BOOST_CHECK_CLOSE(hc.Eg->at(0, 10).imag(), 0.00419287, 5.);
    BOOST_CHECK_CLOSE(hc.Eg->at(0, 10).real(), -0.00225795, 5.);
    BOOST_CHECK_CLOSE(hc.Eg->at(1, 10).imag(), 0.00468632, 5.);
    BOOST_CHECK_CLOSE(hc.Eg->at(1, 10).real(), -0.00280413, 5.);
}

BOOST_AUTO_TEST_SUITE_END()