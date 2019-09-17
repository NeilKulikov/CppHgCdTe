#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Operators
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <complex>
#include <cmath>

#include <operators.hpp>

BOOST_AUTO_TEST_SUITE(OperatorsTests)

BOOST_AUTO_TEST_CASE(matrix_elem)
{
    using namespace std::complex_literals;
    operators::basis<std::complex<double>, double> bas;
    std::function<std::complex<double>(double)> func = 
        [](double x){ return std::sin(2.* M_PI * x); };
    auto rv0 = operators::matrix_elem(bas, func, 1, -1);
    BOOST_CHECK_LE(std::abs(rv0), 1.e-16);
    auto rv1 = operators::matrix_elem(bas, func, 1, 0);
    BOOST_CHECK_CLOSE(rv1.imag(), -0.5, 1.e-2);
    BOOST_CHECK_LE(std::abs(rv1.real()), 1.e-2);
}

BOOST_AUTO_TEST_CASE(oper_matrix)
{
    using namespace std::complex_literals;
    operators::basis<std::complex<double>, double> bas
        (operators::plane_wave, {-2, 3});
    std::function<double(double)> func = 
        [](double x){ return std::sin(19. * M_PI * x); };
    auto mat = operators::herm_op_matr(bas, func);
    //mat.print();
    BOOST_CHECK_CLOSE(mat.at(0, 0).real(), 0.033506, 1.e-2);
    BOOST_CHECK_CLOSE(mat.at(0, 3).real(), 0.037217, 1.e-2);
}

BOOST_AUTO_TEST_CASE(oper_cmp)
{
    using namespace std::complex_literals;
    operators::basis<std::complex<double>, double> bas;
    std::function<double(double)> func = 
        [](double x){ return std::sin(19. * M_PI * x); };
    auto mat_h = operators::herm_op_matr(bas, func);
    auto mat_p = operators::pw_matr(func, 1., {-50, 50});
    //mat_p.print();
    BOOST_CHECK_CLOSE(mat_p.at(10, 1).real(), mat_h.at(10, 1).real(), 1.e-2);
    BOOST_CHECK_CLOSE(mat_p.at(23, 34).real(), mat_h.at(23, 34).real(), 1.e-2);
    BOOST_CHECK_CLOSE(mat_p.at(0, 0).real(), mat_h.at(0, 0).real(), 1.e-2);
    BOOST_CHECK_CLOSE(mat_p.at(0, 0).real(), mat_h.at(0, 0).real(), 1.e-2);
    BOOST_CHECK_CLOSE(mat_p.at(50, 47).real(), mat_h.at(50, 47).real(), 1.e-2);
    BOOST_CHECK_CLOSE(mat_p.at(50, 47).real(), mat_h.at(50, 47).real(), 1.e-2);
}

BOOST_AUTO_TEST_CASE(oper_pw2)
{
    using namespace std::complex_literals;
    operators::basis<std::complex<double>, double> bas;
    std::function<std::complex<double>(double)> func = 
        [](double x){ return (1.i + x) * x; };
    auto mat_p = operators::pw_matr(func, 5., {-1, 2});
    //mat_p.print();
    BOOST_CHECK_CLOSE(mat_p.at(2, 0).real(), 0.7145, 1.);
    BOOST_CHECK_CLOSE(mat_p.at(1, 1).real(), 8.3333, 1.);
}

BOOST_AUTO_TEST_CASE(oper_pw3)
{
    using namespace std::complex_literals;
    std::function<std::complex<double>(double)> func = 
        [](double x){
            return std::complex<double>
                {(x > 2.5) ? 1. : 0., 0.}; 
        };
    auto mat_p = operators::pw_matr(func, 5., {-1, 2}, 4096, 0.00001);
    //mat_p.print();
    BOOST_CHECK_LE(mat_p.at(2, 0).real(), 1.e-3);
    BOOST_CHECK_LE(mat_p.at(2, 0).imag(), 1.e-3);
    BOOST_CHECK_CLOSE(mat_p.at(1, 1).real(), 0.5, 1.);
    BOOST_CHECK_LE(mat_p.at(1, 0).real(), 1.e-3);
    BOOST_CHECK_CLOSE(mat_p.at(1, 0).imag(), 0.31831, 1.e-2);
}

BOOST_AUTO_TEST_SUITE_END()