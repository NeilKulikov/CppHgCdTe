#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Integrator
#include <boost/test/unit_test.hpp>

#include <cmath>
#include <complex>
#include <iostream>

#include <integrate.hpp>

BOOST_AUTO_TEST_SUITE(IntegratorTests)

BOOST_AUTO_TEST_CASE(Integrate_x3)
{
    std::function<double(double)> func = [](double x){ return x * x * x; };
    auto rv = staff::integrate_qag(func, std::pair<double, double>(-1., 1.25));
    BOOST_CHECK_CLOSE(rv.first, 0.360352, 1e-3);
    BOOST_CHECK_LE(rv.second, 1.e-4);
}

BOOST_AUTO_TEST_CASE(Integrate_sinx)
{
    std::function<double(double)> func = [](double x){ return sin(x); };
    auto rv = staff::integrate_qag(func, std::pair<double, double>(100., 12.));
    BOOST_CHECK_CLOSE(rv.first, 0.018465, 1e-3);
    BOOST_CHECK_LE(rv.second, 1.e-4);
}

BOOST_AUTO_TEST_CASE(Integrate_cosx)
{
    std::function<double(double)> func = [](double x){ return cos(x); };
    auto rv = staff::integrate_qag(func, std::pair<double, double>(-10., -1.));
    BOOST_CHECK_CLOSE(rv.first, -1.38549, 1e-3);
    BOOST_CHECK_LE(rv.second, 1.e-4);
}

BOOST_AUTO_TEST_CASE(Integrate_s_signx)
{
    std::function<double(double)> func = [](double x){ return (x > 0.) ? 1. : -1.; };
    auto rv = staff::integrate_qag(func, std::pair<double, double>(-10., 5.));
    BOOST_CHECK_CLOSE(rv.first, -5., 1.e-3);
    BOOST_CHECK_LE(rv.second, 1.e-4);
}

BOOST_AUTO_TEST_CASE(Integrate_cexp)
{
    using namespace std::complex_literals;
    std::function<std::complex<double>(double)> func = [](double x){ return std::exp(1i * x); };
    auto rv = staff::integrate_qag(func, std::pair<double, double>(-10., 5.));
    BOOST_CHECK_CLOSE(rv.first.real(), -1.50295, 1.e-3);
    BOOST_CHECK_CLOSE(rv.first.imag(), -1.12273, 1.e-3);
    BOOST_CHECK_LE(rv.second, 1.e-3);
}

BOOST_AUTO_TEST_SUITE_END()