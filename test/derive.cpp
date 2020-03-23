#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Derive
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>

#include <derivate.hpp>

BOOST_AUTO_TEST_SUITE(DerivatorTests)

BOOST_AUTO_TEST_CASE(Derive_1d)
{
    std::function<double(double)> func = [](double x){ return x * (1. - x); };
    const double d1 = staff::derive::derive(func, 1.);
    BOOST_CHECK_CLOSE(d1, -1., 1.e-4);
    const double d2 = staff::derive::derive(func, 0.5);
    BOOST_CHECK_LE(abs(d2), 1.e-5);
}

BOOST_AUTO_TEST_CASE(Derive_nd)
{
    std::function<double(const std::array<double, 2>&)> func = 
        [](const auto& x){ return x[0] * (1. - x[0]) - x[1] * (1. - x[1]); };
    const auto d1 = staff::derive::derive_nd<2>(func, {2., 1.});
    BOOST_CHECK_CLOSE(d1[0], -3., 1.e-4);
    BOOST_CHECK_CLOSE(d1[1], 1., 1.e-4);
}

BOOST_AUTO_TEST_SUITE_END()