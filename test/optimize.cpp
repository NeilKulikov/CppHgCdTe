#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Optimize
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>

#include <optimize.hpp>

BOOST_AUTO_TEST_SUITE(OptimizeTests)

BOOST_AUTO_TEST_CASE(OptimizeND)
{
    std::function<double(const std::array<double, 2>&)> func = 
        [](const auto& x){ return - x[0] * (1. - x[0]) - 2. * x[1] * (5. - x[1]) + 0.25; };
    //std::cout << "Func declared" << std::endl;
    staff::fdf_minimizer minimizer(func);
    //std::cout << "Min declared" << std::endl;
    const auto res = minimizer.minimize({10., -15.});
    BOOST_CHECK_CLOSE(res.x.at(0), 0.5, 1.e-4);
    BOOST_CHECK_CLOSE(res.x.at(1), 2.5, 1.e-4);
    BOOST_CHECK_CLOSE(res.min, -12.5, 1.e-4);
}

BOOST_AUTO_TEST_SUITE_END()