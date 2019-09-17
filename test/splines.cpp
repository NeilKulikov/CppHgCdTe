#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Splines
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>

#include <spline.hpp>

BOOST_AUTO_TEST_SUITE(IntegratorTests)

BOOST_AUTO_TEST_CASE(Integrate_x)
{
    std::vector<double> xs(256), ys(256);
    for(std::size_t i = 0; i < xs.size(); i++){
        xs[i] = 0.25 * static_cast<double>(i) - 10.;
        ys[i] = xs[i];
    }
    auto spl = staff::spline(xs, ys);
    double res;
    {
        res = spl.eval(1.37);
        //std::cout << res << std::endl;
        BOOST_CHECK_CLOSE(res, 1.37, 1e-2);
    }
    {
        res = spl.eval(2.15);
        //std::cout << res << std::endl;
        BOOST_CHECK_CLOSE(res, 2.15, 1e-1);
    }
}

BOOST_AUTO_TEST_CASE(Integrate_x3)
{
    std::vector<double> xs(256), ys(256);
    for(std::size_t i = 0; i < xs.size(); i++){
        xs[i] = 0.25 * static_cast<double>(i) - 10.;
        ys[i] = pow(xs[i], 3);
    }
    auto spl = staff::spline(xs, ys);
    double res;
    {
        res = spl.eval(1.37);
        //std::cout << res << std::endl;
        BOOST_CHECK_CLOSE(res, 2.63, 1.);
    }
    {
        res = spl.eval(20.15);
        //std::cout << res << std::endl;
        BOOST_CHECK_CLOSE(res, 8181.35, 1.);
    }
}

BOOST_AUTO_TEST_CASE(Integrate_sinx)
{
    std::vector<double> xs(256), ys(256);
    for(std::size_t i = 0; i < xs.size(); i++){
        xs[i] = 0.25 * static_cast<double>(i) - 10.;
        ys[i] = sin(xs[i]);
    }
    auto spl = staff::spline(xs, ys);
    double res;
    {
        res = spl.eval(-1.37);
        //std::cout << res << std::endl;
        BOOST_CHECK_CLOSE(res, -0.9799, 1.);
    }
    {
        res = spl.eval(12.37);
        //std::cout << res << std::endl;
        BOOST_CHECK_CLOSE(res, -0.19511, 1.);
    }
}

BOOST_AUTO_TEST_SUITE_END()