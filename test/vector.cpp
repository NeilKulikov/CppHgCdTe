#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Vector
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <complex>
#include <cmath>

#include <matrix.hpp>
#include <vector.hpp>

BOOST_AUTO_TEST_SUITE(VectorTests)

BOOST_AUTO_TEST_CASE(Norm)
{
    std::vector< std::complex<double> > vec = {
        {1., 2.},
        {3., 4.},
        {-5., 6.},
        {7., -8.},
        {-9., 1.e-4}
    };
    double rv = vector::norm(vec);
    BOOST_CHECK_CLOSE(rv, 16.88194 , 1.e-4);
}

/*BOOST_AUTO_TEST_CASE(Application)
{
    std::vector< std::complex<double> > data = 
        {
            {0.123, 3.54}, {345., -127.}, {1e-5, 1e-2},
            {0.789, 4.59}, {10.1, 7.891}, {678., 1.e8},
            {5.291, 5.78}, {0.02, 7.101}, {8.87, 12.4}
        };
    std::vector< std::complex<double> > vec = {
        {1., 2.},
        {3., 4.},
        {-5., 6.}
    };
    matrix::cmat a(data);
    auto rv = a * vec;
    BOOST_CHECK_CLOSE(rv, 16.88194 , 1.e-4);
}*/

BOOST_AUTO_TEST_SUITE_END()