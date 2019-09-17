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

BOOST_AUTO_TEST_SUITE_END()