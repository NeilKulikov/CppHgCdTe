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

BOOST_AUTO_TEST_CASE(Application)
{
    using namespace vector;
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
    matrix::cmat mat(data);
    const auto rv = mat * vec;
    BOOST_CHECK_CLOSE(rv[0].real(), 1.5359e3 , 1.e-2);
    BOOST_CHECK_CLOSE(rv[0].imag(), 1.0027e3 , 1.e-2);
    BOOST_CHECK_CLOSE(rv[1].real(), -6.000034e8 , 1.e-2);
    BOOST_CHECK_CLOSE(rv[1].imag(), -4.999958e8 , 1.e-2);
    BOOST_CHECK_CLOSE(rv[2].real(), -1.53363e2 , 1.e-2);
    BOOST_CHECK_CLOSE(rv[2].imag(), 2.8965e1 , 1.e-2);
}

BOOST_AUTO_TEST_CASE(NormArr)
{
    std::array< std::complex<double>, 5 > vec = {{
        {1., 2.},
        {3., 4.},
        {-5., 6.},
        {7., -8.},
        {-9., 1.e-4}
    }};
    double rv = vector::norm(vec);
    BOOST_CHECK_CLOSE(rv, 16.88194 , 1.e-4);
}

BOOST_AUTO_TEST_CASE(ApplicationArr)
{
    using namespace vector;
    std::vector< std::complex<double> > data = 
        {
            {0.123, 3.54}, {345., -127.}, {1e-5, 1e-2},
            {0.789, 4.59}, {10.1, 7.891}, {678., 1.e8},
            {5.291, 5.78}, {0.02, 7.101}, {8.87, 12.4}
        };
    std::array< std::complex<double>, 3 > vec = {{
        {1., 2.},
        {3., 4.},
        {-5., 6.}
    }};
    matrix::cmat mat(data);
    const auto rv = mat * vec;
    BOOST_CHECK_CLOSE(rv[0].real(), 1.5359e3 , 1.e-2);
    BOOST_CHECK_CLOSE(rv[0].imag(), 1.0027e3 , 1.e-2);
    BOOST_CHECK_CLOSE(rv[1].real(), -6.000034e8 , 1.e-2);
    BOOST_CHECK_CLOSE(rv[1].imag(), -4.999958e8 , 1.e-2);
    BOOST_CHECK_CLOSE(rv[2].real(), -1.53363e2 , 1.e-2);
    BOOST_CHECK_CLOSE(rv[2].imag(), 2.8965e1 , 1.e-2);
}

BOOST_AUTO_TEST_CASE(ArrScale)
{
    std::array< double, 5 > vec = {{1., 2., 3., 4., -5.}};
    const auto rv = vector::mul(vec, 2.);
    BOOST_CHECK_CLOSE(rv[0], 2. , 1.e-4);
    BOOST_CHECK_CLOSE(rv[3], 8. , 1.e-4);
    BOOST_CHECK_CLOSE(rv[4], -10. , 1.e-4);
    std::array< std::complex<double>, 3 > cvec = {{
        {1., 2.}, {3., 4.}, {-5., -3} 
    }};
    const auto crv = vector::mul(cvec, 2.);
    BOOST_CHECK_CLOSE(crv[0].real(), 2. , 1.e-4);
    BOOST_CHECK_CLOSE(crv[0].imag(), 4. , 1.e-4);
    BOOST_CHECK_CLOSE(crv[2].real(), -10. , 1.e-4);
    BOOST_CHECK_CLOSE(crv[2].imag(), -6. , 1.e-4);
}

BOOST_AUTO_TEST_SUITE_END()