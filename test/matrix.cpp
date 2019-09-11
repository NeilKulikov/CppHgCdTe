#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Integrator
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <complex>
#include <cmath>

#include <matrix.hpp>

BOOST_AUTO_TEST_SUITE(MatrixTests)

BOOST_AUTO_TEST_CASE(Sqrt)
{
    std::size_t a = 169,
                b = 144,
                c = 143;
    BOOST_CHECK_EQUAL(matrix::sqrt(a), 13);
    BOOST_CHECK_EQUAL(matrix::sqrt(b), 12);
    BOOST_CHECK_EQUAL(matrix::sqrt(c), 11);
}

BOOST_AUTO_TEST_CASE(ComplexGsl)
{
    std::complex<double> a{0.123, 0.456};
    auto am = matrix::to_gsl_complex(a);
    BOOST_CHECK_CLOSE(a.real(), GSL_REAL(am), 1.e-4);
    BOOST_CHECK_CLOSE(a.imag(), GSL_IMAG(am), 1.e-4);
    gsl_complex b{0.789, -0.123};
    auto bm = matrix::to_std_complex(b);
    BOOST_CHECK_CLOSE(bm.real(), GSL_REAL(b), 1.e-4);
    BOOST_CHECK_CLOSE(bm.imag(), GSL_IMAG(b), 1.e-4);
    std::complex<double> c{-7.89, 564.};
    auto cm = matrix::to_gsl_complex(c);
    BOOST_CHECK_CLOSE(c.real(), GSL_REAL(cm), 1.e-4);
    BOOST_CHECK_CLOSE(c.imag(), GSL_IMAG(cm), 1.e-4);
    gsl_complex d{1e-5, 1e14};
    auto dm = matrix::to_std_complex(d);
    BOOST_CHECK_CLOSE(dm.real(), GSL_REAL(d), 1.e-4);
    BOOST_CHECK_CLOSE(dm.imag(), GSL_IMAG(d), 1.e-4);
    std::complex<float> e{-7.89, 564.};
    auto em = matrix::to_gsl_complex(c);
    BOOST_CHECK_CLOSE(e.real(), GSL_REAL(em), 1.e-4);
    BOOST_CHECK_CLOSE(e.imag(), GSL_IMAG(em), 1.e-4);
    gsl_complex f{1e-5, 1e14};
    auto fm = matrix::to_std_complex<float>(d);
    BOOST_CHECK_CLOSE(fm.real(), GSL_REAL(f), 1.e-4);
    BOOST_CHECK_CLOSE(fm.imag(), GSL_IMAG(f), 1.e-4);
}

BOOST_AUTO_TEST_CASE(MatrixAccess)
{
    auto mat = matrix::cmat(100);
    std::complex<double> to{0.123, 19.};
    mat.at_gsl(10, 10) = matrix::to_gsl_complex(to);
    auto from_gsl = matrix::to_std_complex(mat.at_gsl(10, 10));
    BOOST_CHECK_CLOSE(to.real(), from_gsl.real(), 1.e-4);
    BOOST_CHECK_CLOSE(to.imag(), from_gsl.imag(), 1.e-4);
    mat.at(50, 34) = to;
    auto from = mat.at(50, 34);
    BOOST_CHECK_CLOSE(to.real(), from.real(), 1.e-4);
    BOOST_CHECK_CLOSE(to.imag(), from.imag(), 1.e-4);
    auto fromg = matrix::to_gsl_complex(from);
    BOOST_CHECK_CLOSE(to.real(), GSL_REAL(fromg), 1.e-4);
    BOOST_CHECK_CLOSE(to.imag(), GSL_IMAG(fromg), 1.e-4);
}

BOOST_AUTO_TEST_CASE(MatrixFromArray)
{
    /*std::vector< std::complex<double> > data = 
        {
            {0.123, 3.54}, {345., -127.}, {1e-5, 1e-2},
            {0.789, 4.59}, {10.1, 7.891}, {678., 1.e8},
            {5.291, 5.78}, {0.02, 7.101}, {8.87, 12.4}
        };*/
    std::vector<double> data = 
        {
            0.123, 3.54, 345., -127., 1e-5, 1e-2,
            0.789, 4.59, 10.1, 7.891, 678., 1.e8,
            5.291, 5.78, 0.02, 7.101, 8.87, 12.4
        };
    auto mat = matrix::cmat(data);
    //BOOST_CHECK_CLOSE(mat.at(2, 1).imag(), 7.101, 1.e-4);
    BOOST_CHECK_CLOSE(mat.at(0, 0).real(), 1.e-5, 1.e-4);
    BOOST_CHECK_CLOSE(mat.at(0, 0).imag(), 1.e-5, 1.e-4);
    //BOOST_CHECK_CLOSE(mat.at(0, 0).imag(), 3.54, 1.e-4);
    //BOOST_CHECK_CLOSE(mat.at(2, 2).real(), 8.87, 1.e-4);
}

BOOST_AUTO_TEST_SUITE_END()