#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Matrices
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

BOOST_AUTO_TEST_CASE(MatrixFromArray1)
{
    std::vector<double> data = 
        {
            0.123, 3.54, 345., -127., 1e-5, 1e-2,
            0.789, 4.59, 10.1, 7.891, 678., 1.e8,
            5.291, 5.78, 0.02, 7.101, 8.87, 12.4
        };
    auto mat = matrix::cmat(data);
    BOOST_CHECK_CLOSE(mat.at(2, 1).imag(), 7.101, 1.e-4);
    BOOST_CHECK_CLOSE(mat.at(0, 0).real(), 0.123, 1.e-4);
    BOOST_CHECK_CLOSE(mat.at(0, 0).imag(), 3.54, 1.e-4);
    BOOST_CHECK_CLOSE(mat.at(1, 2).real(), 678., 1.e-4);
    BOOST_CHECK_CLOSE(mat.at(2, 2).real(), 8.87, 1.e-4);
}

BOOST_AUTO_TEST_CASE(MatrixFromArray2)
{
    std::vector< std::complex<double> > data = 
        {
            {0.123, 3.54}, {345., -127.}, {1e-5, 1e-2},
            {0.789, 4.59}, {10.1, 7.891}, {678., 1.e8},
            {5.291, 5.78}, {0.02, 7.101}, {8.87, 12.4}
        };
    auto mat = matrix::cmat(data);
    BOOST_CHECK_CLOSE(mat.at(2, 1).imag(), 7.101, 1.e-4);
    BOOST_CHECK_CLOSE(mat.at(0, 0).real(), 0.123, 1.e-4);
    BOOST_CHECK_CLOSE(mat.at(0, 0).imag(), 3.54, 1.e-4);
    BOOST_CHECK_CLOSE(mat.at(1, 2).real(), 678., 1.e-4);
    BOOST_CHECK_CLOSE(mat.at(2, 2).real(), 8.87, 1.e-4);
}

BOOST_AUTO_TEST_CASE(MatrixFromArray2Copy)
{
    const std::vector< std::complex<double> > data = 
        {
            {0.123, 3.54}, {345., -127.}, {1e-5, 1e-2},
            {0.789, 4.59}, {10.1, 7.891}, {678., 1.e8},
            {5.291, 5.78}, {0.02, 7.101}, {8.87, 12.4}
        };
    auto mat = matrix::cmat(data);
    BOOST_CHECK_CLOSE(mat.at(2, 1).imag(), 7.101, 1.e-4);
    BOOST_CHECK_CLOSE(mat.at(0, 0).real(), 0.123, 1.e-4);
    BOOST_CHECK_CLOSE(mat.at(0, 0).imag(), 3.54, 1.e-4);
    BOOST_CHECK_CLOSE(mat.at(1, 2).real(), 678., 1.e-4);
    BOOST_CHECK_CLOSE(mat.at(2, 2).real(), 8.87, 1.e-4);
}

BOOST_AUTO_TEST_CASE(real_imag_copy)
{
    std::vector<double> vec = 
        {
            1., 2., 3., 4.,
            -5., -6., -7., -8.,
            9., 10., 11., 12.,
            -13., -14., -15., -16.
        };
    matrix::rmat        inp(vec);
    matrix::cmat        rm = matrix::cmat::real_copy(inp),
                        im = matrix::cmat::imag_copy(inp);
    for(std::size_t i = 0; i < 4; i++){
        for(std::size_t j = 0; j < 4; j++){
            BOOST_CHECK_CLOSE(rm.at(i, j).real(), inp.at(i, j), 1.e-4);
            BOOST_CHECK(std::abs(rm.at(i, j).imag()) < 1.e-4);
            BOOST_CHECK_CLOSE(im.at(i, j).imag(), inp.at(i, j), 1.e-4);
            BOOST_CHECK(std::abs(im.at(i, j).real()) < 1.e-4);
        }
    }
}

BOOST_AUTO_TEST_CASE(MatrixFromMatrix)
{
    std::vector< std::complex<double> > data = 
        {
            {0.123, 3.54}, {345., -127.}, {1e-5, 1e-2},
            {0.789, 4.59}, {10.1, 7.891}, {678., 1.e8},
            {5.291, 5.78}, {0.02, 7.101}, {8.87, 12.4}
        };
    auto mat1 = matrix::cmat(data);
    auto mat2 = matrix::cmat(mat1);
    BOOST_CHECK_CLOSE(mat2.at(2, 1).imag(), 7.101, 1.e-4);
    BOOST_CHECK_CLOSE(mat2.at(0, 0).real(), 0.123, 1.e-4);
    BOOST_CHECK_CLOSE(mat2.at(0, 0).imag(), 3.54, 1.e-4);
    BOOST_CHECK_CLOSE(mat2.at(1, 2).real(), 678., 1.e-4);
    BOOST_CHECK_CLOSE(mat2.at(2, 2).real(), 8.87, 1.e-4);
}

BOOST_AUTO_TEST_CASE(HermFromMatrix)
{
    std::vector< std::complex<double> > data = 
        {
            {0.123, 3.54}, {345., -127.}, {1e-5, 1e-2},
            {0.789, 4.59}, {10.1, 7.891}, {678., 1.e8},
            {5.291, 5.78}, {0.02, 7.101}, {8.87, 12.4}
        };
    auto mat = matrix::cmat(data);
    auto hmat = matrix::herm(mat);
    BOOST_CHECK_CLOSE(hmat.at(2, 1).imag(), -1.e8, 1.e-4);
    BOOST_CHECK_CLOSE(hmat.at(1, 2).imag(), 1.e8, 1.e-4);
    BOOST_CHECK_CLOSE(hmat.at(2, 1).real(), 678., 1.e-4);
    BOOST_CHECK_CLOSE(hmat.at(1, 2).real(), 678., 1.e-4);
    BOOST_CHECK_CLOSE(hmat.at(2, 2).real(), 8.87, 1.e-4);
    BOOST_CHECK_CLOSE(hmat.at(2, 2).imag(), 0., 1.e-4);
}

BOOST_AUTO_TEST_CASE(GeMM)
{
    std::vector< std::complex<double> > a = 
        {
            {1., 2.}, {3., 4.},
            {5., 6.}, {7., 8.}
        };
    std::vector< std::complex<double> > b = 
        {
            {11., 12.}, {13., 14.},
            {15., 16.}, {17., 18.}
        };
    auto    am = matrix::cmat(a),
            bm = matrix::cmat(b);
    auto    cm = am * bm;
    BOOST_CHECK_CLOSE(cm.at(0, 0).real(), -32., 1.e-2);
    BOOST_CHECK_CLOSE(cm.at(0, 0).imag(), 142., 1.e-2);
    BOOST_CHECK_CLOSE(cm.at(1, 0).real(), -40., 1.e-2);
    BOOST_CHECK_CLOSE(cm.at(1, 0).imag(), 358., 1.e-2);
    BOOST_CHECK_CLOSE(cm.at(1, 1).real(), -44., 1.e-2);
    BOOST_CHECK_CLOSE(cm.at(1, 1).imag(), 410., 1.e-2);
}

BOOST_AUTO_TEST_CASE(Sum)
{
    std::vector< std::complex<double> > a = 
        {
            {1., 2.}, {3., 4.},
            {5., 6.}, {7., 8.}
        };
    std::vector< std::complex<double> > b = 
        {
            {11., 12.}, {14., -14.},
            {15., 16.}, {17., 18.}
        };
    auto    am = matrix::cmat(a),
            bm = matrix::cmat(b);
    auto    cm = am + bm;
    BOOST_CHECK_CLOSE(cm.at(0, 0).real(), 12., 1.e-2);
    BOOST_CHECK_CLOSE(cm.at(0, 0).imag(), 14., 1.e-2);
    BOOST_CHECK_CLOSE(cm.at(0, 1).real(), 17., 1.e-2);
    BOOST_CHECK_CLOSE(cm.at(0, 1).imag(), -10., 1.e-2);
    auto    dm = bm - am;
    BOOST_CHECK_CLOSE(dm.at(0, 0).real(), 10., 1.e-2);
    BOOST_CHECK_CLOSE(dm.at(0, 0).imag(), 10., 1.e-2);
    BOOST_CHECK_CLOSE(dm.at(0, 1).real(), 11., 1.e-2);
    BOOST_CHECK_CLOSE(dm.at(0, 1).imag(), -18., 1.e-2);
}

BOOST_AUTO_TEST_CASE(Scale)
{
    std::vector< std::complex<double> > a = 
        {
            {1., 2.}, {3., 4.},
            {5., 6.}, {7., 8.}
        };
    auto am = matrix::cmat(a);
    std::complex<double> b = {3., -6.};
    auto dm = am * b;
    BOOST_CHECK_CLOSE(dm.at(0, 0).real(), 15., 1.e-2);
    BOOST_CHECK_LE(dm.at(0, 0).imag(), 1.e-3);
    BOOST_CHECK_CLOSE(dm.at(0, 1).real(), 33., 1.e-2);
    BOOST_CHECK_CLOSE(dm.at(0, 1).imag(), -6., 1.e-2);
    BOOST_CHECK_CLOSE(dm.at(1, 1).real(), 69., 1.e-2);
    BOOST_CHECK_CLOSE(dm.at(1, 1).imag(), -18., 1.e-2);
}

BOOST_AUTO_TEST_CASE(HDiag)
{
    std::vector< std::complex<double> > a = 
        {
            {1., 0.}, {3., 4.},
            {3., -4.}, {7., 0.}
        };
    auto am = matrix::cmat(a);
    auto ah = matrix::herm(am);
    auto ev = ah.diagonalize();
    std::cout << ev[0] << ' ' << ev[1] << std::endl;
    BOOST_CHECK_CLOSE(ev[0], -1.83095, 1.e-2);
    BOOST_CHECK_CLOSE(ev[1],  9.83095, 1.e-2);
}

BOOST_AUTO_TEST_CASE(HDiag2)
{
    std::vector< std::complex<double> > a = 
        {
            {8., 0.}, {3., 4.}, {9., -6.},
            {3., -4.}, {7., 0.}, {5., 4.},
            {9., 6.}, {5., -4.}, {1., 0.}
        };
    auto am = matrix::cmat(a);
    auto cm = matrix::cmat::copy(am);
    auto ah = matrix::herm(am);
    auto ev = ah.diagonalize_v();
    auto lambda = matrix::rmat::diagonal(ev.first);
    lambda.print();
    auto rv = 
        ev.second * 
        matrix::cmat::real_copy(lambda) *
        ev.second.conjugate();
    rv.print();
    for(std::size_t i = 0; i < rv.size(); i++){
        for(std::size_t j = 0; j < rv.size(); j++)
            BOOST_CHECK(std::abs(rv.at(i, j) - cm.at(i, j)) <  1.e-6);
    }
}

BOOST_AUTO_TEST_CASE(HDiag3)
{
    const std::vector< std::complex<double> > a = 
        {
            {8., 0.}, {3., 4.}, {9., -6.},
            {3., -4.}, {7., 0.}, {5., 4.},
            {9., 6.}, {5., -4.}, {1., 0.}
        };
    auto am = matrix::cmat(a);
    auto ah = matrix::herm(am);
    auto ev = ah.diagonalize();
    BOOST_CHECK_CLOSE(ev[0], -10.0743, 1.e-2);
    BOOST_CHECK_CLOSE(ev[1],  8.64547, 1.e-2);
    BOOST_CHECK_CLOSE(ev[2],  17.4288, 1.e-2);
}

BOOST_AUTO_TEST_CASE(ScaleRaw)
{
    std::vector< std::complex<double> > a = 
        {
            {8., 0.}, {3., 4.}, {9., -6.},
            {3., -4.}, {7., 0.}, {5., 4.},
            {9., 6.}, {5., -4.}, {1., 0.}
        };
    auto am = matrix::cmat(a);
    std::complex<double> mul = {3., 4.};
    auto bm = am.scale(mul);
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            BOOST_CHECK(std::abs(am.at(i, j) * mul - bm.at(i, j)) < 1.e-6);
        }
    }
}

BOOST_AUTO_TEST_CASE(MSub)
{
    std::vector< std::complex<double> > a = 
        {
            {8., -9.}, {3., 4.}, {9., -6.},
            {3., -4.}, {7., 5.}, {5., 4.},
            {9., 6.}, {5., -4.}, {1., 17.}
        };
    std::vector< std::complex<double> > b = 
        {
            {123., 345.}, {789., -327.},
            {890., -1.},  {512., 314.}
        };
    auto    am = matrix::cmat(a),
            bm = matrix::cmat(b);
    BOOST_CHECK_CLOSE(am.at(0, 0).real(), 8., 1.e-2);
    BOOST_CHECK_CLOSE(am.at(2, 2).imag(), 17., 1.e-2);
    BOOST_CHECK_CLOSE(am.at(2, 0).imag(), 6., 1.e-2);
    BOOST_CHECK_CLOSE(am.at(1, 2).imag(), 4., 1.e-2);
    BOOST_CHECK_CLOSE(am.at(1, 1).real(), 7., 1.e-2);
    am.put_submatrix(bm, {0, 0});
    //am.print();
    BOOST_CHECK_CLOSE(am.at(0, 0).real(), 123., 1.e-2);
    BOOST_CHECK_CLOSE(am.at(0, 0).imag(), 345., 1.e-2);
    BOOST_CHECK_CLOSE(am.at(2, 2).imag(), 17., 1.e-2);
    BOOST_CHECK_CLOSE(am.at(2, 0).imag(), 6., 1.e-2);
    BOOST_CHECK_CLOSE(am.at(1, 2).imag(), 4., 1.e-2);
    BOOST_CHECK_CLOSE(am.at(1, 1).real(), 512., 1.e-2);
}

BOOST_AUTO_TEST_CASE(Sub)
{
    std::vector< std::complex<double> > a = 
        {
            {1., 0.}, {3., 9.},
            {3., 12.}, {7., 67.}
        };
    std::vector< std::complex<double> > b = 
        {
            {34., 56.}, {419., 5.},
            {35., 14.}, {217., -187.}
        };
    auto    am = matrix::cmat(a),
            bm = matrix::cmat(b);
    auto    cm = bm - am;
    BOOST_CHECK_CLOSE(cm.at(0, 0).real(), 33., 1.e-2);
    BOOST_CHECK_CLOSE(cm.at(0, 0).imag(), 56., 1.e-2);
    BOOST_CHECK_CLOSE(cm.at(0, 1).imag(), -4., 1.e-2);
    BOOST_CHECK_CLOSE(cm.at(0, 1).real(), 416., 1.e-2);
    BOOST_CHECK_CLOSE(cm.at(1, 1).real(), 210., 1.e-2);
    BOOST_CHECK_CLOSE(cm.at(1, 1).imag(), -254., 1.e-2);
}

BOOST_AUTO_TEST_CASE(Add)
{
    std::vector< std::complex<double> > a = 
        {
            {1., 0.}, {3., 9.},
            {3., 12.}, {7., 67.}
        };
    std::vector< std::complex<double> > b = 
        {
            {34., 56.}, {419., 5.},
            {35., 14.}, {217., -187.}
        };
    auto    am = matrix::cmat(a),
            bm = matrix::cmat(b);
    auto    cm = bm + am;
    BOOST_CHECK_CLOSE(cm.at(0, 0).real(), 35., 1.e-2);
    BOOST_CHECK_CLOSE(cm.at(0, 0).imag(), 56., 1.e-2);
    BOOST_CHECK_CLOSE(cm.at(0, 1).imag(), 14., 1.e-2);
    BOOST_CHECK_CLOSE(cm.at(0, 1).real(), 422., 1.e-2);
    BOOST_CHECK_CLOSE(cm.at(1, 1).real(), 224., 1.e-2);
    BOOST_CHECK_CLOSE(cm.at(1, 1).imag(), -120., 1.e-2);
}

BOOST_AUTO_TEST_CASE(Copy)
{
    std::vector< std::complex<double> > a = 
        {
            {1., 0.}, {3., 9.},
            {3., 12.}, {7., 67.}
        };
    auto    am = matrix::cmat(a);
    BOOST_CHECK_CLOSE(am.at(0, 0).real(), 1., 1.e-2);
    BOOST_CHECK_CLOSE(am.at(0, 1).imag(), 9., 1.e-2);
    BOOST_CHECK_CLOSE(am.at(1, 1).imag(), 67., 1.e-2);
    auto    bm = matrix::cmat::copy(am);
    BOOST_CHECK_CLOSE(bm.at(0, 0).real(), 1., 1.e-2);
    BOOST_CHECK_CLOSE(bm.at(0, 1).imag(), 9., 1.e-2);
    BOOST_CHECK_CLOSE(bm.at(1, 1).imag(), 67., 1.e-2);
    am.at(0, 0) = {5., 7.};
    am.at(1, 1) = {-5., -1.e8};
    BOOST_CHECK_CLOSE(am.at(0, 0).real(), 5., 1.e-2);
    BOOST_CHECK_CLOSE(am.at(1, 1).real(), -5., 1.e-2);
    BOOST_CHECK_CLOSE(am.at(1, 1).imag(), -1.e8, 1.e-2);
    BOOST_CHECK_CLOSE(bm.at(0, 0).real(), 1., 1.e-2);
    BOOST_CHECK_CLOSE(bm.at(0, 1).imag(), 9., 1.e-2);
    BOOST_CHECK_CLOSE(bm.at(1, 1).imag(), 67., 1.e-2);
}

BOOST_AUTO_TEST_CASE(CInvert)
{
    std::vector< std::complex<double> > a = 
        {
            {1., 0.}, {2., 3.}, {4., 5.},
            {-3., 4.}, {7., 8.}, {9., -13.},
            {12., 0.}, {0., 15.}, {17., 18.}
        };
    auto am = matrix::cmat(a);
    auto im = am.inverse();
    BOOST_CHECK_CLOSE(im.at(0, 0).real(), -0.20802248, 1.e-4);
    BOOST_CHECK_CLOSE(im.at(0, 0).imag(), -0.04568681, 1.e-4);
    BOOST_CHECK_CLOSE(im.at(0, 1).real(), -0.01880179, 1.e-4);
    BOOST_CHECK_CLOSE(im.at(0, 1).imag(), -0.0478055, 1.e-4);
    BOOST_CHECK_CLOSE(im.at(0, 2).real(), 0.08003293, 1.e-4);
    BOOST_CHECK_CLOSE(im.at(0, 2).imag(), -0.00187686, 1.e-3);
    BOOST_CHECK_CLOSE(im.at(1, 0).real(), 0.23913714, 1.e-4);
    BOOST_CHECK_CLOSE(im.at(1, 0).imag(), 0.02527385, 1.e-4);
    BOOST_CHECK_CLOSE(im.at(1, 1).real(), 0.00453895, 1.e-4);
    BOOST_CHECK_CLOSE(im.at(1, 1).imag(), -0.04353086, 1.e-4);
    BOOST_CHECK_CLOSE(im.at(1, 2).real(), -0.03330365, 1.e-4);
    BOOST_CHECK_CLOSE(im.at(1, 2).imag(), -0.01450185, 1.e-4);
    BOOST_CHECK_CLOSE(im.at(2, 0).real(), -0.00948982, 1.e-4);
    BOOST_CHECK_CLOSE(im.at(2, 0).imag(), -0.16870581, 1.e-4);
    BOOST_CHECK_CLOSE(im.at(2, 1).real(), 0.00299457, 1.e-3);
    BOOST_CHECK_CLOSE(im.at(2, 1).imag(), 0.02656939, 1.e-4);
    BOOST_CHECK_CLOSE(im.at(2, 2).real(), 0.01039592, 1.e-4);
    BOOST_CHECK_CLOSE(im.at(2, 2).imag(), 0.01970298, 1.e-4);
}

BOOST_AUTO_TEST_CASE(CInvertComp)
{
    std::vector< std::complex<double> > a = 
        {
            {1., 123.}, {2., 3.}, {4., 5.},
            {-3., 4.}, {7., 8.}, {9., -13.},
            {12., 0.}, {0., 15.}, {17., 18.}
        };
    auto am = matrix::cmat(a);
    auto im = am.inverse();
    auto cm = im * am;
    for(std::size_t i = 0; i < cm.size(); i++){
        for(std::size_t j = 0; j < cm.size(); j++){
            if(i == j){
                BOOST_CHECK_CLOSE(cm.at(i, i).real(), 1., 1.e-9);
                BOOST_CHECK_SMALL(cm.at(i, i).imag(), 1.e-9);
            }else{
                BOOST_CHECK_SMALL(std::abs(cm.at(i, j)), 1.e-9);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(RInvertComp)
{
    std::vector<double> a = 
        {
            789., 123., -435, -413,
            892., -900, -45., -1.,
            1.e-5, 123., 327, -2.,
            9085., 14., 12., 1.
        };
    auto am = matrix::rmat(a);
    auto im = am.inverse();
    auto cm = im * am;
    for(std::size_t i = 0; i < cm.size(); i++){
        for(std::size_t j = 0; j < cm.size(); j++){
            if(i == j){
                BOOST_CHECK_CLOSE(cm.at(i, i), 1., 1.e-9);
            }else{
                BOOST_CHECK_SMALL(std::abs(cm.at(i, j)), 1.e-9);
            }
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()