#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Rotations
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <cmath>

#include <matrix.hpp>
#include <vector.hpp>

#include <hamiltonian.hpp>
#include <rotations.hpp>

BOOST_AUTO_TEST_SUITE(RotationTests)

BOOST_AUTO_TEST_CASE(IdentityVecRot)
{
    auto rot = rotations::vec_rot(0., 0., 0.);
    for(std::size_t i = 0; i < 3; i++){
        for(std::size_t j = 0; j < 3; j++){
            if(i == j)
                BOOST_CHECK_CLOSE(rot.at(i, j), 1., 1.e-6);
            else
                BOOST_CHECK(std::abs(rot.at(i , j)) < 1.e-9);
        }
    }
}

BOOST_AUTO_TEST_CASE(ZVecRot)
{
    using namespace vector;
    std::vector<double> ex = {1., 0., 0.},
                        ey = {0., 1., 0.},
                        ez = {0., 0., 1.};
    const auto rotz = 
        rotations::vec_rot(0.5 * M_PI, 0., 0.);
    const auto rzx = rotz * ex;
    BOOST_CHECK(std::abs(rzx[0]) < 1.e-9);
    BOOST_CHECK_CLOSE(rzx[1], -1., 1.e-9);
    BOOST_CHECK(std::abs(rzx[2]) < 1.e-9);
    const auto rzy = rotz * ey;
    BOOST_CHECK_CLOSE(rzy[0], 1., 1.e-9);
    BOOST_CHECK(std::abs(rzy[1]) < 1.e-9);
    BOOST_CHECK(std::abs(rzy[2]) < 1.e-9);
    const auto rzz = rotz * ez;
    BOOST_CHECK(std::abs(rzz[0]) < 1.e-9);
    BOOST_CHECK(std::abs(rzz[1]) < 1.e-9);
    BOOST_CHECK_CLOSE(rzz[2], 1., 1.e-9);
}

BOOST_AUTO_TEST_CASE(YVecRot)
{
    using namespace vector;
    std::vector<double> ex = {1., 0., 0.},
                        ey = {0., 1., 0.},
                        ez = {0., 0., 1.};
    const auto roty = 
        rotations::vec_rot(0., 0.5 * M_PI, 0.);
    const auto ryx = roty * ex;
    BOOST_CHECK(std::abs(ryx[0]) < 1.e-9);
    BOOST_CHECK(std::abs(ryx[1]) < 1.e-9);
    BOOST_CHECK_CLOSE(ryx[2], 1., 1.e-9);
    const auto ryy = roty * ey;
    BOOST_CHECK(std::abs(ryy[0]) < 1.e-9);
    BOOST_CHECK_CLOSE(ryy[1], 1., 1.e-9);
    BOOST_CHECK(std::abs(ryy[2]) < 1.e-9);
    const auto ryz = roty * ez;
    BOOST_CHECK_CLOSE(ryz[0], -1., 1.e-9);
    BOOST_CHECK(std::abs(ryz[1]) < 1.e-9);
    BOOST_CHECK(std::abs(ryz[2]) < 1.e-9);  
}

BOOST_AUTO_TEST_CASE(VecRot013)
{
    const auto rot = rotations::rotator({0.5 * M_PI, atan(1. / 3.), - 0.5 * M_PI});
    auto vr = matrix::rmat::copy(rot.v_rot);
    vr.print();
    BOOST_CHECK_CLOSE(vr.at(0, 0), 1., 1.e-9);
    BOOST_CHECK_CLOSE(vr.at(1, 1), 3. / sqrt(10), 1.e-9);
    BOOST_CHECK_CLOSE(vr.at(1, 2), -1. / sqrt(10), 1.e-9);
    BOOST_CHECK_CLOSE(vr.at(2, 2), 3. / sqrt(10), 1.e-9);
    BOOST_CHECK_CLOSE(vr.at(2, 1), 1. / sqrt(10), 1.e-9);
}

BOOST_AUTO_TEST_CASE(J12Rot)
{
    const double    alpha = M_PI * 0.5,
                    beta = atan(1. / 3.),
                    gamma = -M_PI * 0.5;
    const std::array<double, 3> ags = {alpha, beta, gamma};
    auto j12r = rotations::J12_rot(ags);
    j12r.print();
    BOOST_CHECK_CLOSE(j12r.at(0, 0).real(), cos(beta * 0.5), 1.e-9);
    BOOST_CHECK(abs(j12r.at(0, 0).imag()) < 1.e-9);
    BOOST_CHECK(abs(j12r.at(1, 0).real()) < 1.e-9);
    BOOST_CHECK_CLOSE(j12r.at(1, 0).imag(), -sin(beta * 0.5), 1.e-9);
    BOOST_CHECK(abs(j12r.at(0, 1).real()) < 1.e-9);
    BOOST_CHECK_CLOSE(j12r.at(0, 1).imag(), -sin(beta * 0.5), 1.e-9);
    BOOST_CHECK_CLOSE(j12r.at(1, 1).real(), cos(beta * 0.5), 1.e-9);
    BOOST_CHECK(abs(j12r.at(1, 1).imag()) < 1.e-9);
}

BOOST_AUTO_TEST_CASE(J32Rot)
{
    const double    alpha = M_PI * 0.5,
                    beta = atan(1. / 3.),
                    gamma = -M_PI * 0.5;
    const std::array<double, 3> ags = {alpha, beta, gamma};
    auto j32r = rotations::J32_rot(ags);
    j32r.print();
    BOOST_CHECK_CLOSE(j32r.at(0, 0).real(), 0.75 * cos(beta * 0.5) + 0.25 * cos(1.5 * beta), 1.e-9);
    BOOST_CHECK(abs(j32r.at(0, 0).imag()) < 1.e-9);
    BOOST_CHECK_CLOSE(j32r.at(1, 0).imag(), - 0.25 * sqrt(3.) * (sin(beta * 0.5) + sin(1.5 * beta)), 1.e-5);
    BOOST_CHECK(abs(j32r.at(1, 0).real()) < 1.e-9);
    BOOST_CHECK_CLOSE(j32r.at(2, 0).real(), 0.25 * sqrt(3.) * ( - cos(beta * 0.5) + cos(1.5 * beta)), 1.e-5);
    BOOST_CHECK(abs(j32r.at(2, 0).imag()) < 1.e-9);
    BOOST_CHECK_CLOSE(j32r.at(3, 0).imag(), 0.75 * sin(beta * 0.5) - 0.25 * sin(1.5 * beta), 1.e-5);
    BOOST_CHECK(abs(j32r.at(3, 0).real()) < 1.e-9);
    BOOST_CHECK_CLOSE(j32r.at(1, 1).real(), 0.25 * cos(beta * 0.5) + 0.75 * cos(1.5 * beta), 1.e-9);
    BOOST_CHECK(abs(j32r.at(1, 1).imag()) < 1.e-9);
}

BOOST_AUTO_TEST_CASE(Rotator013)
{
    const double    alpha = M_PI * 0.5,
                    beta = atan(1. / 3.),
                    gamma = -M_PI * 0.5;
    const auto rot = rotations::rotator(alpha, beta, gamma);
    auto cr = matrix::cmat(rot.c_rot);
    cr.print();
    BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(real_test)
{
    const double    alpha = M_PI * 0.5,
                    beta = atan(1. / 3.),
                    gamma = -M_PI * 0.5;
    auto rot = rotations::rotator(alpha, beta, gamma);
    std::vector<double> xs = 
        { 0.0, 5.,   5.0001, 20.0 };
    std::vector<double> ys = 
        { 0.1, 0.1,  0.7,    0.7 };
    materials::heterostruct hs(xs, ys);
    hamiltonian::hcore hc(hs, 61);
    auto    spec0 = hc.full_h({0., 0.}, &rot).diagonalize(),
            spec1 = hc.full_h({1., 0.}, &rot).diagonalize();
    BOOST_CHECK_CLOSE(spec0[368] - spec0[364], 0.727207, 1.e-1);
    BOOST_CHECK_CLOSE(spec0[366] - spec0[364], 0.273551, 1.e-1);
    BOOST_CHECK_CLOSE(spec0[362] - spec0[364], -0.131296, 1.e-1);
    BOOST_CHECK_CLOSE(spec0[360] - spec0[364], -0.154354, 1.e-1);
    BOOST_CHECK_CLOSE(spec1[368] - spec0[364], 1.0359056, 1.e-1);
    BOOST_CHECK_CLOSE(spec1[366] - spec0[364], 0.7820483, 1.e-1);
    BOOST_CHECK_CLOSE(spec1[364] - spec0[364], -0.07061443, 1.e-1);
    BOOST_CHECK_CLOSE(spec1[362] - spec0[364], -0.16837806, 1.e-1);
    BOOST_CHECK_CLOSE(spec1[360] - spec0[364], -0.33397717, 1.e-1);
}

BOOST_AUTO_TEST_SUITE_END()