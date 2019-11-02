#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Rotations
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <cmath>

#include <matrix.hpp>
#include <vector.hpp>

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
    BOOST_CHECK_CLOSE(rzx[1], 1., 1.e-9);
    BOOST_CHECK(std::abs(rzx[2]) < 1.e-9);
    const auto rzy = rotz * ey;
    BOOST_CHECK_CLOSE(rzy[0], -1., 1.e-9);
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

BOOST_AUTO_TEST_CASE(J12Rot)
{
    const std::array<double, 3> ags = {1., 1., 1.};
    auto j12r = rotations::J12_rot(ags);
    auto vr = rotations::vec_rot(ags);
    auto cvr = matrix::cmat::real_copy(vr);
    std::vector< std::complex<double> > 
        ex = {{1., 0.}, {0., 0.}},
        ey = {{0., 0.}, {1., 0.}};
    
    j12r.print();
    BOOST_CHECK(true);
}


BOOST_AUTO_TEST_SUITE_END()