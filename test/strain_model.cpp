#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ModelStrain
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <array>
#include <cmath>

#include <strain_model.hpp>

BOOST_AUTO_TEST_SUITE(ModelTests)

BOOST_AUTO_TEST_CASE(OnePoint)
{
    auto md = strain::materials::CdHgTe(0.612);
    BOOST_CHECK_CLOSE(md.A, 6.472224, 1.e-3);
    BOOST_CHECK_CLOSE(md.B, -1.31640, 1.e-3);
    BOOST_CHECK_CLOSE(md.D, -4.2748, 1.e-3);
}

BOOST_AUTO_TEST_CASE(StrainIndexing)
{
    const std::array<int, 6> 
        inp = {1, 2, 3, 4, 5, 6};
    strain::materials::str_mod<int> 
        md = inp;
    for(std::size_t i = 0; i < inp.size(); i++)
        BOOST_CHECK_EQUAL(md.get()[i], inp[i]);
    BOOST_CHECK_EQUAL(md.at(0, 0), 1);
    BOOST_CHECK_EQUAL(md.at(0, 1), 2);
    BOOST_CHECK_EQUAL(md.at(1, 0), 2);
    BOOST_CHECK_EQUAL(md.at(1, 1), 3);
    BOOST_CHECK_EQUAL(md.at(2, 0), 4);
    BOOST_CHECK_EQUAL(md.at(0, 2), 4);
    BOOST_CHECK_EQUAL(md.at(2, 1), 5);
    BOOST_CHECK_EQUAL(md.at(1, 2), 5);
    BOOST_CHECK_EQUAL(md.at(2, 2), 6);
}

BOOST_AUTO_TEST_CASE(StrainTestZero)
{
    strain::materials::strain str(0.7, 0.7);
    for(std::size_t i = 0; i < 6; i++)
        BOOST_CHECK_CLOSE(str.get()[i], 0., 1.e-4);
}

BOOST_AUTO_TEST_CASE(StrainTestPure)
{
    strain::materials::strain str(0.0, 1.0);
    BOOST_CHECK_CLOSE(str.at(0, 0), 3.0959e-3, 1.e-1);
    BOOST_CHECK_CLOSE(str.at(0, 1), 0., 1.e-1);
    BOOST_CHECK_CLOSE(str.at(1, 1), 3.0959e-3, 1.e-1);
    BOOST_CHECK_CLOSE(str.at(2, 2), -6.5063e-3, 1.e-1);
}

BOOST_AUTO_TEST_SUITE_END()