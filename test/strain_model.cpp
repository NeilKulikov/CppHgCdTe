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
    BOOST_CHECK_CLOSE(str.at(2, 2), -4.3301e-3, 1.e-1);
}

BOOST_AUTO_TEST_CASE(StrainTestImPure1)
{
    strain::materials::strain str(0.0, 0.7);
    BOOST_CHECK_CLOSE(str.at(0, 0), 2.16718e-3, 1.e-1);
    BOOST_CHECK_CLOSE(str.at(0, 1), 0., 1.e-1);
    BOOST_CHECK_CLOSE(str.at(1, 1), 2.16718e-3, 1.e-1);
    BOOST_CHECK_CLOSE(str.at(2, 2), -3.03113e-3, 1.e-1);
}

BOOST_AUTO_TEST_CASE(StrainTestImPure)
{
    strain::materials::strain str(0.1, 0.65);
    BOOST_CHECK_CLOSE(str.at(2, 2), -2.3814e-3, 1.e-1);
}

BOOST_AUTO_TEST_CASE(StrHtr)
{
    const std::size_t num = 100;
    const double len = 20.;
    std::vector<double> zs(num + 1), xs(num + 1);
    const double step = len / 
                static_cast<double>(num);
    for(std::size_t i = 0; i <= num; i++){
        zs[i] = step * static_cast<double>(i);
        xs[i] = 0.5 + 0.45 * std::sin(0.5 * zs[i]);
    }
    strain::materials::heterostruct hs(zs, xs);
    //std::cout << "HETerostruct" << std::endl;
    const double bufx = 0.5;
    //std::cout << "!" << std::endl;
    strain::materials::strhtr sh(hs, bufx);
    //std::cout << "!!" << std::endl;
    const std::size_t anum = 500;
    const double astep = len / 
                static_cast<double>(anum);
    for(std::size_t i = 0; i < anum; i++){
        //std::cout << "!" << std::endl;
        const double z = astep * static_cast<double>(i);
        const auto str_i = sh.get_strain(z).get();
        //std::cout << "!!" << std::endl;
        const auto str_b = strain::materials::strain(hs.at(z), bufx).get();
        //std::cout << i << std::endl;
        for(std::size_t j = 0; j < 6; j++){
            BOOST_CHECK_CLOSE(str_i[j], str_b[j], 1.e-1);
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()