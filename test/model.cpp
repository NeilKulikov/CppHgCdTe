#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Model
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <cmath>

#include <model.hpp>

BOOST_AUTO_TEST_SUITE(ModelTests)

BOOST_AUTO_TEST_CASE(OnePoint)
{
    auto md = materials::CdHgTe(0.612);
    BOOST_CHECK_CLOSE(md.Eg, 0.83396, 0.1);
    BOOST_CHECK_CLOSE(md.Es, 0.97595, 0.1);
    BOOST_CHECK_CLOSE(md.Ep, 18.8000, 0.1);
    BOOST_CHECK_CLOSE(md.G1, 2.49044, 0.1);
    BOOST_CHECK_CLOSE(md.G2, 0.02264, 0.1);
    BOOST_CHECK_CLOSE(md.G3, 0.52276, 0.1);
    BOOST_CHECK_CLOSE(md.F, -0.05508, 0.1);
    BOOST_CHECK_CLOSE(md.K, -0.95692, 0.1);
}

BOOST_AUTO_TEST_CASE(HeteroStruct)
{
    std::vector<double> xs(256), ys(256);
    for(std::size_t i = 0; i < xs.size(); i++){
        xs[i] = 0.1 * static_cast<double>(i) + 10.;
        ys[i] = abs(sin(xs[i]));
    }
    auto hs = materials::heterostruct(xs, ys);
    BOOST_CHECK_CLOSE(hs.length(), 255. * 0.1, 1.);
    BOOST_CHECK_CLOSE(hs.composition(1.), 1., 1);
    auto md = hs.at(6.366);
    BOOST_CHECK_CLOSE(md.Eg, 0.83396, 2.);
    BOOST_CHECK_CLOSE(md.Es, 0.97595, 1.);
    BOOST_CHECK_CLOSE(md.Ep, 18.8000, 1.);
    BOOST_CHECK_CLOSE(md.G1, 2.49044, 1.);
    BOOST_CHECK_CLOSE(md.G2, 0.02264, 5.);
    BOOST_CHECK_CLOSE(md.G3, 0.52276, 1.);
    BOOST_CHECK_CLOSE(md.F, -0.05508, 1.);
    BOOST_CHECK_CLOSE(md.K, -0.95692, 1.);
}

BOOST_AUTO_TEST_SUITE_END()