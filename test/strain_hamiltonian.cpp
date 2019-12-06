#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE StrainHamiltonian
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <complex>
#include <cmath>


#include <matrix.hpp>
#include <hamiltonian.hpp>
#include <strain_model.hpp>
#include <operators.hpp>
#include <strain_hamiltonian.hpp>

BOOST_AUTO_TEST_SUITE(StrainHamiltonianTests)

BOOST_AUTO_TEST_CASE(model_test)
{
    std::vector<double> xs = 
        { 0.0, 9.999, 10.001, 14.999, 15.001, 20. };
    std::vector<double> ys = 
        { 0.7,   0.7,    0.1,    0.1,    0.7, 0.7 };
    strain::materials::heterostruct hs(xs, ys);
    BOOST_CHECK_CLOSE(hs.composition(12.5), 0.1, 1.e-2);
}

BOOST_AUTO_TEST_CASE(impure_test)
{
    std::vector<double> xs = 
        { 0.0, 0.0001, 3., 3.0001, 20. };
    std::vector<double> ys = 
        { 0.65, 0.1, 0.1, 0.65, 0.65 };
    const std::size_t bsize = 41;
    materials::heterostruct hs(xs, ys);
    hamiltonian::hcore hc(hs, bsize);
    strain::materials::strhtr ssh(xs, ys, 0.5);
    strain::hamiltonian::hcore shc(ssh, bsize);
    const auto shci = shc.full_h(); 
    const std::pair<double, double> 
        kxky0 = {0., 0.}, 
        kxky1 = {0., 1.}, 
        kxky2 = {1., 1.};
    auto    ham0 = hc.full_h(kxky0) + shci,
            ham1 = hc.full_h(kxky1) + shci,
            ham2 = hc.full_h(kxky2) + shci;
    const auto  spc0 = matrix::herm(ham0).diagonalize(),
                spc1 = matrix::herm(ham1).diagonalize(),
                spc2 = matrix::herm(ham2).diagonalize();
    BOOST_CHECK_CLOSE(spc0[238] - spc0[244], -0.25261283, 1.e-4);
    BOOST_CHECK_CLOSE(spc0[240] - spc0[244], -0.16203509, 1.e-4);
    BOOST_CHECK_CLOSE(spc0[242] - spc0[244], -0.11661292, 1.e-4);
    BOOST_CHECK_CLOSE(spc0[246] - spc0[244], 0.27380030, 1.e-4);
    BOOST_CHECK_CLOSE(spc0[248] - spc0[244], 0.68049689, 1.e-4);
    BOOST_CHECK_CLOSE(spc1[238] - spc0[244], -0.34245081, 1.e-4);
    BOOST_CHECK_CLOSE(spc1[240] - spc0[244], -0.29975227, 1.e-4);
    BOOST_CHECK_CLOSE(spc1[242] - spc0[244], -0.18554772, 1.e-4);
    BOOST_CHECK_CLOSE(spc1[244] - spc0[244], -0.09322957, 1.e-4);
    BOOST_CHECK_CLOSE(spc1[246] - spc0[244], 0.76392115, 1.e-4);
    BOOST_CHECK_CLOSE(spc1[248] - spc0[244], 1.02336368, 1.e-4);
    BOOST_CHECK_CLOSE(spc2[238] - spc0[244], -0.37552268, 1.e-4);
    BOOST_CHECK_CLOSE(spc2[240] - spc0[244], -0.35075110, 1.e-4);
    BOOST_CHECK_CLOSE(spc2[242] - spc0[244], -0.22101852, 1.e-4);
    BOOST_CHECK_CLOSE(spc2[244] - spc0[244], -0.10900867, 1.e-4);
    BOOST_CHECK_CLOSE(spc2[246] - spc0[244], 0.96447296, 1.e-4);
    BOOST_CHECK_CLOSE(spc2[248] - spc0[244], 1.18896688, 1.e-4);
}


BOOST_AUTO_TEST_SUITE_END()