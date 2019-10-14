#ifndef HAMILTONIAN
#define HAMILTONIAN

#include <map>
#include <list>
#include <string>
#include <utility>
#include <complex>
#include <functional>
#include <iostream>

#ifdef USE_PSTL
    #include <pstl/algorithm>
    #include <pstl/execution>
#else
    #include <algorithm>
    //#include <execution>
#endif


#include <model.hpp>
#include <matrix.hpp>
#include <constants.hpp>
#include <operators.hpp>

namespace hamiltonian{

    class hcore : 
        public materials::model< std::shared_ptr<matrix::herm> >{
            public:
                double accuracy = 1.e-6;
                std::size_t integ_space = 16384;
            protected:
                double len = 0;
                std::pair<int, int> blims = {0, 1};
                std::map< std::string, std::shared_ptr<matrix::herm>& > mapping =
                    {
                        {"Eg", Eg },
                        {"Es", Es },
                        {"Ep", Ep },
                        {"VBO", VBO },
                        {"G1", G1 },
                        {"G2", G2 },
                        {"G3", G3 },
                        {"F", F },
                        {"K", K }
                    };
                void fill_model(materials::heterostruct const & hs){
                    std::map< std::string, std::function<double(double)> > flow = 
                        {
                            {"Eg", [&](double x){ return hs.at(x).Eg; } },
                            {"Es", [&](double x){ return hs.at(x).Es; } },
                            {"Ep", [&](double x){ return hs.at(x).Ep; } },
                            {"VBO", [&](double x){ return hs.at(x).VBO; } },
                            {"G1", [&](double x){ return hs.at(x).G1; } },
                            {"G2", [&](double x){ return hs.at(x).G2; } },
                            {"G3", [&](double x){ return hs.at(x).G3; } },
                            {"F", [&](double x){ return hs.at(x).F; } },
                            {"K", [&](double x){ return hs.at(x).K; } }
                        };
                    std::for_each(
#ifdef USE_PSTL
                        std::execution::par,
#endif
                        flow.begin(), flow.end(),
                        [&](auto& it){
                            auto hmat = operators::pw_matr(
                                it.second, 
                                len, 
                                blims,
                                integ_space,
                                accuracy);
                            at(it.first) = 
                                std::shared_ptr<matrix::herm>(new matrix::herm(hmat));
                        });
                };
            public:
                std::shared_ptr<matrix::herm>& at(std::string const & in){
                    return mapping.at(in);
                };
                hcore(  materials::heterostruct const & hs, 
                        std::size_t basis_size = 101,
                        const double acc = 1.e-6){
                    len = hs.length();
                    accuracy = acc;
                    blims = {- (basis_size / 2), basis_size / 2};
                    fill_model(hs);
                };
                double kz(std::size_t i) const{
                    const int n = blims.first + static_cast<int>(i);
                    return 2. * M_PI * static_cast<double>(n) / len;
                };
                std::complex<double> Kz(
                    std::pair<std::size_t, std::size_t> const & ij) const{
                        const double kzv = 
                            (ij.first == ij.second) 
                                ? kz(ij.first) 
                                : 0.;
                        return {kzv, 0.};
                };
                std::complex<double> P_full(
                    std::pair<std::size_t, std::size_t> const & ij) const{
                        return std::sqrt(esk * Ep->at(ij));
                };
                std::complex<double> PKz(
                    std::pair<std::size_t, std::size_t> const & ij) const{
                        return P_full(ij) * kz(ij.second);
                };
                std::complex<double> KzP(
                    std::pair<std::size_t, std::size_t> const & ij) const{
                        return kz(ij.first) * P_full(ij);
                };
                std::complex<double> AG3Kz(
                    std::pair<std::size_t, std::size_t> const & ij) const{
                        const auto g3 = G3->at(ij);
                        return 
                            g3 * kz(ij.second) + kz(ij.first) * g3;
                };
                std::complex<double> CKKz(
                    std::pair<std::size_t, std::size_t> const & ij) const{
                        const auto k = K->at(ij);
                        return
                            k * kz(ij.second) - kz(ij.first) * k;
                };
                std::complex<double> TFPO(
                    std::pair<std::size_t, std::size_t> const & ij) const{
                        const std::complex<double> delta = 
                            (ij.first == ij.second) ? 
                                std::complex<double>{1., 0.} : 
                                std::complex<double>{0., 0.};
                        return 2. * F->at(ij) + delta;
                };
                std::complex<double> KzTKz(
                    std::pair<std::size_t, std::size_t> const & ij) const{
                        return kz(ij.first) * TFPO(ij) * kz(ij.second);
                };
                std::complex<double> KzG2Kz(
                    std::pair<std::size_t, std::size_t> const & ij) const{
                        return kz(ij.first) * G2->at(ij) * kz(ij.second);
                };
                std::complex<double> KzG1Kz(
                    std::pair<std::size_t, std::size_t> const & ij) const{
                        return kz(ij.first) * G1->at(ij) * kz(ij.second);
                };
                std::complex<double> T_term(
                        std::pair<std::size_t, std::size_t> const & ij,
                        std::pair<double, double> const & kxky) const {
                    const double    kx = kxky.first,
                                    ky = kxky.second;
                    const double    skln = kx * kx + ky * ky;
                    const auto      Ec = Eg->at(ij) + VBO->at(ij);
                    const auto      Kin = esk * (TFPO(ij) * skln + KzTKz(ij));
                    return Ec + Kin;
                };
                std::complex<double> U_term(
                        std::pair<std::size_t, std::size_t> const & ij,
                        std::pair<double, double> const & kxky) const {
                    const double    kx = kxky.first,
                                    ky = kxky.second;
                    const double    skln = kx * kx + ky * ky;
                    const auto      Ev = VBO->at(ij);
                    const auto      Kin = esk * (G1->at(ij) * skln + KzG1Kz(ij));
                    return Ev - Kin;
                };
                std::complex<double> V_term(
                        std::pair<std::size_t, std::size_t> const & ij,
                        std::pair<double, double> const & kxky) const {
                    const double    kx = kxky.first,
                                    ky = kxky.second;
                    const double    skln = kx * kx + ky * ky;
                    return - esk * (G2->at(ij) * skln - 2. * KzG2Kz(ij));
                };
                std::complex<double> R_term(
                        std::pair<std::size_t, std::size_t> const & ij,
                        std::pair<double, double> const & kxky) const {
                    const double    kx = kxky.first,
                                    ky = kxky.second;
                    const std::complex<double>
                                    kp = {kx, ky},
                                    km = {kx, -ky};
                    const auto      mu = 0.5 * (G3->at(ij) - G2->at(ij)),
                                    gt = 0.5 * (G3->at(ij) + G2->at(ij));
                    return - esk * st3 * (mu * kp * kp - gt * km * km);
                };
                std::complex<double> C_term(
                        std::pair<std::size_t, std::size_t> const & ij,
                        std::pair<double, double> const & kxky) const {
                    const double    kx = kxky.first,
                                    ky = kxky.second;
                    const std::complex<double> km = {kx, -ky};
                    return esk * 2. * km * CKKz(ij);
                };
                std::complex<double> Stp_term(
                        std::pair<std::size_t, std::size_t> const & ij,
                        std::pair<double, double> const & kxky) const {
                    const double    kx = kxky.first,
                                    ky = kxky.second;
                    const std::complex<double> kp = {kx, ky};
                    return - esk * st3 * kp * (AG3Kz(ij) + CKKz(ij));
                };
                std::complex<double> Stm_term(
                        std::pair<std::size_t, std::size_t> const & ij,
                        std::pair<double, double> const & kxky) const {
                    const double    kx = kxky.first,
                                    ky = kxky.second;
                    const std::complex<double> km = {kx, -ky};
                    return - esk * st3 * km * (AG3Kz(ij) + CKKz(ij));
                };
                std::complex<double> Swp_term(
                        std::pair<std::size_t, std::size_t> const & ij,
                        std::pair<double, double> const & kxky) const {
                    const double    kx = kxky.first,
                                    ky = kxky.second;
                    const std::complex<double> kp = {kx, ky};
                    return - esk * st3 * kp * (AG3Kz(ij) - ot3 * CKKz(ij));
                };
                std::complex<double> Swm_term(
                        std::pair<std::size_t, std::size_t> const & ij,
                        std::pair<double, double> const & kxky) const {
                    const double    kx = kxky.first,
                                    ky = kxky.second;
                    const std::complex<double> km = {kx, -ky};
                    return - esk * st3 * km * (AG3Kz(ij) - ot3 * CKKz(ij));
                };
                matrix::cmat get_hblock(
                        const std::pair<std::size_t, std::size_t> ij,
                        const std::pair<double, double> kxky) const {
                    const auto ji = std::make_pair(ij.second, ij.first);
                    const auto  kx = kxky.first,
                                ky = kxky.second;
                    const std::complex<double>
                                kp = {kx, ky},
                                km = {kx, -ky};
                    const auto  tt = T_term(ij, kxky),
                                ut = U_term(ij, kxky),
                                vt = V_term(ij, kxky),
                                rt = R_term(ij, kxky),
                                ct = C_term(ij, kxky);
                    const auto  stm = Stm_term(ij, kxky),
                                stp = Stp_term(ij, kxky);  
                    const auto  swm = Swm_term(ij, kxky),
                                swp = Swp_term(ij, kxky);         
                    const auto  p = P_full(ij),
                                pkz = PKz(ij),
                                kzp = KzP(ij),
                                es = Es->at(ij);
                    const auto  rth = std::conj(R_term(ji, kxky)),
                                cth = std::conj(C_term(ji, kxky));
                    const auto  stmh = std::conj(Stm_term(ji, kxky)),
                                stph = std::conj(Stp_term(ji, kxky));  
                    const auto  swmh = std::conj(Swm_term(ji, kxky)),
                                swph = std::conj(Swp_term(ji, kxky));
                    std::vector< std::complex<double> > rv = 
                        {
                            tt,                     {0.,0.},                - p * kp / st2,         st2 * pkz / st3,            p * km / (st2 * st3),       {0.,0.},            - pkz / st3,            - p * km / st3,
                            {0., 0},                tt,                     {0., 0.},               - p * kp / (st2 * st3),     st2 * pkz / st3,            p * km / st2,       - p * kp / st3,         pkz / st3,
                            - km * p / st2,         {0., 0.},               ut + vt,                -stm,                       rt,                         {0., 0.},           stm / st2,              - st2 * rt,
                            st2 * kzp / st3,        - km * p / (st2 * st3), - stmh,                 ut - vt,                    ct,                         rt,                 st2 * vt,               - st3 * swm / st2,
                            kp * p / (st2 * st3),   st2 * kzp / st3,        rth,                    cth,                        ut - vt,                    stph,               - st3 * swp / st2,      - st2 * vt,
                            {0., 0},                kp * p / st2,           {0., 0},                rth,                        stp,                        ut + vt,            st2 * rth,              stp / st2,
                            - kzp / st3,            - km * p / st3,         stmh / st2,             st2 * vt,                   - st3 * swph / st2,         st2 * rt,           ut - es,                ct,
                            - kp * p / st3,         kzp / st3,              - st2 * rth,            - st3 * swmh / st2,         - st2 * vt,                 stph / st2,         cth,                    ut - es    
                        };
                    auto am = matrix::cmat(rv);
                    return matrix::cmat::copy(am);
                };
            matrix::herm full_h(const std::pair<double, double> kxky) const{
                std::size_t bsize = 
                        static_cast<std::size_t>(blims.second - blims.first + 1);
                matrix::cmat rv(8 * bsize);
                for(std::size_t r = 0; r < bsize; r++){
                    std::size_t i = r * 8;
                    for(std::size_t c = r; c < bsize; c++){
                        std::size_t j = c * 8;
                        auto ham = get_hblock({r, c}, kxky);
                        rv.put_submatrix(ham, {i, j});
                    }
                }
                auto ret_val = matrix::herm(rv);
                //ret_val.print();
                return ret_val;
            };
    };
};

#endif