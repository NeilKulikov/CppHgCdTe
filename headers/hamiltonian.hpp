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
        public materials::model< std::shared_ptr<matrix::cmat> >{
            public:
                double accuracy = 1.e-6;
                std::size_t integ_space = 16384;
                std::shared_ptr<matrix::cmat> P = nullptr;
                std::shared_ptr<matrix::cmat> Kz = nullptr;
                std::shared_ptr<matrix::cmat> PKz = nullptr;
                std::shared_ptr<matrix::cmat> KzP = nullptr;
                std::shared_ptr<matrix::cmat> AG3Kz = nullptr;
                std::shared_ptr<matrix::cmat> CKKz = nullptr;
                std::shared_ptr<matrix::cmat> TFPO = nullptr;
                std::shared_ptr<matrix::cmat> KzTKz = nullptr;
                std::shared_ptr<matrix::cmat> KzG2Kz = nullptr;
                std::shared_ptr<matrix::cmat> KzG1Kz = nullptr;
            protected:
                double len = 0;
                std::pair<int, int> blims = {0, 1};
                std::map< std::string, std::shared_ptr<matrix::cmat>& > mapping =
                    {
                        {"Eg", Eg },
                        {"Es", Es },
                        {"Ep", Ep },
                        {"VBO", VBO },
                        {"G1", G1 },
                        {"G2", G2 },
                        {"G3", G3 },
                        {"F", F },
                        {"K", K },
                        {"P", P },
                        {"Kz", Kz },
                        {"PKz", PKz },
                        {"KzP", KzP },
                        {"AG3Kz", AG3Kz },
                        {"CKKz", CKKz },
                        {"TFPO", TFPO },
                        {"KzTKz", TFPO },
                        {"KzG2Kz", KzG2Kz },
                        {"KzG2Kz", KzG1Kz }
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
                                std::shared_ptr<matrix::cmat>(new matrix::herm(hmat));
                        });
                };
                void fill_add(void){
                    std::size_t bsize = 
                        static_cast<std::size_t>(blims.second - blims.first) + 1;
                    std::vector< std::complex<double> > ones(bsize, {1., 0.});
                    {
                    std::vector< std::complex<double> > kzs(bsize);
                    std::generate(kzs.begin(), kzs.end(), 
                        [&, n = blims.first](void) mutable {
                            const double val = static_cast<double>(n++) * 2. * M_PI / len;
                            return std::complex<double>{val, 0.};
                        });
                    auto kzcmat = matrix::cmat::diagonal(kzs);
                    Kz = std::shared_ptr<matrix::cmat>(new matrix::herm(kzcmat));
                    }
                    {
                    auto _pmat = matrix::cmat::diagonal(ones) * 
                        std::complex<double>{std::sqrt(esk * materials::CdHgTe(0.5).Ep), 0.};
                    P = std::shared_ptr<matrix::herm>(new matrix::herm(_pmat));
                    }
                    {
                    auto _pkz =  (*P) * (*Kz);
                    PKz = std::shared_ptr<matrix::cmat>(new matrix::cmat(_pkz));
                    }
                    {
                    auto _kzp =  (*Kz) * (*P);
                    KzP = std::shared_ptr<matrix::cmat>(new matrix::cmat(_kzp));
                    }
                    {
                    auto _ag3kz = ((*G3) * (*Kz)) + ((*Kz) * (*G3));
                    AG3Kz = std::shared_ptr<matrix::cmat>(new matrix::cmat(_ag3kz));
                    }
                    {
                    auto _ckkz = ((*K) * (*Kz)) - ((*Kz) * (*K));
                    CKKz = std::shared_ptr<matrix::cmat>(new matrix::cmat(_ckkz));
                    }
                    {
                    auto _tfpo = ((*F) * std::complex<double>{2., 0.})
                        + matrix::cmat::diagonal(ones);
                    TFPO = std::shared_ptr<matrix::cmat>(new matrix::herm(_tfpo));
                    }
                    {
                    auto _kztkz = ((*Kz) * (*TFPO)) * (*Kz);
                    KzTKz = std::shared_ptr<matrix::cmat>(new matrix::cmat(_kztkz));
                    }
                    {
                    auto _kzg2kz = ((*Kz) * (*G2)) * (*Kz);
                    KzG2Kz = std::shared_ptr<matrix::cmat>(new matrix::cmat(_kzg2kz));
                    }
                    {
                    auto _kzg1kz = ((*Kz) * (*G1)) * (*Kz);
                    KzG1Kz = std::shared_ptr<matrix::cmat>(new matrix::cmat(_kzg1kz));
                    }
                };
            public:
                std::shared_ptr<matrix::cmat>& at(std::string const & in){
                    return mapping.at(in);
                };
                hcore(  materials::heterostruct const & hs, 
                        std::size_t basis_size = 101,
                        const double acc = 1.e-6){
                    len = hs.length();
                    accuracy = acc;
                    blims = {- (basis_size / 2), basis_size / 2};
                    fill_model(hs);
                    fill_add();
                };
                std::complex<double> T_term(
                        std::pair<std::size_t, std::size_t> const & ij,
                        std::pair<double, double> const & kxky) const {
                    const double    kx = kxky.first,
                                    ky = kxky.second;
                    const double    skln = kx * kx + ky * ky;
                    const auto      Ec = Eg->at(ij) + VBO->at(ij);
                    const auto      Kin = esk * (TFPO->at(ij) * skln + KzTKz->at(ij));
                    return Ec + Kin;
                };
                std::complex<double> U_term(
                        std::pair<std::size_t, std::size_t> const & ij,
                        std::pair<double, double> const & kxky) const {
                    const double    kx = kxky.first,
                                    ky = kxky.second;
                    const double    skln = kx * kx + ky * ky;
                    const auto      Ev = VBO->at(ij);
                    const auto      Kin = esk * (G1->at(ij) * skln + KzG1Kz->at(ij));
                    return Ev - Kin;
                };
                std::complex<double> V_term(
                        std::pair<std::size_t, std::size_t> const & ij,
                        std::pair<double, double> const & kxky) const {
                    const double    kx = kxky.first,
                                    ky = kxky.second;
                    const double    skln = kx * kx + ky * ky;
                    return - esk * (G2->at(ij) * skln - 2. * KzG2Kz->at(ij));
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
                    return esk * 2. * km * CKKz->at(ij);
                };
                std::complex<double> Stp_term(
                        std::pair<std::size_t, std::size_t> const & ij,
                        std::pair<double, double> const & kxky) const {
                    const double    kx = kxky.first,
                                    ky = kxky.second;
                    const std::complex<double> kp = {kx, ky};
                    return - esk * st3 * kp * (AG3Kz->at(ij) + CKKz->at(ij));
                };
                std::complex<double> Stm_term(
                        std::pair<std::size_t, std::size_t> const & ij,
                        std::pair<double, double> const & kxky) const {
                    const double    kx = kxky.first,
                                    ky = kxky.second;
                    const std::complex<double> km = {kx, -ky};
                    return - esk * st3 * km * (AG3Kz->at(ij) + CKKz->at(ij));
                };
                std::complex<double> Swp_term(
                        std::pair<std::size_t, std::size_t> const & ij,
                        std::pair<double, double> const & kxky) const {
                    const double    kx = kxky.first,
                                    ky = kxky.second;
                    const std::complex<double> kp = {kx, ky};
                    return - esk * st3 * kp * (AG3Kz->at(ij) - ot3 * CKKz->at(ij));
                };
                std::complex<double> Swm_term(
                        std::pair<std::size_t, std::size_t> const & ij,
                        std::pair<double, double> const & kxky) const {
                    const double    kx = kxky.first,
                                    ky = kxky.second;
                    const std::complex<double> km = {kx, -ky};
                    return - esk * st3 * km * (AG3Kz->at(ij) - ot3 * CKKz->at(ij));
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
                    const auto  p = P->at(ij),
                                pkz = PKz->at(ij),
                                kzp = KzP->at(ij),
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