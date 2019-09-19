#ifndef HAMILTONIAN
#define HAMILTONIAN

#include <map>
#include <list>
#include <string>
#include <utility>
#include <complex>
#include <algorithm>
//#include <execution>
#include <functional>
#include <iostream>

#include <model.hpp>
#include <matrix.hpp>
#include <constants.hpp>
#include <operators.hpp>

namespace hamiltonian{

    class hcore : 
        public materials::model< std::shared_ptr<matrix::herm> >{
            public:
                double accuracy = 1.e-6;
                std::shared_ptr<matrix::herm> P = nullptr;
                std::shared_ptr<matrix::herm> Kz = nullptr;
                std::shared_ptr<matrix::herm> PKz = nullptr;
                std::shared_ptr<matrix::herm> AG3Kz = nullptr;
                std::shared_ptr<matrix::herm> CKKz = nullptr;
                std::shared_ptr<matrix::herm> TFPO = nullptr;
                std::shared_ptr<matrix::herm> KzTKz = nullptr;
                std::shared_ptr<matrix::herm> KzG2Kz = nullptr;
                std::shared_ptr<matrix::herm> KzG1Kz = nullptr;
            protected:
                double len = 0;
                std::pair<int, int> blims;
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
                        {"K", K },
                        {"P", P },
                        {"Kz", Kz },
                        {"PKz", PKz },
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
                    std::for_each(flow.begin(), flow.end(),
                        [&](auto& it){
                            auto hmat = operators::pw_matr(
                                it.second, 
                                len, 
                                blims,
                                16384,
                                accuracy);
                            at(it.first) = 
                                std::shared_ptr<matrix::herm>(new matrix::herm(hmat));
                        });
                };
                void fill_add(void){
                    std::size_t bsize = 
                        static_cast<std::size_t>(blims.second - blims.first);
                    std::vector< std::complex<double> > kzs(bsize);
                    std::generate(kzs.begin(), kzs.end(), 
                        [&, n = blims.first](void) mutable {
                            double val = static_cast<double>(n++) * 2. * M_PI / len;
                            return std::complex<double>{val, 0.};
                        });
                    std::vector< std::complex<double> > ones(bsize, {1., 0.});
                    auto kzcmat = matrix::cmat::diagonal(kzs);
                    Kz = std::shared_ptr<matrix::herm>(new matrix::herm(kzcmat));
                    auto _pmat = matrix::cmat::diagonal(ones) * 
                        std::complex<double>{std::sqrt(esk * materials::CdHgTe(0.5).Ep), 0.};
                    P = std::shared_ptr<matrix::herm>(new matrix::herm(_pmat));
                    auto _pkz =  (*P) * (*Kz);
                    PKz = std::shared_ptr<matrix::herm>(new matrix::herm(_pkz));
                    auto _ag3kz = ((*G3) * (*Kz)) + ((*Kz) * (*G3));
                    AG3Kz = std::shared_ptr<matrix::herm>(new matrix::herm(_ag3kz));
                    auto _ckkz = ((*K) * (*Kz)) - ((*Kz) * (*K));
                    CKKz = std::shared_ptr<matrix::herm>(new matrix::herm(_ckkz));
                    auto _tfpo = ((*F) * std::complex<double>{2., 0.})
                        + matrix::cmat::diagonal(ones);
                    TFPO = std::shared_ptr<matrix::herm>(new matrix::herm(_tfpo));
                    auto _kztkz = ((*Kz) * (*TFPO)) * (*Kz);
                    KzTKz = std::shared_ptr<matrix::herm>(new matrix::herm(_kztkz));
                    auto _kzg2kz = ((*Kz) * (*G2)) * (*Kz);
                    KzG2Kz = std::shared_ptr<matrix::herm>(new matrix::herm(_kzg2kz));
                    auto _kzg1kz = ((*Kz) * (*G1)) * (*Kz);
                    KzG1Kz = std::shared_ptr<matrix::herm>(new matrix::herm(_kzg1kz));
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
                    blims = {- basis_size / 2, basis_size / 2 + 1};
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
                matrix::herm get_hblock(
                        const std::pair<std::size_t, std::size_t> ij,
                        const std::pair<double, double> kxky) const {
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
                                es = Es->at(ij);
                    std::vector< std::complex<double> > rv = 
                        {
                            tt,         {0.,0.},        - p * kp / st2,         st2 * pkz / st3,            p * km / (st2 * st3),       {0.,0.},            - pkz / st3,            - p * km / st3,
                            {0., 0},    tt,             {0., 0.},               - p * kp / (st2 * st3),     st2 * pkz / st3,            p * km / st2,       - p * kp / st3,         pkz / st3,
                            {0., 0},    {0., 0},        ut + vt,                -stm,                       rt,                         {0., 0.},           stm / st2,              - st2 * rt,
                            {0., 0},    {0., 0},        {0., 0},                ut - vt,                    ct,                         rt,                 st2 * vt,               - st3 * swm / st2,
                            {0., 0},    {0., 0},        {0., 0},                {0., 0},                    ut - vt,                    std::conj(stp),     - st3 * swp / st2,      - st2 * vt,
                            {0., 0},    {0., 0},        {0., 0},                {0., 0},                    {0., 0},                    ut + vt,            st2 * std::conj(rt),    stp / st2,
                            {0., 0},    {0., 0},        {0., 0},                {0., 0},                    {0., 0},                    {0., 0.},           ut - es,                ct,
                            {0., 0},    {0., 0},        {0., 0},                {0., 0},                    {0., 0},                    {0., 0.},           {0., 0.},               ut - es    
                        };
                    auto am = matrix::cmat(rv);
                    auto cm = matrix::cmat::copy(am);
                    return matrix::herm(cm);
                };
            matrix::herm full_h(const std::pair<double, double> kxky) const{
                std::size_t bsize = 
                        static_cast<std::size_t>(blims.second - blims.first);
                matrix::cmat rv(8 * bsize);
                for(std::size_t r = 0; r < bsize; r++){
                    std::size_t i = r * 8;
                    for(std::size_t c = r; c < bsize; c++){
                        std::size_t j = c * 8;
                        auto ham = get_hblock({r, c}, kxky);
                        rv.put_submatrix(ham, {i, j});
                    }
                }
                return matrix::herm(rv);
            };
    };
};

#endif