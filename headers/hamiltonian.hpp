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
#include <vector.hpp>
#include <service.hpp>
#include <rotations.hpp>
#include <constants.hpp>
#include <operators.hpp>

namespace hamiltonian{

    const static std::complex<double> cn = {0., 0.};
    const static std::complex<double> co = {1., 0.};
    const static std::complex<double> ci = {0., 1.};

    const static auto id_rot = rotations::rotator();

    class hcore : 
        public materials::model< std::shared_ptr<matrix::herm> >{
            public:
                const double accuracy = 1.e-6;
                const std::size_t integ_space = 16384;
            protected:
                const double len = 0;
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
                        const double acc = 1.e-6) : 
                            accuracy(acc), len(hs.length()), 
                            blims({- (basis_size / 2), basis_size / 2}){
                    fill_model(hs);
                };
                double kz(std::size_t i) const{
                    const int n = blims.first + static_cast<int>(i);
                    return 2. * M_PI * static_cast<double>(n) / len;
                };
                std::complex<double> P_full(
                    std::pair<std::size_t, std::size_t> const & ij) const{
                        return std::sqrt(esk * Ep->at(ij));
                };
                std::complex<double> TFPO(
                    std::pair<std::size_t, std::size_t> const & ij) const{
                        const std::complex<double> delta = 
                            (ij.first == ij.second) ? 
                                std::complex<double>{1., 0.} : 
                                std::complex<double>{0., 0.};
                        return 2. * F->at(ij) + delta;
                };
                std::complex<double> T_term(
                        std::pair<std::size_t, std::size_t> const & ij,
                        std::vector< std::complex<double> >& q1,
                        std::vector< std::complex<double> >& q2,
                        const rotations::rotator& rot = id_rot) const{
                    const auto Ec = Eg->at(ij) + VBO->at(ij);
                    const auto ef = esk * TFPO(ij);
                    std::vector< std::complex<double> > core_vec  = 
                        {
                            Ec,     cn,     cn,     cn,
                            cn,     ef,     cn,     cn,
                            cn,     cn,     ef,     cn,
                            cn,     cn,     cn,     ef
                        };
                    const matrix::cmat core(core_vec);
                    matrix::cmat r_core = rot.transform_term(core);
                    auto cq1 = vector::dot(r_core, q1);
                    return vector::dot(q2, cq1);
                };
                std::complex<double> U_term(
                        std::pair<std::size_t, std::size_t> const & ij,
                        std::vector< std::complex<double> >& q1,
                        std::vector< std::complex<double> >& q2,
                        const rotations::rotator& rot = id_rot) const{
                    const auto Ev = VBO->at(ij);
                    const auto g1 = esk * G1->at(ij);
                    std::vector< std::complex<double> > core_vec  = 
                        {
                            Ev,     cn,     cn,     cn,
                            cn,    -g1,     cn,     cn,
                            cn,     cn,    -g1,     cn,
                            cn,     cn,     cn,    -g1
                        };
                    const matrix::cmat core(core_vec);
                    matrix::cmat r_core = rot.transform_term(core);
                    auto cq1 = vector::dot(r_core, q1);
                    return vector::dot(q2, cq1);
                };
                std::complex<double> V_term(
                        std::pair<std::size_t, std::size_t> const & ij,
                        std::vector< std::complex<double> >& q1,
                        std::vector< std::complex<double> >& q2,
                        const rotations::rotator& rot = id_rot) const{
                    const auto g2 = esk * G2->at(ij);
                    std::vector< std::complex<double> > core_vec  = 
                        {
                            cn,     cn,     cn,     cn,
                            cn,    -g2,     cn,     cn,
                            cn,     cn,    -g2,     cn,
                            cn,     cn,     cn,     2. * g2
                        };
                    const matrix::cmat core(core_vec);
                    matrix::cmat r_core = rot.transform_term(core);
                    auto cq1 = vector::dot(r_core, q1);
                    return vector::dot(q2, cq1);
                };
                std::complex<double> Rp_term(
                        std::pair<std::size_t, std::size_t> const & ij,
                        std::vector< std::complex<double> >& q1,
                        std::vector< std::complex<double> >& q2,
                        const rotations::rotator& rot = id_rot) const{
                    const auto mu = 0.5 * esk * (G3->at(ij) - G2->at(ij));
                    std::vector< std::complex<double> > core_vec  = 
                        {
                            cn,     cn,     cn,     cn,
                            cn,     co,     ci,     cn,
                            cn,     ci,    -co,     cn,
                            cn,     cn,     cn,     cn
                        };
                    const matrix::cmat core(core_vec);
                    matrix::cmat r_core = rot.transform_term(core);
                    auto cq1 = vector::dot(r_core, q1);
                    return mu * vector::dot(q2, cq1);
                };
                std::complex<double> Rm_term(
                        std::pair<std::size_t, std::size_t> const & ij,
                        std::vector< std::complex<double> >& q1,
                        std::vector< std::complex<double> >& q2,
                        const rotations::rotator& rot = id_rot) const{
                    const auto nu = 0.5 * esk * (G3->at(ij) + G2->at(ij));
                    std::vector< std::complex<double> > core_vec  = 
                        {
                            cn,     cn,     cn,     cn,
                            cn,     co,    -ci,     cn,
                            cn,    -ci,    -co,     cn,
                            cn,     cn,     cn,     cn
                        };
                    const matrix::cmat core(core_vec);
                    matrix::cmat r_core = rot.transform_term(core);
                    auto cq1 = vector::dot(r_core, q1);
                    return nu * vector::dot(q2, cq1);
                };
                std::complex<double> R_term(
                        std::pair<std::size_t, std::size_t> const & ij,
                        std::vector< std::complex<double> >& q1,
                        std::vector< std::complex<double> >& q2,
                        const rotations::rotator& rot = id_rot) const{
                    return -st3 * (Rp_term(ij, q1, q2, rot) - Rm_term(ij, q1, q2, rot));
                };
                std::complex<double> Qm_term(
                        std::pair<std::size_t, std::size_t> const & ij,
                        std::vector< std::complex<double> >& q1,
                        std::vector< std::complex<double> >& q2,
                        const rotations::rotator& rot = id_rot) const{
                    const auto k2 = esk * K->at(ij);
                    std::vector< std::complex<double> > core_vec  = 
                        {
                            cn,     cn,     cn,         cn,
                            cn,     cn,     cn,         -k2,
                            cn,     cn,     cn,         ci * k2,
                            cn,     k2,     -ci * k2,   cn
                        };
                    const matrix::cmat core(core_vec);
                    matrix::cmat r_core = rot.transform_term(core);
                    auto cq1 = vector::dot(r_core, q1);
                    return vector::dot(q2, cq1);
                };
                std::complex<double> Qp_term(
                        std::pair<std::size_t, std::size_t> const & ij,
                        std::vector< std::complex<double> >& q1,
                        std::vector< std::complex<double> >& q2,
                        const rotations::rotator& rot = id_rot) const{
                    const auto k2 = esk * K->at(ij);
                    std::vector< std::complex<double> > core_vec  = 
                        {
                            cn,     cn,     cn,         cn,
                            cn,     cn,     cn,         -k2,
                            cn,     cn,     cn,         -ci * k2,
                            cn,     k2,   ci * k2,    cn
                        };
                    const matrix::cmat core(core_vec);
                    matrix::cmat r_core = rot.transform_term(core);
                    auto cq1 = vector::dot(r_core, q1);
                    return vector::dot(q2, cq1);
                };
                std::complex<double> Sp_term(
                        std::pair<std::size_t, std::size_t> const & ij,
                        std::vector< std::complex<double> >& q1,
                        std::vector< std::complex<double> >& q2,
                        const rotations::rotator& rot = id_rot) const{
                    const auto g3 = esk * G3->at(ij);
                    std::vector< std::complex<double> > core_vec  = 
                        {
                            cn,     cn,     cn,         cn,
                            cn,     cn,     cn,         g3,
                            cn,     cn,     cn,         ci * g3,
                            cn,     g3,     ci * g3,   cn
                        };
                    const matrix::cmat core(core_vec);
                    matrix::cmat r_core = rot.transform_term(core);
                    auto cq1 = vector::dot(r_core, q1);
                    return vector::dot(q2, cq1);
                };
                std::complex<double> Sm_term(
                        std::pair<std::size_t, std::size_t> const & ij,
                        std::vector< std::complex<double> >& q1,
                        std::vector< std::complex<double> >& q2,
                        const rotations::rotator& rot = id_rot) const{
                    const auto g3 = esk * G3->at(ij);
                    std::vector< std::complex<double> > core_vec  = 
                        {
                            cn,     cn,     cn,         cn,
                            cn,     cn,     cn,         g3,
                            cn,     cn,     cn,         -ci * g3,
                            cn,     g3,     -ci * g3,   cn
                        };
                    const matrix::cmat core(core_vec);
                    matrix::cmat r_core = rot.transform_term(core);
                    auto cq1 = vector::dot(r_core, q1);
                    return vector::dot(q2, cq1);
                };
                std::complex<double> C_term(
                        std::pair<std::size_t, std::size_t> const & ij,
                        std::vector< std::complex<double> >& q1,
                        std::vector< std::complex<double> >& q2,
                        const rotations::rotator& rot = id_rot) const{
                    return 2. * Qm_term(ij, q1, q2, rot);
                };
                std::complex<double> Stp_term(
                        std::pair<std::size_t, std::size_t> const & ij,
                        std::vector< std::complex<double> >& q1,
                        std::vector< std::complex<double> >& q2,
                        const rotations::rotator& rot = id_rot) const{
                    return -st3 * (Sp_term(ij, q1, q2, rot) + Qp_term(ij, q1, q2, rot));
                };
                std::complex<double> Stm_term(
                        std::pair<std::size_t, std::size_t> const & ij,
                        std::vector< std::complex<double> >& q1,
                        std::vector< std::complex<double> >& q2,
                        const rotations::rotator& rot = id_rot) const{
                    return -st3 * (Sm_term(ij, q1, q2, rot) + Qm_term(ij, q1, q2, rot));
                };
                std::complex<double> Swp_term(
                        std::pair<std::size_t, std::size_t> const & ij,
                        std::vector< std::complex<double> >& q1,
                        std::vector< std::complex<double> >& q2,
                        const rotations::rotator& rot = id_rot) const{
                    return -st3 * (Sp_term(ij, q1, q2, rot) - ot3 * Qp_term(ij, q1, q2, rot));
                };
                std::complex<double> Swm_term(
                        std::pair<std::size_t, std::size_t> const & ij,
                        std::vector< std::complex<double> >& q1,
                        std::vector< std::complex<double> >& q2,
                        const rotations::rotator& rot = id_rot) const{
                    return -st3 * (Sm_term(ij, q1, q2, rot) - ot3 * Qm_term(ij, q1, q2, rot));
                };
                std::vector<double> k_vec(
                    const std::size_t& i,
                    const std::pair<double, double>& kxky,
                    const rotations::rotator& rot = id_rot) const {
                        std::vector<double> k = 
                            {kxky.first, kxky.second, kz(i)};
                        return rot.transform_vector(k);
                };
                std::vector<double> q_vec(
                    const std::size_t& i,
                    const std::pair<double, double>& kxky,
                    const rotations::rotator& rot = id_rot) const {
                        const auto kv = k_vec(i, kxky, rot);
                        return {1., kv[0], kv[1], kv[2]};
                };
                matrix::cmat get_hblock(
                        const std::pair<std::size_t, std::size_t>& ij,
                        const std::pair<double, double>& kxky_in,
                        const rotations::rotator& rot = id_rot) const {
                    const auto  ji = pair_swap(ij);
                    const auto  q1r = q_vec(ij.first, kxky_in, rot),
                                q2r = q_vec(ji.first, kxky_in, rot);
                    auto        q1 = vector::real_copy(q1r),
                                q2 = vector::real_copy(q2r);
                    const std::pair<double, double>
                                kxky_1 = {q1r[1], q2r[2]},
                                kxky_2 = {q2r[1], q2r[2]};
                    const std::complex<double>
                                kz_1 = {q1r[3], 0.},
                                kz_2 = {q2r[3], 0.},
                                kp_1 = {kxky_1.first, kxky_1.second},
                                km_1 = {kxky_1.first, -kxky_1.second},
                                kp_2 = {kxky_2.first, kxky_2.second},
                                km_2 = {kxky_2.first, -kxky_2.second};
                    const auto  tt = T_term(ij, q1, q2),
                                ut = U_term(ij, q1, q2),
                                vt = V_term(ij, q1, q2),
                                rt = R_term(ij, q1, q2);
                    const auto  qp = Qp_term(ij, q1, q2),
                                qm = Qm_term(ij, q1, q2),
                                sp = Sp_term(ij, q1, q2),
                                sm = Sm_term(ij, q1, q2);
                    const auto  stm = -st3 * (sm + qm),
                                stp = -st3 * (sp + qp);
                    const auto  swm = -st3 * (sm - ot3 * qm),
                                swp = -st3 * (sp - ot3 * qp);    
                    const auto  ct = 2. * qm;           
                    const auto  p = P_full(ij),
                                pkz = p * kz_2,
                                kzp = kz_1 * p,
                                es = Es->at(ij);
                    const auto  rth = std::conj(R_term(ji, q2, q1)),
                                cth = std::conj(C_term(ji, q2, q1));
                    const auto  qph = Qp_term(ji, q2, q1),
                                qmh = Qm_term(ji, q2, q1),
                                sph = Sp_term(ji, q2, q1),
                                smh = Sm_term(ji, q2, q1); 
                    const auto  stmh = -st3 * std::conj(smh + qmh),
                                stph = -st3 * std::conj(sph + qph);
                    const auto  swmh = -st3 * std::conj(smh - ot3 * qmh),
                                swph = -st3 * std::conj(sph - ot3 * qph);
                    const std::vector< std::complex<double> > rv = 
                        {
                              tt,                       {0.,0.},                - p * kp_1 / st2,          st2 * pkz / st3,            p * km_1 / (st2 * st3),     {0.,0.},            - pkz / st3,            - p * km_1 / st3,
                              {0., 0},                  tt,                       {0., 0.},              - p * kp_1 / (st2 * st3),     st2 * pkz / st3,            p * km_1 / st2,     - p * kp_1 / st3,         pkz / st3,
                            - km_2 * p / st2,           {0., 0.},                 ut + vt,               - stm,                        rt,                         {0., 0.},             stm / st2,            - st2 * rt,
                              st2 * kzp / st3,        - km_2 * p / (st2 * st3), - stmh,                    ut - vt,                    ct,                         rt,                   st2 * vt,             - st3 * swm / st2,
                              kp_2 * p / (st2 * st3),   st2 * kzp / st3,          rth,                     cth,                        ut - vt,                    stph,               - st3 * swp / st2,      - st2 * vt,
                              {0., 0},                  kp_2 * p / st2,           {0., 0},                 rth,                        stp,                        ut + vt,              st2 * rth,              stp / st2,
                            - kzp / st3,              - km_2 * p / st3,           stmh / st2,              st2 * vt,                 - st3 * swph / st2,           st2 * rt,             ut - es,                ct,
                            - kp_2 * p / st3,           kzp / st3,              - st2 * rth,             - st3 * swmh / st2,         - st2 * vt,                   stph / st2,           cth,                    ut - es    
                        };
                    auto am = matrix::cmat(rv);
                    if(ij == std::pair<std::size_t, std::size_t>(0, 0)){
                        std::cout << std::endl << "0, 0 untransformed" << std::endl;
                        am.print();
                        std::cout << std::endl;
                    }
                    return rot.transform_hamiltonian(am);
                };
            matrix::herm full_h(
                const std::pair<double, double> kxky,
                const rotations::rotator* rot = &id_rot) const{
                    const std::size_t bsize = 
                        static_cast<std::size_t>(blims.second - blims.first + 1);
                    matrix::cmat rv(8 * bsize);
                    for(std::size_t r = 0; r < bsize; r++){
                        const std::size_t i = r * 8;
                        for(std::size_t c = r; c < bsize; c++){
                            const std::size_t j = c * 8;
                            auto ham = get_hblock({r, c}, kxky, *rot);
                            if((c == 0) && (r == 0)){
                                std::cout << "0,0:" << std::endl;
                                ham.print();
                            }
                            rv.put_submatrix(ham, {i, j});
                        }
                    }
                    auto ret_val = matrix::herm(rv);
                    return ret_val;
            };
    };
};

#endif