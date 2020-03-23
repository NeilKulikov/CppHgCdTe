#ifndef ROTATIONS
#define ROTATIONS

#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <exception>
#include <functional>

#include <matrix.hpp>
#include <vector.hpp>

#include <constants.hpp>

#include <iostream>

namespace rotations{
    const static std::complex<double> cn = {0., 0.};
    const static std::complex<double> co = {1., 0.};
    const static std::complex<double> ci = {0., 1.};
    namespace inner{
        static const std::vector< std::complex<double> > 
            scx =   {   
                        cn, co,
                        co, cn  
                    },
            scy =   {
                        cn, -ci,
                        ci,  cn
                    },
            scz =   {
                        co, cn,
                        cn, -co
                    },
            jcx =   {
                        cn,         st3 * co,           cn,         cn,
                        st3 * co,         cn,      2. * co,         cn,
                        cn,          2. * co,           cn,   st3 * co,
                        cn,               cn,     st3 * co,         cn
                    },
            jcy =   {
                        cn,        -st3 * ci,           cn,         cn,
                        st3 * ci,         cn,     -2. * ci,         cn,
                        cn,          2. * ci,           cn,  -st3 * ci,
                        cn,               cn,     st3 * ci,         cn
                    },
            jcz =   {
                        3. * co,               cn,           cn,         cn,
                             cn,               co,           cn,         cn,
                             cn,               cn,          -co,         cn,
                             cn,               cn,           cn,   -3. * co
                    };
        static const matrix::cmat 
                              sx = matrix::cmat(scx).scale(0.5 * co),
                              sy = matrix::cmat(scy).scale(0.5 * co),
                              sz = matrix::cmat(scz).scale(0.5 * co),
                              jx = matrix::cmat(jcx).scale(0.5 * co),
                              jy = matrix::cmat(jcy).scale(0.5 * co),
                              jz = matrix::cmat(jcz).scale(0.5 * co);

        matrix::rmat rot_x(const double phi){
            const double    sinx = sin(phi),
                            cosx = cos(phi);
            const std::vector<double> core = 
                {
                      1.,     0.,    0.,
                      0.,   cosx, -sinx,
                      0.,   sinx,  cosx 
                };
            return matrix::rmat(core);
        };
        matrix::rmat rot_y(const double phi){
            const double    sinx = sin(phi),
                            cosx = cos(phi);
            const std::vector<double> core = 
                {
                    cosx,     0., -sinx,
                      0.,     1.,    0.,
                    sinx,     0.,  cosx 
                };
            return matrix::rmat(core);
        };
        matrix::rmat rot_z(const double phi){
            const double    sinx = sin(phi),
                            cosx = cos(phi);
            const std::vector<double> core = 
                {
                    cosx,   sinx,   0.,
                   -sinx,   cosx,   0.,
                      0.,     0.,   1. 
                };
            return matrix::rmat(core);
        };
        matrix::cmat exp_rot(
            matrix::cmat core, 
            const std::complex<double> phi){
                const auto ev = matrix::herm(core).diagonalize_v();
                std::vector< std::complex<double> > nd(core.size());
                std::transform(
                    ev.first.cbegin(),
                    ev.first.cend(),
                    nd.begin(),
                    [=](const double x){ return std::exp(x * phi); }
                );
                const auto lambda = matrix::cmat::diagonal(nd);
                return ev.second * lambda * ev.second.conjugate();
        };
    };
    matrix::rmat vec_rot(
        const std::array<double, 3>& ang = {0., 0., 0.}){
        return 
            inner::rot_z(ang[2]) *
            inner::rot_y(ang[1]) *
            inner::rot_z(ang[0]);
    };
    matrix::rmat vec_rot(
        const double alpha = 0., 
        const double beta = 0., 
        const double gamma = 0.){
            return vec_rot({alpha, beta, gamma});
    };

    matrix::cmat J12_rot(
        const std::array<double, 3> ang = {0., 0., 0.}){
            return 
                inner::exp_rot(inner::sz, - ci * ang[2]) *
                inner::exp_rot(inner::sy, - ci * ang[1]) *
                inner::exp_rot(inner::sz, - ci * ang[0]);
    };
    matrix::cmat J12_rot(
        const double alpha = 0., 
        const double beta = 0., 
        const double gamma = 0.){
            return J12_rot({alpha, beta, gamma});
    };

    matrix::cmat J32_rot(
        const std::array<double, 3> ang = {0., 0., 0.}){
            return 
                inner::exp_rot(inner::jz, - ci * ang[2]) *
                inner::exp_rot(inner::jy, - ci * ang[1]) *
                inner::exp_rot(inner::jz, - ci * ang[0]);
    };
    matrix::cmat J32_rot(
        const double alpha = 0., 
        const double beta = 0., 
        const double gamma = 0.){
            return J32_rot({alpha, beta, gamma});
    };

    matrix::cmat ham_rot(
        const std::array<double, 3> ang = {0., 0., 0.}){
            matrix::cmat rv(8);
            auto    J12 = J12_rot(ang),
                    J32 = J32_rot(ang);
            rv.put_submatrix(J12, {0, 0});
            rv.put_submatrix(J32, {2, 2});
            rv.put_submatrix(J12, {6, 6});
            return rv;
    };
    matrix::cmat ham_rot(
        const double alpha = 0., 
        const double beta = 0., 
        const double gamma = 0.){
            return ham_rot({alpha, beta, gamma});
    };

    matrix::cmat term_rot(const matrix::rmat& v_rot){
        auto v_rot_copy = matrix::rmat::copy(v_rot);
        std::vector< std::complex<double> > diag{co, cn, cn, cn};
        auto cv_rot = matrix::cmat::real_copy(v_rot_copy);
        matrix::cmat ret = matrix::cmat::diagonal(diag);
        ret.put_submatrix(cv_rot, {1, 1});
        return ret;
    };

    class rotator{
        public:
            const std::array<double, 3> angles = {0., 0., 0.};
            const matrix::rmat v_rot, vr_rot;
            const matrix::cmat c_rot, cr_rot;
            const matrix::cmat t_rot, tr_rot;
            rotator(const std::array<double, 3>& ags) : 
                angles(ags), 
                v_rot(vec_rot(angles)), vr_rot(v_rot.transpose()),
                c_rot(ham_rot(angles)), cr_rot(c_rot.conjugate()),
                t_rot(term_rot(v_rot)), tr_rot(t_rot.conjugate()){
                    /*std::cout << std::endl;
                    matrix::rmat::copy(v_rot).print();
                    std::cout << std::endl;
                    matrix::cmat::copy(c_rot).print();
                    std::cout << std::endl;
                    matrix::cmat::copy(t_rot).print();
                    std::cout << std::endl;*/
                };
            rotator(
                const double alpha = 0., 
                const double beta = 0.,
                const double gamma = 0.) :
                    rotator({alpha, beta, gamma}) {};
            std::vector<double> transform_vector(std::vector<double>& vec) const{
                if(vec.size() != 3)
                    throw std::length_error("Vector should have 3 size");
                return vector::dot(v_rot, vec);
            };
            matrix::cmat transform_term(const matrix::cmat& tc) const{
                if(tc.size() != 4)
                    throw std::length_error("Term should have 4x4 size");
                return t_rot * tc * tr_rot;
            };
            matrix::cmat transform_hamiltonian(const matrix::cmat& hc) const{
                if(hc.size() != 8)
                    throw std::length_error("Kane hamiltonian should have 8x8 size");
                return c_rot * hc * cr_rot;
            };
    };
};

#endif