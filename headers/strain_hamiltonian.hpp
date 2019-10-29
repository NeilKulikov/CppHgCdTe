#ifndef STRAIN_HAMILTONIAN
#define STRAIN_HAMILTONIAN

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

#include <strain_model.hpp>
#include <matrix.hpp>
#include <vector.hpp>
#include <constants.hpp>
#include <operators.hpp>

namespace strain{
namespace hamiltonian{

    const static std::complex<double> cn = {0., 0.};
    const static std::complex<double> co = {1., 0.};
    const static std::complex<double> ci = {0., 1.};

    class hcore : strain::materials::model< std::shared_ptr<matrix::herm> >{
            public:
                double accuracy = 1.e-6;
                std::size_t integ_space = 16384;
                std::shared_ptr<matrix::cmat> Rterm = nullptr;
                std::shared_ptr<matrix::cmat> Sterm = nullptr;
                std::shared_ptr<matrix::cmat> Tterm = nullptr;
                std::shared_ptr<matrix::cmat> Uterm = nullptr;
                std::shared_ptr<matrix::cmat> Vterm = nullptr;
            protected:
                std::array< std::shared_ptr<matrix::herm>, 6 > str;
                double len = 0.;
                std::pair<int, int> blims = {0, 1};
                std::map< std::string, std::shared_ptr<matrix::herm>& > mapping =
                    {
                        {"exx", str[0] },
                        {"exy", str[1] },
                        {"eyx", str[1] },
                        {"eyy", str[2] },
                        {"exz", str[3] },
                        {"ezx", str[3] },
                        {"eyz", str[4] },
                        {"ezy", str[4] },
                        {"ezz", str[5] },
                        {"B", B },
                        {"D", D },
                        {"Ac", Ac },
                        {"Av", Av }
                    };
                void fill_model(strain::materials::strhtr const& sh){
                    std::map< std::string, std::function<double(double)> > flow = 
                        {
                            {"exx", [&](double x){ return sh.get_strain(x)[0]; } },
                            {"exy", [&](double x){ return sh.get_strain(x)[1]; } },
                            {"eyy", [&](double x){ return sh.get_strain(x)[2]; } },
                            {"exz", [&](double x){ return sh.get_strain(x)[3]; } },
                            {"eyz", [&](double x){ return sh.get_strain(x)[4]; } },
                            {"ezz", [&](double x){ return sh.get_strain(x)[5]; } },
                            {"B", [&](double x){ return sh.get_model(x).B; } },
                            {"D", [&](double x){ return sh.get_model(x).D; } },
                            {"Ac", [&](double x){ return sh.get_model(x).Ac; } },
                            {"Av", [&](double x){ return sh.get_model(x).Av; } }
                        };
                    std::for_each(
                        flow.begin(), flow.end(),
                        [&](auto& it){
                            //std::cout << it.first << std::endl;
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
                void fill_terms(void){
                    {
                        const auto trace =  (*str[0]) + (*str[2]) + (*str[5]);
                        auto tt = (*Ac) * trace;
                        Tterm = std::shared_ptr<matrix::cmat>(new matrix::cmat(tt));
                        auto ut = (*Av) * trace;
                        Uterm = std::shared_ptr<matrix::cmat>(new matrix::cmat(ut));
                    }
                    {
                        auto vcore = (*str[0]) + (*str[2]) - (*str[5]).scale(2. * co);
                        auto vt = ((*B) * vcore).scale(0.5 * co);
                        Vterm = std::shared_ptr<matrix::cmat>(new matrix::cmat(vt));
                    }
                    {
                        auto st = (*D).scale(- co) * 
                                                ((*str[3]) - (*str[4]).scale({0., 1.}));
                        Sterm = std::shared_ptr<matrix::cmat>(new matrix::cmat(st));
                    }
                    {
                        const double fac = - st3 * 0.5;
                        auto rt = ((*B).scale(fac * co) * ((*str[0]) - (*str[2])))
                            + ((*D).scale(ci) * (*str[1]));
                        Rterm = std::shared_ptr<matrix::cmat>(new matrix::cmat(rt));
                    }
                };  
            public:
                hcore(  strain::materials::strhtr const & sh, 
                        std::size_t basis_size = 101,
                        const double acc = 1.e-6){
                    len = sh.length();
                    accuracy = acc;
                    blims = {- (basis_size / 2), basis_size / 2};
                    fill_model(sh);
                    //std::cout << "Model filled" << std::endl;
                    fill_terms();
                    //std::cout << "Terms filled" << std::endl;
                };
                std::shared_ptr<matrix::herm>& at(std::string const & in){
                    return mapping.at(in);
                };
                matrix::cmat get_hblock(
                    const std::pair<std::size_t, std::size_t> ij) const{
                    const auto  ji = std::make_pair(ij.second, ij.first);
                    const auto  tt = Tterm->at(ij),
                                rt = Rterm->at(ij),
                                ut = Uterm->at(ij),
                                vt = Vterm->at(ij),
                                st = Sterm->at(ij);
                    const auto  rth = std::conj(Rterm->at(ji)),
                                sth = std::conj(Sterm->at(ji));        
                    std::vector< std::complex<double> > rv = 
                        {
                            tt,     cn,     cn,             cn,             cn,             cn,         cn,             cn,
                            cn,     tt,     cn,             cn,             cn,             cn,         cn,             cn,
                            cn,     cn,   ut + vt,          st,             rt,             cn,      -st / st2,     -st2 * rt,
                            cn,     cn,    sth,           ut - vt,          cn,             rt,      st2 * vt,    st3 * st / st2,
                            cn,     cn,    rth,             cn,           ut - vt,         -st,   st3 * sth / st2,  -st2 * vt,
                            cn,     cn,     cn,            rth,            -sth,          ut + vt,   st2 * rth,     -sth / st2,
                            cn,     cn, -sth / st2,      st2 * vt,    st3 * st / st2,    st2 * rt,      ut,             cn,
                            cn,     cn, -st2 * rth,   st3 * sth / st2,  -st2 * vt,       -st / st2,     cn,             ut
                        };
                    auto am = matrix::cmat(rv);
                    return matrix::cmat::copy(am);
                };
                matrix::herm full_h(void) const{
                    std::size_t bsize = 
                            static_cast<std::size_t>(blims.second - blims.first + 1);
                    matrix::cmat rv(8 * bsize);
                    for(std::size_t r = 0; r < bsize; r++){
                        std::size_t i = r * 8;
                        for(std::size_t c = r; c < bsize; c++){
                            std::size_t j = c * 8;
                            auto ham = get_hblock({r, c});
                            rv.put_submatrix(ham, {i, j});
                        }
                    }
                    auto ret_val = matrix::herm(rv);
                    return ret_val;
                };
    };

};
};

#endif