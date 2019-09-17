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
#include <operators.hpp>

namespace hamiltonian{

    class hcore : 
        public materials::model< std::shared_ptr<matrix::herm> >{
            public:
                double accuracy = 1.e-6;
                std::shared_ptr<matrix::herm> Kz = nullptr;
                std::shared_ptr<matrix::herm> PKz = nullptr;
                std::shared_ptr<matrix::herm> AG3Kz = nullptr;
                std::shared_ptr<matrix::herm> CKKz = nullptr;
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
                        {"Kz", Kz },
                        {"PKz", PKz },
                        {"AG3Kz", AG3Kz },
                        {"CKKz", CKKz }
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
                    auto kzcmat = matrix::cmat::diagonal(kzs);
                    Kz = std::shared_ptr<matrix::herm>(new matrix::herm(kzcmat));
                    auto _pkz = (*Ep) * (*Kz);
                    PKz = std::shared_ptr<matrix::herm>(new matrix::herm(_pkz));
                    auto _ag3kz = ((*G3) * (*Kz)) + ((*Kz) * (*G3));
                    AG3Kz = std::shared_ptr<matrix::herm>(new matrix::herm(_ag3kz));
                    auto _ckkz = ((*K) * (*Kz)) - ((*Kz) * (*K));
                    CKKz = std::shared_ptr<matrix::herm>(new matrix::herm(_ckkz));
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
                    
                    const auto Ec = Eg->at(ij) + VBO->at(ij);
                    return Ec;
                };
                matrix::herm get_hblock(
                        const std::pair<std::size_t, std::size_t> ij,
                        const std::pair<double, double> kxky) const {
                    using namespace std::complex_literals;
                    const std::complex<double> 
                        kp = {kxky.first, kxky.second},
                        km = std::conj(kp);
                    
                    std::vector< std::complex<double> > rv = 
                        {
                            {0, 0}
                        };
                    auto am = matrix::cmat(rv);
                    auto cm = matrix::cmat::copy(am);
                    return matrix::herm(cm);
                };
            
    };
};

#endif