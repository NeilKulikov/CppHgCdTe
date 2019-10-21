#ifndef STRAIN_MODEL
#define STRAIN_MODEL

#include <map>
#include <array>
#include <string>
#include <exception>
#include <algorithm>
#include <iostream>

#include <matrix.hpp>
#include <vector.hpp>
#include <spline.hpp>
#include <service.hpp>

namespace strain{
namespace materials{
    template<typename T = double>
    struct model{
        T A;
        T B;
        T D;
        T Ac;
        T Av;
        T C11;
        T C12;
        T C44;
    };

    const static model<double> CdTe = { 6.48, -1.2, -5.4, -2.925, 0.00, 5.62, 3.94, 2.06};
    const static model<double> HgTe = { 6.46, -1.5, -2.5, -2.380, 1.31, 5.92, 4.14, 2.19};
    const static double GaAsA = 5.653;

    model<double> CdHgTe(double x){
        if(!((0. <= x) && (x <= 1.)))
            throw std::domain_error("x should be in range [0;1]");
        return  {
                    lin(CdTe.A, HgTe.A, x),
                    lin(CdTe.B, HgTe.B, x),
                    lin(CdTe.D, HgTe.D, x),
                    lin(CdTe.Ac, HgTe.Ac, x),
                    lin(CdTe.Av, HgTe.Av, x),
                    lin(CdTe.C11, HgTe.C11, x),
                    lin(CdTe.C12, HgTe.C12, x),
                    lin(CdTe.C44, HgTe.C44, x)
                };
    };

    #include <heterostruct.hpp>

    class strhtr;

    template<typename T = double>
    class str_mod{
        friend strhtr;
        protected:
            std::array<T, 6> tensor;
            std::array<T, 6>& raw(void){
                return tensor;
            };
        public:
            str_mod(void): 
                tensor({T(), T(), T(), T(), T(), T()}) {};
            str_mod(std::array<T, 6> const& inp) :
                tensor(inp) {};
            static std::size_t index(
                    const std::size_t i, 
                    const std::size_t j){
                const auto  in = std::min(i, j),
                            jn = std::max(i, j);
                if(jn > 2)
                    throw std::out_of_range("Index of tensor should be in range 0...3");
                const auto  jt = jn + static_cast<std::size_t>(jn == 2);
                return in + jt;
            };
            T at(std::size_t i) const{
                return tensor.at(i);
            }
            T operator[] (std::size_t i) const{
                return at(i);
            }
            T at(std::size_t i, std::size_t j) const{
                return at(index(i, j));
            };
            T at(std::pair<std::size_t, std::size_t> ij) const{
                return at(ij.first, ij.second);
            };
            std::array<T, 6> const & get(void) const{
                return tensor;
            };
    };

    class strain : public str_mod<double>{
        public:
            strain(std::array<double, 6> const & inp)
                : str_mod<double>(inp) {};
            strain(
                model<double> const & md,
                double bufx = 0.7){
                if(!((0. <= bufx) && (bufx <= 1.)))
                    throw std::out_of_range("Buf x should be in range 0...1");
                const double bufa = CdHgTe(bufx).A;
                /*
                    C = ((B11, B12), (B21, B22))
                */
                std::vector<double> B21 = {
                        0.,     0.,     0.,
                        0.,     0.,     0.,
                    md.C12,     0.,  md.C12
                };
                std::vector<double> B22 = {
                    2. * md.C44,     0.,     0.,
                    0.,     2. * md.C44,     0.,
                    0.,              0.,  md.C11
                };
                const auto  MB21 = matrix::rmat(B21),
                            IB22 = matrix::rmat(B22).inverse();
                //IB22.print();
                const double str0 = (bufa - md.A) / md.A;
                std::array<double, 3> str1 = { str0, 0., str0 };
                auto alpha = vector::dot(MB21, str1);
                auto str2 = vector::mul(vector::dot(IB22, alpha), -1.);
                std::copy(str1.cbegin(), str1.cend(), tensor.begin());
                std::copy(str2.cbegin(), str2.cend(), tensor.begin() + 3);
            };
            strain(double x, double bufx = 0.7) :
                strain(CdHgTe(x), bufx) {};
    };

    class strhtr{
        friend class str_mod<double>;
        protected:
            heterostruct hs;
            str_mod< std::shared_ptr<staff::spline> > str;
        public:
            strhtr(
                heterostruct hs, 
                const double bufx = 0.7,
                const std::size_t npoints = 1024) : hs(hs){
                    const double step = hs.length() / 
                                static_cast<double>(npoints);
                    str_mod< std::vector<double> > str_v;
                    std::vector<double> zs;
                    for(std::size_t i = 0; i <= npoints; i++){
                        const double z = step *
                                        static_cast<double>(i);
                        const auto mod_c = hs.at(z);
                        const auto str_c = strain(mod_c, bufx).get();
                        zs.push_back(z);
                        for(std::size_t j = 0; j < 6; j++)
                            str_v.raw().at(j).push_back(str_c.at(j));
                    }
                    std::transform(
                        str_v.raw().begin(), 
                        str_v.raw().end(), 
                        str.raw().begin(),
                        [&](auto& xs){ 
                            return std::shared_ptr<staff::spline>(
                                                new staff::spline(zs, xs));
                        });
            };
            strhtr(
                std::vector<double> const& zs, 
                std::vector<double> const& xs,
                const double bufx = 0.7,
                const std::size_t npoints = 1024):
                    strhtr(heterostruct(zs, xs), bufx, npoints) {};
            double length(void) const{
                return hs.length();
            };
            model<double> get_model(double z) const{
                return hs.at(z);
            };
            strain get_strain(double z) const {
                std::array<double, 6> rv;
                std::transform(
                    str.get().cbegin(), 
                    str.get().cend(), 
                    rv.begin(),
                    [&](auto& x){ return x->eval(z); }
                );
                return strain(rv);
            };
    };
};
};

#endif