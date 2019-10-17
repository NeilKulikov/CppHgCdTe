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

    template<typename T = double>
    class str_mod{
        protected:
            std::array<T, 6> tensor;
        public:
            str_mod(void) :
                tensor({0., 0., 0., 0., 0., 0.}) {};
            str_mod(std::array<T, 6> const& inp) :
                tensor(inp) {};
            T at(std::size_t i, std::size_t j) const{
                const auto  in = std::min(i, j),
                            jn = std::max(i, j);
                if(jn > 2)
                    throw std::out_of_range("Index of tensor should be in range 0...3");
                const auto  jt = jn + static_cast<std::size_t>(jn == 2);
                return tensor[in + jt];
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
                auto  MB21 = matrix::rmat(B21),
                            IB22 = matrix::rmat(B22).inverse();
                IB22.print();
                const double str0 = (bufa - md.A) / md.A;
                std::array<double, 3> str1 = { str0, 0., str0 };
                auto alpha = vector::dot(MB21, str1);
                auto str2 = vector::dot(IB22, alpha);
                tensor = { str1[0], str1[1], str1[2], str2[0], str2[1], str2[2] };
            };
            strain(double x, double bufx = 0.7) :
                strain(CdHgTe(x), bufx) {};
    };

    /*class strhtr{
        protected:
            heterostruct hs;
            str_mode<
        public:
            model<double> model(double z) const;
            strain strain(double z) const;
    };*/
};
};

#endif