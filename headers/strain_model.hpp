#ifndef STRAIN_MODEL
#define STRAIN_MODEL

#include <map>
#include <array>
#include <vector>
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
            str_mod(str_mod<T> const& inp) :
                tensor(inp.tensor) {};
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
        using grid_type = std::map<double, str_mod<double> >;
        protected:
            heterostruct hs;
            str_mod< std::shared_ptr<staff::spline> > str;
        public:
            double length(void) const{
                return hs.length();
            };
            model<double> get_model(double z) const{
                return hs.at(z);
            };
        protected:
            static std::pair<double, str_mod<double> > make_point(
                heterostruct const & hs,
                const double z,
                const double bufx = 0.7){
                    //std::cout << __func__ << std::endl;
                    //std::cout << length() << std::endl;
                    const auto mod_c = hs.at(z);
                    //std::cout << __func__ << std::endl;
                    const auto str_c = strain(mod_c, bufx);
                    //std::cout << __func__ << std::endl;
                    return std::make_pair(z, str_c);
            };
            static grid_type ext_grid(
                heterostruct const & hs,
                std::vector<double> const& zs, 
                const double bufx = 0.7){
                    //std::cout << __func__ << std::endl;
                    grid_type ret_v;
                    std::for_each(
                        zs.cbegin(), 
                        zs.cend(),
                        [&](double z){
                            const auto res_p = 
                                make_point(hs, z, bufx);
                            ret_v.emplace(res_p);
                            //std::cout << "EMPLACE" << std::endl;
                        });
                    //std::cout << __func__ << std::endl;
                    return ret_v;
            };
            static grid_type grid(
                heterostruct const & hs,
                const double bufx = 0.7,
                const std::size_t npoints = 1024){
                    //std::cout << __func__ << std::endl;
                    const double step = hs.length() / 
                            static_cast<double>(npoints);
                    std::vector<double> zs;
                    for(std::size_t i = 0; i <= npoints; i++){
                        const double z = step *
                                        static_cast<double>(i);
                        zs.push_back(z);
                    }
                    //std::cout << __func__ << std::endl;
                    return ext_grid(hs, zs, bufx);
            };
            grid_type merge(
                grid_type const & a,
                grid_type const & b) const{
                    grid_type ret_v;
                    ret_v.insert(a.cbegin(), a.cend());
                    ret_v.insert(b.cbegin(), b.cend());
                    return ret_v;
            };
        public:
            strhtr(
                heterostruct const & hs,
                grid_type gt): hs(hs){
                    //std::cout << __func__ << std::endl;
                    str_mod< std::vector<double> > str_v;
                    std::vector<double> zs(gt.size());
                    //std::cout << gt.size() << std::endl;
                    std::transform(
                        gt.cbegin(), 
                        gt.cend(),
                        zs.begin(),
                        [&](const auto& p){
                            //std::cout << p.first << std::endl;
                            return p.first;
                        });
                    //std::cout << __func__ << std::endl;
                    for(std::size_t j = 0; j < 6; j++){
                        str_v.raw().at(j) = 
                                std::vector<double>(gt.size());
                        std::transform(
                            gt.cbegin(), 
                            gt.cend(),
                            str_v.raw().at(j).begin(),
                            [&](auto& p){
                                return p.second.get().at(j);
                            });
                    }
                    //std::cout << __func__ << std::endl;
                    std::transform(
                        str_v.raw().begin(), 
                        str_v.raw().end(), 
                        str.raw().begin(),
                        [&](auto& xs){ 
                            return std::shared_ptr<staff::spline>(
                                                new staff::spline(zs, xs));
                        });
                    //std::cout << __func__ << std::endl;
            };
            strhtr(
                heterostruct hs, 
                const double bufx = 0.7,
                const std::size_t npoints = 1024): 
                    strhtr(hs, grid(hs, bufx, npoints)) {};
            strhtr(
                heterostruct hs,
                std::vector<double> const& zs,
                const double bufx = 0.7,
                const std::size_t npoints = 1024):
                    strhtr(
                        hs, 
                        merge(
                            grid(hs, bufx, npoints),
                            ext_grid(hs, zs, bufx)
                        )
                    ) {};
            strhtr(
                std::vector<double> const& zs, 
                std::vector<double> const& xs,
                const double bufx = 0.7,
                const std::size_t npoints = 1024):
                    strhtr(
                        heterostruct(zs, xs), 
                        zs, bufx, npoints) {};
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