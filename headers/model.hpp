#ifndef MODEL
#define MODEL

#include <vector>
#include <algorithm>
#include <exception>

#include <spline.hpp>

namespace materials{
    struct model{
        double Eg;
        double Es;
        double Ep;
        double G1;
        double G2;
        double G3;
        double F;
        double K;
    };

    const static model CdTe = {1.606, 0.91, 18.8, 1.47, -0.28, 0.03, -0.09, -1.31};
    const static model HgTe = {-0.303, 1.08, 18.8, 4.1, 0.5, 1.3, 0., -0.4};

    inline double lin(double a, double b, double x){
        return a * x + b * (1. - x);
    }

    model CdHgTe(double x){
        if(!((0. <= x) && (x <= 1.)))
            throw std::domain_error("x should be in range [0;1]");
        double Eg = 1.606 * x - 0.303 * (1. - x) - 0.132 * x * (1. - x);
        return  {
                    Eg,
                    lin(CdTe.Es, HgTe.Es, x),
                    lin(CdTe.Ep, HgTe.Ep, x),
                    lin(CdTe.G1, HgTe.G1, x),
                    lin(CdTe.G2, HgTe.G2, x),
                    lin(CdTe.G3, HgTe.G3, x),
                    lin(CdTe.F, HgTe.F, x),
                    lin(CdTe.K, HgTe.K, x)
                };
    };

    class heterostruct{
        private:
            double len = 0.;
            std::shared_ptr<staff::spline> spl = nullptr;
        public:
            heterostruct(
                std::vector<double> const & zs, 
                std::vector<double> const & xs){
                    const bool zsorted = std::is_sorted(zs.begin(), zs.end());
                    if(!zsorted)
                        throw std::logic_error("z should be sorted");
                    const auto zmm = std::minmax_element(zs.begin(), zs.end());
                    len = *zmm.second - *zmm.first;
                    auto zc = zs;
                    std::for_each(zc.begin(), zc.end(), [zmm](double & z){ z = z - *zmm.first; });
                    const auto xmm = std::minmax_element(xs.begin(), xs.end());
                    if(!((0. <= *xmm.first) && (*xmm.second <= 1.)))
                        throw std::domain_error("x should be in range [0;1]");
                    spl = std::make_shared<staff::spline>(zc, xs);
            };
            double length(void) const {
                return len;
            };
            double composition(double z){
                if(!((0. <= z) && (z <= len)))
                    throw std::out_of_range("z should be in interval [0;L]");
                return spl->eval(z);
            };
            model at(double z){
                double comp = composition(z);
                auto rv = CdHgTe(comp);
                return std::move(rv);
            };
            
    };
};

#endif