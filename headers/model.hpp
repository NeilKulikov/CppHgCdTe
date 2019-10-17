#ifndef MODEL
#define MODEL

#include <vector>
#include <iostream>
#include <algorithm>
#include <exception>

#include <spline.hpp>
#include <service.hpp>

namespace materials{

    template<typename T = double>
    struct model{
        T Eg;
        T Es;
        T Ep;
        T VBO;
        T G1;
        T G2;
        T G3;
        T F;
        T K;
    };

    const static model<double> CdTe = { 1.606, 0.91,  18.8, -0.57, 1.47, -0.28, 0.03, -0.09, -1.31};
    const static model<double> HgTe = {-0.303, 1.08,  18.8, 0.,    4.1,   0.5,   1.3,  0.,    -0.4};

    model<double> CdHgTe(double x){
        if(!((0. <= x) && (x <= 1.)))
            throw std::domain_error("x should be in range [0;1]");
        double Eg = 1.606 * x - 0.303 * (1. - x) - 0.132 * x * (1. - x);
        return  {
                    Eg,
                    lin(CdTe.Es, HgTe.Es, x),
                    lin(CdTe.Ep, HgTe.Ep, x),
                    lin(CdTe.VBO, HgTe.VBO, x),
                    lin(CdTe.G1, HgTe.G1, x),
                    lin(CdTe.G2, HgTe.G2, x),
                    lin(CdTe.G3, HgTe.G3, x),
                    lin(CdTe.F, HgTe.F, x),
                    lin(CdTe.K, HgTe.K, x)
                };
    };

    #include <heterostruct.hpp>
};

#endif