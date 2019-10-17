#ifndef STRAIN_MODEL
#define STRAIN_MODEL

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

    std::vectorget_strain
};
};

#endif