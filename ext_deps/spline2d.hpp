#ifndef SPLINE_2D
#define SPLINE_2D

#include <vector>
#include <exception>
#include <algorithm>

#include <gsl/gsl_interp.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>

namespace staff{

    class spline2d{
        private:
            const std::size_t   lenx = 0,
                                leny = 0;
            gsl_spline2d*   spl = nullptr;
            gsl_interp_accel*   xacc = gsl_interp_accel_alloc(),
                                yacc = gsl_interp_accel_alloc();
            std::vector<double> za;
        public:
            spline2d(
                std::vector<double> const & xs, 
                std::vector<double> const & ys,
                std::vector<double> const & zs,
                gsl_interp2d_type const * type = gsl_interp2d_bicubic):
                    lenx(xs.size()), leny(ys.size()), za(lenx * leny),
                    spl(gsl_spline2d_alloc(type, lenx, leny)){
                        if(zs.size() == lenx * leny)
                            throw std::length_error("Xs and Ys shoul have equal size");
                        if( !std::is_sorted(xs.cbegin(), xs.cend()) 
                                ||
                            !std::is_sorted(ys.cbegin(), ys.cend()))
                                throw std::length_error("Xs and Ys should be sorted");
                        for(std::size_t i = 0; i < lenx; i++){
                            for(std::size_t j = 0; j < leny; j++){
                                gsl_spline2d_set(spl, za.data(), i, j, )
                            }
                        }
                        gsl_spline_init(spl, xs.data(), ys.data(), len);
            };
            ~spline2d(){
                gsl_spline2d_free(spl);
                gsl_interp_accel_free(xacc);
                gsl_interp_accel_free(yacc);
            };
    };

};

#endif