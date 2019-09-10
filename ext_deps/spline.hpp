#ifndef SPLINE
#define SPLINE

#include <vector>
#include <exception>

#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>

namespace staff{

    class spline{
        private:
            std::size_t len = 0;
            gsl_spline* spl = nullptr;
            gsl_interp_accel* acc = nullptr;
        public:
            spline(std::vector<double> const & xs, std::vector<double> const & ys,
                gsl_interp_type const * type = gsl_interp_linear){
                    len = xs.size();
                    if(xs.size() != ys.size())
                        throw std::length_error("Xs and Ys shoul have equal size");
                    spl = gsl_spline_alloc(type, len);
                    acc = gsl_interp_accel_alloc();
                    gsl_spline_init(spl, xs.data(), ys.data(), len);
            };
            ~spline(){
                gsl_spline_free(spl);
                gsl_interp_accel_free(acc);
            }
            double eval(double x) const {
                return gsl_spline_eval(spl, x, acc);
            }
    };
};

#endif