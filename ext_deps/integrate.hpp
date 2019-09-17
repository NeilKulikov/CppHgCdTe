#ifndef INTEGRATE
#define INTEGRATE

#include <list>
#include <utility>
#include <numeric>
#include <complex>
#include <iostream>
#include <functional>

#include <gsl/gsl_integration.h>

namespace staff{
    enum class integr_rule{
        gauss_15 = GSL_INTEG_GAUSS15,
        gauss_21 = GSL_INTEG_GAUSS21,
        gauss_31 = GSL_INTEG_GAUSS31,
        gauss_41 = GSL_INTEG_GAUSS41,
        gauss_51 = GSL_INTEG_GAUSS51,
        gauss_61 = GSL_INTEG_GAUSS61
    };

    template<typename F, typename X>
    double integrand(double x, void* param){
        auto fptr = reinterpret_cast<std::function<F(X)>*>(param);
        X nx = static_cast<X>(x);
        F rv = (*fptr)(nx);
        return static_cast<double>(rv); 
    };

    template<typename F = double, typename X = double>
    std::pair<F, double> integrate_qag(
        std::function<F(X)>& func, 
        std::pair<X, X> lims,
        gsl_integration_workspace* ws = nullptr,
        std::size_t max_eval = 4096,
        int ir = GSL_INTEG_GAUSS21,
        double eps_abs = 1.e-8, 
        double eps_rel = 1.e-4){
            bool ext_ws = (ws != nullptr);
            ws = ext_ws ? ws : gsl_integration_workspace_alloc(max_eval);
            gsl_function f = {&integrand<F,X>, reinterpret_cast<void*>(&func)};
            double ret_val, ret_err;
            gsl_integration_qag(
                &f, 
                static_cast<double>(lims.first), 
                static_cast<double>(lims.second), 
                eps_abs, 
                eps_rel, 
                max_eval,
                ir,
                ws, 
                &ret_val, 
                &ret_err);
            if(!ext_ws) gsl_integration_workspace_free(ws);
            return std::make_pair(static_cast<F>(ret_val), ret_err);
    };

    template<typename F = double, typename X = double>
    std::pair<std::complex<F>, double> 
    integrate_qag(
        std::function<std::complex<F>(X)>& func, 
        std::pair<X, X> lims,
        gsl_integration_workspace* ws = nullptr,
        std::size_t max_eval = 4096,
        int ir = GSL_INTEG_GAUSS21,
        double eps_abs = 1.e-8, 
        double eps_rel = 1.e-4){
            bool ext_ws = (ws != nullptr);
            ws = ext_ws ? ws : gsl_integration_workspace_alloc(max_eval);
            std::function<F(X)> rfunc = [&](X x) -> F { return func(x).real(); },
                                ifunc = [&](X x) -> F { return func(x).imag(); };
            auto    rres = integrate_qag(rfunc, lims, ws, max_eval, ir, eps_abs, eps_rel),
                    ires = integrate_qag(ifunc, lims, ws, max_eval, ir, eps_abs, eps_rel);
            if(!ext_ws) gsl_integration_workspace_free(ws);
            return std::make_pair(
                    std::complex<F>(rres.first, ires.first), 
                    rres.second + ires.second);
    };

    template<typename F = double, typename X = double>
    std::pair<F, double> integrate_cquad(
        std::function<F(X)>& func, 
        std::pair<X, X> lims,
        gsl_integration_cquad_workspace* ws = nullptr,
        std::size_t max_eval = 4096,
        double eps_abs = 1.e-8, 
        double eps_rel = 1.e-4){
            bool ext_ws = (ws != nullptr);
            ws = ext_ws ? ws : gsl_integration_cquad_workspace_alloc(max_eval);
            gsl_function f = {&integrand<F,X>, reinterpret_cast<void*>(&func)};
            double ret_val, ret_err;
            std::size_t nits;
            gsl_integration_cquad(
                &f, 
                static_cast<double>(lims.first), 
                static_cast<double>(lims.second), 
                eps_abs, 
                eps_rel,
                ws, 
                &ret_val, 
                &ret_err,
                &nits);
            if(!ext_ws) gsl_integration_cquad_workspace_free(ws);
            return std::make_pair(static_cast<F>(ret_val), ret_err);
    };

    template<typename F = double, typename X = double>
    std::pair<std::complex<F>, double> 
    integrate_cquad(
        std::function<std::complex<F>(X)>& func, 
        std::pair<X, X> lims,
        gsl_integration_cquad_workspace* ws = nullptr,
        std::size_t max_eval = 4096,
        double eps_abs = 1.e-8, 
        double eps_rel = 1.e-4){
            bool ext_ws = (ws != nullptr);
            ws = ext_ws ? ws : gsl_integration_cquad_workspace_alloc(max_eval);
            std::function<F(X)> rfunc = [&](X x) -> F { return func(x).real(); },
                                ifunc = [&](X x) -> F { return func(x).imag(); };
            auto    rres = integrate_cquad(rfunc, lims, ws, max_eval, eps_abs, eps_rel),
                    ires = integrate_cquad(ifunc, lims, ws, max_eval, eps_abs, eps_rel);
            if(!ext_ws) gsl_integration_cquad_workspace_free(ws);
            return std::make_pair(
                    std::complex<F>(rres.first, ires.first), 
                    rres.second + ires.second);
    };

    template<typename F = double, typename X = double>
    std::pair<F, double> integrate_qags(
        std::function<F(X)>& func, 
        std::pair<X, X> lims,
        gsl_integration_workspace* ws = nullptr,
        std::size_t max_eval = 4096,
        double eps_abs = 1.e-8, 
        double eps_rel = 1.e-4){
            bool ext_ws = (ws != nullptr);
            ws = ext_ws ? ws : gsl_integration_workspace_alloc(max_eval);
            gsl_function f = {&integrand<F,X>, reinterpret_cast<void*>(&func)};
            double ret_val, ret_err;
            gsl_integration_qags(
                &f, 
                static_cast<double>(lims.first), 
                static_cast<double>(lims.second), 
                eps_abs, 
                eps_rel, 
                max_eval,
                ws, 
                &ret_val, 
                &ret_err);
            if(!ext_ws) gsl_integration_workspace_free(ws);
            return std::make_pair(static_cast<F>(ret_val), ret_err);
    };

    template<typename F = double, typename X = double>
    std::pair<std::complex<F>, double> 
    integrate_qags(
        std::function<std::complex<F>(X)>& func, 
        std::pair<X, X> lims,
        gsl_integration_workspace* ws = nullptr,
        std::size_t max_eval = 4096,
        double eps_abs = 1.e-8, 
        double eps_rel = 1.e-4){
            bool ext_ws = (ws != nullptr);
            ws = ext_ws ? ws : gsl_integration_workspace_alloc(max_eval);
            std::function<F(X)> rfunc = [&](X x){ return func(x).real(); },
                                ifunc = [&](X x){ return func(x).imag(); };
            auto    rres = integrate_qags(rfunc, lims, ws, max_eval, eps_abs, eps_rel),
                    ires = integrate_qags(ifunc, lims, ws, max_eval, eps_abs, eps_rel);
            if(!ext_ws) gsl_integration_workspace_free(ws);
            return std::make_pair(
                    std::complex<F>(rres.first, ires.first), 
                    rres.second + ires.second);
    };

    std::pair<std::complex<double>, double> 
    integrate(
        const std::function<std::complex<double>(double)> & func, 
        const std::pair<double, double> lims,
        const std::size_t max_eval = 4096){
            const std::size_t   nsteps = max_eval - max_eval % 2,
                                hsteps = nsteps / 2;
            const double    length = (lims.second - lims.first),
                            step = length / static_cast<double>(nsteps);
            std::complex<double> sigmas[2] = {{0., 0.}, {0., 0.}};
            for(std::size_t st = 1; st + 1 < hsteps; st++){
                double  ax = lims.first + step * static_cast<double>(2 * st + 1),
                        bx = lims.first + step * static_cast<double>(2 * st);
                sigmas[0] += func(ax);
                sigmas[1] += func(bx);
            }
            const std::complex<double>  sum = sigmas[0] + sigmas[1],
                                        igr = sum * step;
            const double rerr = std::abs(sigmas[0] / sum) - 1.;
            return std::make_pair(igr, rerr);
        };
};

#endif