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
        double eps_rel = 1.e-5){
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
        double eps_rel = 1.e-5){
            bool ext_ws = (ws != nullptr);
            ws = ext_ws ? ws : gsl_integration_workspace_alloc(max_eval);
            std::function<F(X)> rfunc = [&](X x){ return func(x).real(); },
                                ifunc = [&](X x){ return func(x).imag(); };
            auto    rres = integrate_qag(rfunc, lims, ws, max_eval, ir, eps_abs, eps_rel),
                    ires = integrate_qag(ifunc, lims, ws, max_eval, ir, eps_abs, eps_rel);
            if(!ext_ws) gsl_integration_workspace_free(ws);
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
        double eps_rel = 1.e-5){
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
};

#endif