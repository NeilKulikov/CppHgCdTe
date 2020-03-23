#ifndef DERIVATE
#define DERIVATE

#include <utility>
#include <algorithm>
#include <exception>
#include <functional>

#include <iostream>

#include <gsl/gsl_deriv.h>

#include "./vector.hpp"

namespace staff{
    namespace derive{
        using func_1d = std::function<double(double)>;
        template<std::size_t dim>
        using array = std::array<double, dim>;
        template<std::size_t dim, typename f_arg_type = const array<dim>&>
        using func_nd = std::function<double(f_arg_type)>;

        template<std::size_t dim>
        constexpr array<dim> eq_array(const double def_value = 1.){
            array<dim> ret_val;
            ret_val.fill(def_value);
            return ret_val;
        };

        template<typename F, typename X>
        double derivand(double x, void* param){
            auto fptr = reinterpret_cast<std::function<F(X)>*>(param);
            X nx = static_cast<X>(x);
            F rv = (*fptr)(nx);
            return static_cast<double>(rv); 
        };

        std::pair<double, double> derive_full(func_1d& func, const double x, 
                double h = 1.e-8){
            const gsl_function gsl_func = 
                    {&derivand<double, double>, reinterpret_cast<void*>(&func)};
            double ret_val, err;
            gsl_deriv_central(&gsl_func, x, h, &ret_val, &err);
            return {ret_val, err};
        };

        double derive(func_1d& func, const double x, double h = 1.e-8){
            return derive_full(func, x, h).first;
        };

        template<std::size_t dim>
        array<dim> step_nd(const std::size_t d, 
                            const array<dim>& a, const array<dim>& b){
            array<dim> ret_val(a);
            ret_val.at(d) = b.at(d);
            return ret_val;
        };

        template<std::size_t dim, typename f_arg_type = const array<dim>&>
        array<dim> derive_nd(const func_nd<dim, f_arg_type>& func, const array<dim>& x,
                const double h = 1.e-6, array<dim> h_ratio = eq_array<dim>(1.)){
            const auto h_rat = array<dim>(h_ratio);
            const auto step = vector::mul(h_rat, h);
            constexpr int d = static_cast<int>(dim);
            const auto  x_plus = vect::sum<double, d>(x, step),
                        x_minus = vect::sub<double, d>(x, step);
            array<dim> ret_val;
            for(std::size_t i = 0; i < dim; i++){
                const auto  xp = step_nd<dim>(i, x, x_plus),
                            xm = step_nd<dim>(i, x, x_minus);
                const double fp = func(xp), fm = func(xm);
                ret_val.at(i) = 0.5 * (fp - fm) / step.at(i);
            }
            return ret_val;
        };

    };
};

#endif