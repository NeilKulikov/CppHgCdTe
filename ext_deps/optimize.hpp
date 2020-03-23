#ifndef OPTIMIZE
#define OPTIMIZE

#include <mutex>
#include <algorithm>
#include <exception>
#include <functional>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>

#include "./vector.hpp"
#include "./derivate.hpp"


namespace staff{
    template<std::size_t dim = 2, typename f_arg_type = const std::array<double, dim>&>
    struct fdf_minimizer{
        using gslf = gsl_multimin_function_fdf;
        using array = std::array<double, dim>;
        using m_type = gsl_multimin_fdfminimizer_type;
        using f_type = std::function<double(f_arg_type)>;
        using g_type = std::function<array(f_arg_type)>;
        public:
            struct fdf_function{
                public:
                    const f_type func;
                    const g_type grad;
                private:
                    static array _def_gradienta(const f_type& _func, const array& x, 
                            const double step = 1.e-8){
                        return derive::derive_nd<dim>(_func, x, step);
                    };
                    static array _def_gradient(const f_type& _func, const array& x){
                        return fdf_function::_def_gradienta(_func, x);
                    };
                public:
                    static gsl_vector to_gsl_vec(const array& x){
                        const auto vw = gsl_vector_const_view_array(x.data(), dim);
                        return vw.vector;
                    };
                    static gsl_vector to_gsl_vec(array& x){
                        auto vw = gsl_vector_view_array(x.data(), dim);
                        return vw.vector;
                    };
                    static array to_std_arr(const gsl_vector* x){
                        array ret_val;
                        std::copy(&(x->data[0]), &(x->data[dim]), ret_val.begin());
                        return ret_val;
                    };
                    static array to_std_arr(gsl_vector* x){
                        return to_std_arr(reinterpret_cast<const gsl_vector*>(x));
                    };
                private:
                    static double gsl_func(const gsl_vector* x, void* param){
                        const auto th = 
                                reinterpret_cast<const fdf_function*>(param);
                        const array arr = to_std_arr(x);
                        const double res = (th->func)(arr);
                        return res;
                    };
                    static void gsl_grad(const gsl_vector* x, void* param, gsl_vector* y){
                        const auto th = 
                                reinterpret_cast<const fdf_function*>(param);
                        const array arr = to_std_arr(x);
                        const auto gvec = (th->grad)(arr);
                        std::copy(gvec.cbegin(), gvec.cend(), &(y->data[0]));
                    };
                    static void gsl_full(const gsl_vector* x, void* param, double* f, gsl_vector* y){
                        const auto th = 
                                reinterpret_cast<const fdf_function*>(param);
                        const array arr = to_std_arr(x);
                        f[0] = (th->func)(arr);
                        const auto gvec = (th->grad)(arr);
                        std::copy(gvec.cbegin(), gvec.cend(), &(y->data[0]));
                    };
                public:
                    gslf gen_gslf(void){
                        gslf ret_val{ 
                            &fdf_function::gsl_func,
                            &fdf_function::gsl_grad,
                            &fdf_function::gsl_full,
                            dim,
                            reinterpret_cast<void*>(this)};
                        return ret_val;
                    };
                public:
                    array def_gradient(const array& x, 
                            const double step = 1.e-8) const{
                        return _def_gradienta(func, x, step);
                    };
                    fdf_function(const f_type& _func, const g_type& _grad): 
                        func(_func), 
                        grad(_grad)
                        {
                            //std::cout << "Zero stage" << std::endl;
                        };
                    fdf_function(const f_type& _func): 
                        func(_func), 
                        grad(std::bind(_def_gradient, func, std::placeholders::_1))
                        {
                            //std::cout << "S1" << std::endl;
                        };
            };
            const m_type* type = nullptr; 
        private:
            std::mutex in_use;
            gsl_multimin_fdfminimizer* minimizer = nullptr;
        private:
            fdf_function func;
        public:
            fdf_minimizer(const fdf_function& _func,
                    const m_type* _type = gsl_multimin_fdfminimizer_conjugate_pr):
                type(_type),
                minimizer(gsl_multimin_fdfminimizer_alloc(type, dim)),
                func(_func)
                {
                    //std::cout << "2-nd stage" << std::endl;
                };
            fdf_minimizer(const f_type& _func,
                    const m_type* _type = gsl_multimin_fdfminimizer_conjugate_pr):
                fdf_minimizer(fdf_function(_func), _type)
                {
                    //std::cout << "3-rd stage" << std::endl;
                };
            ~fdf_minimizer(){
                gsl_multimin_fdfminimizer_free(minimizer);
            };
            struct minimize_props{
                const double grad_stop = 1.e-4;
                const double tol = 1.e-4;
                const double step = 1.e-7;
                const std::size_t max_steps = 10000;
            };
            struct minimize_result{
                enum status_flag{
                    ERROR,
                    SUCCESS,
                    TOO_MANY_STEPS
                };
                status_flag status = status_flag::ERROR;
                double min;
                array x;
            };
            minimize_result minimize(const array& x, 
                    minimize_props mp = minimize_props()){
                using flags = typename minimize_result::status_flag;
                std::lock_guard lock(in_use);
                const auto gsl_x = fdf_function::to_gsl_vec(x);
                gslf gs = func.gen_gslf();
                gsl_multimin_fdfminimizer_set(minimizer, &gs, &gsl_x, mp.tol, mp.step);
                int status = GSL_CONTINUE;
                minimize_result res;
                for(std::size_t i = 0; i < mp.max_steps && status == GSL_CONTINUE; i++){
                    status = gsl_multimin_fdfminimizer_iterate(minimizer);
                    if(status) break;
                    status = gsl_multimin_test_gradient(minimizer->gradient, mp.grad_stop);
                    if(status == GSL_SUCCESS) break;
                }
                res = (status == GSL_SUCCESS) ? 
                        minimize_result{
                            flags::SUCCESS, 
                            gsl_multimin_fdfminimizer_minimum(minimizer), 
                            fdf_function::to_std_arr(
                                gsl_multimin_fdfminimizer_x(minimizer))
                        }:
                        minimize_result{
                            flags::ERROR, 
                            0., 
                            array(x)
                        };
                gsl_multimin_fdfminimizer_restart(minimizer);
                return res;
            };
    };
};

#endif