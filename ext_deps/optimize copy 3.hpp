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
        using p_type = std::pair<const f_type&, const g_type&>;
        public:
            struct fdf_function{
                struct fg{
                    f_type func;
                    g_type grad;
                };
                /*
                const f_type func;
                const g_type grad;
                */
                public:
                    const fg fg_pair;
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
                /*
                private:
                    static double gsl_func(const gsl_vector* x, void* param){
                        const fdf_function* th = 
                                reinterpret_cast<const fdf_function*>(param);
                        const array arr = to_std_arr(x);
                        return (th->func)(arr);
                    };
                    static void gsl_grad(const gsl_vector* x, void* param, gsl_vector* y){
                        const fdf_function* th = 
                                reinterpret_cast<const fdf_function*>(param);
                        const array arr = to_std_arr(x);
                        const auto gvec = (th->grad)(arr);
                        std::copy(gvec.cbegin(), gvec.cend(), &(y->data[0]));
                    };
                    static void gsl_full(const gsl_vector* x, void* param, double* f, gsl_vector* y){
                        const fdf_function* th = 
                                reinterpret_cast<const fdf_function*>(param);
                        const array arr = to_std_arr(x);
                        f[0] = (th->func)(arr);
                        const auto gvec = (th->grad)(arr);
                        std::copy(gvec.cbegin(), gvec.cend(), &(y->data[0]));
                    };
                public:
                    gslf gen_gslf(void){
                        std::cout << "Gen gslf" << std::endl;
                        gslf ret_val{ 
                            &fdf_function::gsl_func,
                            &fdf_function::gsl_grad,
                            &fdf_function::gsl_full,
                            dim,
                            reinterpret_cast<void*>(this)};
                        return ret_val;
                    };
                */
                public:
                    array def_gradient(const array& x, 
                            const double step = 1.e-8) const{
                        return _def_gradienta(fg_pair.func, x, step);
                        //return _def_gradienta(func, x, step);
                    };
                    fdf_function(const f_type& _func, const g_type& _grad): 
                        //func(_func), grad(_grad)
                        fg_pair(fg{_func, _grad})
                        {
                            std::cout << "S0" << std::endl;
                        };
                    fdf_function(const f_type& _func): 
                        fdf_function(
                            _func, 
                            //std::bind(_def_gradient, func, std::placeholders::_1)
                            std::bind(_def_gradient, fg_pair.func, std::placeholders::_1)
                        )
                        {
                            std::cout << "S1" << std::endl;
                        };
            };
            const m_type* type = gsl_multimin_fdfminimizer_conjugate_pr; 
        private:
            std::mutex in_use;
            gsl_multimin_fdfminimizer* minimizer = nullptr;
        public:
            fdf_function func;
        public:
            fdf_minimizer(const m_type* _type, fdf_function _func):
                type(_type),
                minimizer(gsl_multimin_fdfminimizer_alloc(type, dim)), 
                func(_func){
                    std::cout << "Third stage" << std::endl;
                };
            fdf_minimizer(const m_type* _type, const f_type& _func):
                fdf_minimizer(_type, fdf_function(_func)){
                    std::cout << "Fourth stage" << std::endl;
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
                double min_value;
                array x;
            };
            /*minimize_result minimize(const array& x, 
                    minimize_props mp = minimize_props()){
                using flags = typename minimize_result::status_flag;
                //std::lock_guard lock(in_use);
                std::cout << "M-1" << std::endl;
                const auto gsl_x = fdf_function::to_gsl_vec(x);
                std::cout << "M0" << std::endl;
                //gslf gs = func.gen_gslf();
                std::cout << "M1" << std::endl;
                //gsl_multimin_fdfminimizer_set(minimizer, &gs, &gsl_x, mp.tol, mp.step);
                std::cout << "Set finished" << std::endl;
                int status = GSL_CONTINUE;
                for(std::size_t i = 0; i < mp.max_steps && status == GSL_CONTINUE; i++){
                    status = gsl_multimin_fdfminimizer_iterate(minimizer);
                    if(status) break;
                    status = gsl_multimin_test_gradient(minimizer->gradient, mp.grad_stop);
                    if(status == GSL_SUCCESS) return {flags::SUCCESS, 0., array(x)};
                }
                return {flags::ERROR, 0., array(x)};
            };*/
    };
};

#endif