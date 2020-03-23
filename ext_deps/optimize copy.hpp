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
                public:
                    const f_type func;
                    const g_type grad;
                private:
                    std::shared_ptr<p_type> params = nullptr;
                public:
                    gslf body;
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
                        std::cout << "GSL Full 0" << std::endl;
                        const auto* fg = reinterpret_cast<p_type*>(param);
                        std::cout << "GSL Full 0" << std::endl;
                        const auto& fu = fg->first;
                        std::cout << "GSL Full 1" << std::endl;
                        const array arr = to_std_arr(x);
                        std::cout << "GSL Full 2" << std::endl;
                        const double res = fu(arr);
                        std::cout << "GSL Full 3" << std::endl;
                        return res;
                    };
                    static void gsl_grad(const gsl_vector* x, void* param, gsl_vector* y){
                        std::cout << "GSL Full" << std::endl; 
                        const auto* fg = reinterpret_cast<p_type*>(param);
                        const auto grad = (fg->second)(to_std_arr(x));
                        std::copy(grad.cbegin(), grad.cend(), &(y->data[0]));
                    };
                    static void gsl_full(const gsl_vector* x, void* param, double* f, gsl_vector* y){
                        std::cout << "GSL Full" << std::endl;
                        const auto* fg = reinterpret_cast<p_type*>(param);
                        const auto arr = to_std_arr(x);
                        *f = (fg->first)(arr);
                        const auto grad = (fg->second)(arr);
                        std::copy(grad.cbegin(), grad.cend(), &(y->data[0]));
                    };
                    static gslf _gen_gslf(p_type* p){
                        gslf ret_val;
                        ret_val.n = dim;
                        ret_val.f = &gsl_func;
                        ret_val.df = &gsl_grad;
                        ret_val.fdf = &gsl_full;
                        ret_val.params = reinterpret_cast<void*>(p);
                        return ret_val;
                    };
                    static std::shared_ptr<p_type> _gen_params(const f_type& f, const g_type& g){
                        const auto ret_val = std::make_shared<p_type>(f, g);
                        return ret_val;
                    };
                public:
                    array def_gradient(const array& x, 
                            const double step = 1.e-8) const{
                        return _def_gradienta(func, x, step);
                    };
                    fdf_function(const f_type& _func, const g_type& _grad): 
                        func(_func), grad(_grad){
                            params = _gen_params(func, grad);
                            body = _gen_gslf(params.get());};
                    fdf_function(const f_type& _func): 
                        func(_func), 
                        grad(std::bind(_def_gradient, func, std::placeholders::_1)),
                        params(_gen_params(func, grad)),
                        body(_gen_gslf(params.get())){
                            std::cout << "Body " << std::endl;
                            const auto& fu = reinterpret_cast<p_type*>(body.params)->first;
                            std::cout << "Body " << std::endl;
                            std::cout << fu({0.5, 0.5}) << std::endl;
                            std::cout << "Body " << std::endl;
                        };
            };
            const m_type* type = gsl_multimin_fdfminimizer_conjugate_pr; 
        private:
            std::mutex in_use;
            gsl_multimin_fdfminimizer* minimizer = nullptr;
        public:
            const fdf_function func;
        public:
            fdf_minimizer(const m_type* _type, const fdf_function _func):
                func(_func), minimizer(gsl_multimin_fdfminimizer_alloc(_type, dim)){};
            fdf_minimizer(const m_type* _type, const f_type& _func):
                fdf_minimizer(_type, fdf_function(_func)){};
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
            minimize_result minimize(const array& x, 
                    minimize_props mp = minimize_props()){
                using flags = typename minimize_result::status_flag;
                std::lock_guard lock(in_use);
                const auto gsl_x = fdf_function::to_gsl_vec(x);
                auto op_vec = gsl_vector_alloc(dim);
                gslf gsl_func = func.body;
                std::cout << "Pre Direct started" << std::endl;
                const auto* p = reinterpret_cast<p_type*>(func.body.params);
                std::cout << "PPD" << std::endl;
                const f_type& f = p->first;
                const array a{0.75, 0.75};
                std::cout << "PD" << std::endl;
                const auto r = f(a);
                std::cout << "D" << std::endl;
                std::cout << r << std::endl;
                std::cout << "Direct started" << std::endl;
                std::cout << (*func.body.f)(&gsl_x, func.body.params) << std::endl;
                std::cout << "Set started" << std::endl;
                gsl_multimin_fdfminimizer_set(minimizer, &gsl_func, &gsl_x, mp.tol, mp.step);
                std::cout << "Set finished" << std::endl;
                int status = GSL_CONTINUE;
                for(std::size_t i = 0; i < mp.max_steps && status == GSL_CONTINUE; i++){
                    status = gsl_multimin_fdfminimizer_iterate(minimizer);
                    if(status) break;
                    status = gsl_multimin_test_gradient(minimizer->gradient, mp.grad_stop);
                    if(status == GSL_SUCCESS) return {flags::SUCCESS, 0., array(x)};
                }
                return {flags::ERROR, 0., array(x)};
            };
    };
};

#endif