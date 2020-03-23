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
                public:
                    typedef double gfunc_type (const gsl_vector*, void*);
                    static double gsl_func(const f_type& func, const gsl_vector* x, void* param){
                        const array arr = to_std_arr(x);
                        return func(arr);
                    };
                    std::function<gfunc_type> gfunc;
                    gfunc_type* get_gfunc(void){
                        gfunc_type** ret_val = gfunc.template target<gfunc_type*>();
                        if(ret_val == nullptr) throw std::domain_error("gfunc is null");
                        return *ret_val;
                    };
                    typedef void ggrad_type (const gsl_vector*, void*, gsl_vector*);
                    static void gsl_grad(const g_type& grad, const gsl_vector* x, void* param, gsl_vector* y){
                        const array arr = to_std_arr(x);
                        const auto gvec = grad(arr);
                        std::copy(gvec.cbegin(), gvec.cend(), &(y->data[0]));
                    };
                    std::function<ggrad_type> ggrad;
                    ggrad_type* get_ggrad(void){
                        ggrad_type** ret_val = gfunc.template target<ggrad_type*>();
                        if(ret_val == nullptr) throw std::domain_error("ggrad is null");
                        return *ret_val;
                    };
                    typedef void gfull_type (const gsl_vector*, void*, double*, gsl_vector*);
                    static void gsl_full(p_type fg, const gsl_vector* x, void* param, double* f, gsl_vector* y){
                        const array arr = to_std_arr(x);
                        *f = (fg.first)(arr);
                        const auto gvec = (fg.second)(arr);
                        std::copy(gvec.cbegin(), gvec.cend(), &(y->data[0]));
                    };
                    std::function<gfull_type> gfull;
                    gfull_type* get_gfull(void){
                        gfull_type** ret_val = gfull.template target<gfull_type*>();
                        if(ret_val == nullptr) throw std::domain_error("gfull is null");
                        return *ret_val;
                    };
                public:
                    gslf gen_gslf(void){
                        gslf ret_val{ 
                            get_gfunc(),
                            get_ggrad(),
                            get_gfull(),
                            dim,
                            nullptr};
                        return ret_val;
                    };
                public:
                    array def_gradient(const array& x, 
                            const double step = 1.e-8) const{
                        return _def_gradienta(func, x, step);
                    };
                    fdf_function(const f_type& _func, const g_type& _grad): 
                        func(_func), 
                        grad(_grad),
                        gfunc(std::bind(gsl_func, func, std::placeholders::_1, std::placeholders::_2)),
                        ggrad(std::bind(gsl_grad, grad, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3)),
                        gfull(std::bind(gsl_full, p_type(func, grad), std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4))
                        {
                            std::cout << "First stage" << std::endl;
                        };
                    fdf_function(const f_type& _func): 
                        fdf_function(
                            _func, 
                            std::bind(_def_gradient, func, std::placeholders::_1))
                        {
                            std::cout << "Second stage" << std::endl;
                        };
            };
            const m_type* type = gsl_multimin_fdfminimizer_conjugate_pr; 
        private:
            std::mutex in_use;
            gsl_multimin_fdfminimizer* minimizer = nullptr;
        public:
            fdf_function func;
        public:
            fdf_minimizer(const m_type* _type, const fdf_function _func):
                func(_func), minimizer(gsl_multimin_fdfminimizer_alloc(_type, dim)){
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
            minimize_result minimize(const array& x, 
                    minimize_props mp = minimize_props()){
                using flags = typename minimize_result::status_flag;
                std::lock_guard lock(in_use);
                const auto gsl_x = fdf_function::to_gsl_vec(x);
                std::cout << "Gen started" << std::endl;
                std::cout << (*func.get_gfunc())(&gsl_x, nullptr) << std::endl;
                std::cout << "Gen started" << std::endl;
                gslf gs = func.gen_gslf();
                std::cout << "Set started" << std::endl;
                std::cout << (*(gs.f))(&gsl_x, nullptr) << std::endl;
                //gsl_multimin_fdfminimizer_set(minimizer, &(func.body), &gsl_x, mp.tol, mp.step);
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