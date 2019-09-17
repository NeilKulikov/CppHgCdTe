#ifndef COMP_MATRIX
#define COMP_MATRIX

#include <cmath>
#include <complex>
#include <algorithm>
#include <functional>

#include <matrix.hpp>
#include <integrate.hpp>

namespace operators{

    std::complex<double> plane_wave(int idx, double x){
        using namespace std::complex_literals;
        return std::exp(2i * M_PI * static_cast<double>(idx) * x);
    };

    template<   typename F = std::complex<double>, 
                typename X = double>
    class basis{
        private: 
            X sc;
            std::pair<int, int> bparams;
            std::function<F(int, X)> bfunc;
        public:
            basis(  std::function<F(int, X)> const & func = plane_wave,
                    std::pair<int, int> basis_params = {-50, 50},
                    X scale_coeff = 1.) :
                        sc(scale_coeff), bparams(basis_params),
                        bfunc(func){
                if(!(bparams.first < bparams.second))
                    throw std::logic_error("Params range should be positive");

            };
            std::function<F(X)> operator[] (const int idx) const {
                if(!((bparams.first <= idx) && (idx <= bparams.second)))
                    throw std::out_of_range("Params not in range");
                return [&](X x){ return bfunc(idx, x / sc) / std::sqrt(sc); };
            };
            int size(void) const {
                return bparams.second - bparams.first;
            };
            int from(void) const {
                return bparams.first;
            }
            int upto(void) const {
                return bparams.second;
            }
    };

    template<typename F = double, typename X = double>
    F matrix_elem(
        basis<F, X> const & bas, 
        std::function<F(X)> const & func, 
        int i, int j,
        gsl_integration_workspace* ws = nullptr,
        std::size_t max_eval = 16384){
            std::function<F(X)> igr = 
                [&](X x){ return std::conj(bas[i](x)) * func(x) * bas[j](x); };
            auto rv = staff::integrate_qag(igr, {0., 1.}, ws, max_eval);
            return rv.first;
    };

    matrix::herm herm_op_matr(
        basis<std::complex<double>, double> const & bas,
        std::function<std::complex<double>(double)> const & inp,
        std::size_t max_eval = 16384){
            matrix::cmat rv(bas.size());
            gsl_integration_workspace* ws = 
                gsl_integration_workspace_alloc(max_eval);
            for(int r = 0; r < bas.size(); r++){
                for(int c = r; c < bas.size(); c++){
                    int ri = bas.from() + r,
                        ci = bas.from() + c;
                    rv.at(
                        static_cast<std::size_t>(r), 
                        static_cast<std::size_t>(c)) = 
                        matrix_elem(bas, inp, ri, ci, ws, max_eval);
                }
            }
            gsl_integration_workspace_free(ws);
            return matrix::herm(rv);
    };

    matrix::herm herm_op_matr(
        basis<std::complex<double>, double> const & bas,
        std::function<double(double)> const & inp,
        std::size_t max_eval = 16384){
            using namespace std::complex_literals;
            std::function<std::complex<double>(double)> func = 
                [&](double x){ return inp(x) + 0i; };
            return herm_op_matr(bas, func, max_eval);
    };

    std::vector< std::complex<double> > fourier(
        std::function<std::complex<double>(double)> const & inp,
        const double len = 1.,
        const std::size_t bsize = 101,
        const std::size_t max_eval = 4096,
        const double eps_abs = 1.e-8,
        const double eps_rel = 1.e-3){
            using namespace std::placeholders;
            std::vector< std::complex<double> > rv(bsize);
            std::function<std::complex<double>(int, double)> func = 
                [&](int p, double x){ return plane_wave(p, x / len) * inp(x); };
            gsl_integration_cquad_workspace* ws = 
                gsl_integration_cquad_workspace_alloc(max_eval);
            /*gsl_integration_workspace* ws = 
                gsl_integration_workspace_alloc(max_eval);*/
            for(std::size_t i = 0; i < bsize; i++){
                std::function< std::complex<double>(double) > igr =
                    std::bind(func, static_cast<int>(i), _1);
                /*auto ires = staff::integrate_qag(
                        igr, 
                        {0., len}, 
                        ws, 
                        max_eval,
                        GSL_INTEG_GAUSS21,
                        eps_abs,
                        eps_rel);*/
                auto ires = staff::integrate_cquad(
                        igr, 
                        {0., len}, 
                        ws, 
                        max_eval,
                        eps_abs,
                        eps_rel);
                //auto ires = staff::integrate(igr, {0., len}, max_eval);
                rv[i] = ires.first / len;
            }
            gsl_integration_cquad_workspace_free(ws);
            /*gsl_integration_workspace_free(ws);*/
            return rv;
        };

    void op_fill(
        std::vector< std::complex<double> > const & cache,
        matrix::cmat& mat){
            int bsize = static_cast<int>(cache.size());
            for(int r = 0; r < bsize; r++){
                for(int c = r; c < bsize; c++){
                    int cds = c - r;
                    mat.at( static_cast<std::size_t>(r),
                            static_cast<std::size_t>(c)) =
                        cache[static_cast<std::size_t>(cds)];
                }
            }
        };

    matrix::herm pw_matr(
        std::function<std::complex<double>(double)> const & inp,
        const double len = 1.,
        std::pair<int, int> blims = {-50, 51},
        std::size_t max_eval = 16384,
        const double eps_abs = 0.,
        const double eps_rel = 1.e-2){
            const int bs = blims.second - blims.first;
            const std::size_t bsize = static_cast<std::size_t>(bs);
            if(bs <= 0)
                throw std::length_error("Length of basis should be > 1");
            matrix::cmat rv(bsize);
            auto cache = fourier(inp, len, bsize, max_eval, eps_abs, eps_rel);
            op_fill(cache, rv);
            return matrix::herm(rv);
    };

    matrix::herm pw_matr(
        std::function<double(double)> const & inp,
        double len = 1.,
        std::pair<int, int> blims = {-50, 51},
        std::size_t max_eval = 16384,
        const double eps_abs = 1.e-8,
        const double eps_rel = 1.e-4){
            using namespace std::complex_literals;
            std::function<std::complex<double>(double)> nfunc = 
                [&](double x){ return inp(x) + 0.i; };
            return pw_matr(nfunc, len, blims, max_eval, eps_abs, eps_rel);
    };
};

#endif