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
                return [&](X x){ return bfunc(idx, sc * x) / sc; };
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
        std::size_t max_eval = 4096){
            std::function<F(X)> igr = 
                [&](X x){ return std::conj(bas[i](x)) * func(x) * bas[j](x); };
            auto rv = staff::integrate_qag(igr, {0., 1.}, ws, max_eval);
            return rv.first;
    };

    matrix::herm herm_op_matr(
        basis<std::complex<double>, double> const & bas,
        std::function<std::complex<double>(double)> const & inp,
        std::size_t max_eval = 4096){
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
        std::size_t max_eval = 4096){
            using namespace std::complex_literals;
            std::function<std::complex<double>(double)> func = 
                [&](double x){ return inp(x) + 0i; };
            return herm_op_matr(bas, func, max_eval);
    };

    matrix::herm pw_matr(
        std::function<std::complex<double>(double)> const & inp,
        double len = 1.,
        std::pair<int, int> blims = {-50, 50},
        std::size_t max_eval = 4096){
            using namespace std::placeholders;
            int bs = blims.second - blims.first;
            if(bs <= 0)
                throw std::length_error("Length of basis should be > 1");
            std::size_t bsize = static_cast<std::size_t>(bs);
            matrix::cmat rv(bsize);
            gsl_integration_workspace* ws = 
                gsl_integration_workspace_alloc(max_eval);
            std::vector< std::complex<double> > mcache(bsize);
            std::function<std::complex<double>(int, double)> func = 
                [&](int p, double x){ return plane_wave(p, x) * inp(x); };
            for(std::size_t i = 0; i < bsize; i++){
                std::function< std::complex<double>(double) > igr =
                    std::bind(func, static_cast<int>(i), _1);
                mcache[i] = staff::integrate_qag(
                        igr, 
                        {0., len}, 
                        ws, 
                        max_eval).first; 
            }
            for(int r = 0; r < bs; r++){
                for(int c = r; c < bs; c++){
                    int cds = c - r;
                    rv.at(  static_cast<std::size_t>(r),
                            static_cast<std::size_t>(c)) =
                        mcache[static_cast<std::size_t>(cds)];

                }
            }
            gsl_integration_workspace_free(ws);
            return matrix::herm(rv);
    };

    matrix::herm pw_matr(
        std::function<double(double)> const & inp,
        double len = 1.,
        std::pair<int, int> blims = {-50, 50},
        std::size_t max_eval = 4096){
            using namespace std::complex_literals;
            std::function<std::complex<double>(double)> nfunc = 
                [&](double x){ return inp(x) + 0i; };
            return pw_matr(nfunc, len, blims, max_eval);
    };
};

#endif