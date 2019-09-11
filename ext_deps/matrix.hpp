#ifndef MATRIX
#define MATRIX

#include <vector>
#include <complex>
#include <algorithm>
#include <cstdlib>
#include <cstdio>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>

namespace matrix{
    template<typename T = double>
    gsl_complex to_gsl_complex(std::complex<T> const & in){
        return  {
                    static_cast<double>(in.real()), 
                    static_cast<double>(in.imag())
                };
    };

    template<typename T = double>
    std::complex<T> to_std_complex(gsl_complex const & in){
        return  {
                    static_cast<T>(GSL_REAL(in)), 
                    static_cast<T>(GSL_IMAG(in))
                };
    };

    template<>
    gsl_complex to_gsl_complex(std::complex<double> const & in){
        return reinterpret_cast<gsl_complex const &>(in);
    };

    template<>
    std::complex<double> to_std_complex(gsl_complex const & in){
        return reinterpret_cast<std::complex<double> const &>(in);
    };

    constexpr std::size_t sqrt(std::size_t in){
        if(in < 2){
            return in;
        }
        std::size_t smaller = sqrt(in >> 2) << 1,
                    greater = smaller + 1;
        return (greater * greater > in) ? smaller : greater;
    };

    class cmat{
        protected:
            bool owner = true;
            std::size_t msiz = 0;
            gsl_matrix_complex* matr= nullptr;
            void force_assign(double* data){
                owner = false;
                std::free(reinterpret_cast<void*>(matr->block->data));
                matr->block->data = data;
                matr->data = data;
            };
            gsl_matrix_complex* raw(){
                return matr;
            };
        public:
            cmat(std::size_t size){
                msiz = size;
                matr = gsl_matrix_complex_alloc(size, size);
            };
            cmat(std::size_t size, std::complex<double> fill_value){
                msiz = size;
                matr = gsl_matrix_complex_alloc(size, size);
                gsl_matrix_complex_set_all(matr, to_gsl_complex(fill_value));
            };
            cmat(cmat& inp) : owner(false), matr(inp.raw()), msiz(inp.size()){};
            cmat(std::vector< std::complex<double> >& inp){
                msiz = static_cast<std::size_t>(sqrt(inp.size()));
                if(msiz * msiz != inp.size())
                    throw std::logic_error("Size of input vector should be square of integer number");
                matr = gsl_matrix_complex_alloc(msiz, msiz);
                force_assign(reinterpret_cast<double*>(inp.data()));
            };
            cmat(std::vector<double>& inp){
                std::size_t rsiz = inp.size() / 2;
                msiz = sqrt(rsiz);
                if(2 * msiz * msiz != inp.size())
                    throw std::logic_error("Size of input vector" 
                        " should be 2 square of integer number");
                matr = gsl_matrix_complex_alloc(msiz, msiz);
                force_assign(inp.data());
            };
            gsl_complex& at_gsl(std::size_t const i, std::size_t const j){
                return *gsl_matrix_complex_ptr(matr, i, j);
            };
            std::complex<double>& at(std::size_t const i, std::size_t const j){
                return reinterpret_cast<std::complex<double>&>(at_gsl(i,j));
            };
            std::size_t size() const {
                return msiz;
            }; 
            ~cmat(void){
                if(owner) gsl_matrix_complex_free(matr);
            };
            static void gemm(  
                cmat& a, 
                cmat& b,
                cmat& c,
                std::complex<double> alpha = {1., 0.}, 
                std::complex<double> beta  = {0., 0.},
                CBLAS_TRANSPOSE_t tra = CblasNoTrans,
                CBLAS_TRANSPOSE_t trb = CblasNoTrans){
                    const auto  gsl_a = to_gsl_complex(alpha),
                                gsl_b = to_gsl_complex(beta);
                    gsl_blas_zgemm(
                        tra,
                        trb,
                        gsl_a,
                        a.raw(),
                        b.raw(),
                        gsl_b,
                        c.raw());
                };
            cmat operator*(cmat& a){
                if(size() != a.size())
                    throw std::length_error("Invalid size of matrices");
                auto rv = cmat(size(), {0., 0.});
                gemm(*this, a, rv);
                return rv;
            };
    };

    class herm : public cmat{
        public:
            herm(cmat& inp, bool trust = false) : 
                cmat(inp){
                    if(!trust) make_herm();
                    if(!check())
                        throw std::logic_error("Not hermetian");
            };
            herm(herm& inp) : cmat(inp){};
            bool check(double rtol = 1.e-4){
                bool rv = true;
                for(std::size_t r = 0; r < msiz; r++){
                    for(std::size_t c = r; c < msiz; c++){
                        auto dif = at(r, c) - std::conj(at(c, r));
                        rv &= (abs(dif) / abs(at(r, c))) < rtol;
                    }
                }
                return rv;
            };
            void make_herm(void){
                if(check()) return;
                for(std::size_t r = 0; r < msiz; r++){
                    at(r, r) = {at(r, r).real(), 0.};
                    for(std::size_t c = r + 1; c < msiz; c++){
                        at(c, r) = std::conj(at(r, c));
                    }
                }
            };
    };
};

#endif