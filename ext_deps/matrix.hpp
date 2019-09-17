#ifndef MATRIX
#define MATRIX

#include <vector>
#include <complex>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <memory>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>
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

    void dcmat(gsl_matrix_complex* inp){
        if(inp != nullptr)
            gsl_matrix_complex_free(inp);
    };

    class herm;

    class cmat{
        friend herm;
        protected:
            std::size_t msiz = 0;
            std::shared_ptr<gsl_matrix_complex> matr = nullptr;
            std::shared_ptr<gsl_matrix_complex> alloc(std::size_t s) const {
                return std::shared_ptr<gsl_matrix_complex>
                    (gsl_matrix_complex_calloc(s, s), dcmat);
            };
            void force_assign(double* data){
                auto gb = new gsl_block_complex{ msiz, data };
                auto gm = new gsl_matrix_complex;
                *gm = *matr;
                gm->data = data;
                gm->block = gb;
                matr.~shared_ptr();
                matr = std::shared_ptr<gsl_matrix_complex>(gm);
            };
            gsl_matrix_complex* raw(){
                return matr.get();
            };
        public:
            gsl_matrix_complex const * raw_const() const {
                return matr.get();
            };
            cmat(std::size_t size = 1){
                msiz = size;
                matr = alloc(size);
            };
            cmat(std::size_t size, std::complex<double> fill_value) : cmat(size)
                { gsl_matrix_complex_set_all(matr.get(), to_gsl_complex(fill_value)); };
            cmat(cmat& inp) : msiz(inp.size()), matr(inp.matr){ };
            cmat(std::vector< std::complex<double> >& inp){
                msiz = static_cast<std::size_t>(sqrt(inp.size()));
                if(msiz * msiz != inp.size())
                    throw std::logic_error("Size of input vector should be square of integer number");
                matr = alloc(msiz);
                force_assign(reinterpret_cast<double*>(inp.data()));
            };
            static cmat diagonal(std::vector< std::complex<double> >& in){
                cmat rv(in.size(), 0.);
                auto g_inpv = gsl_vector_complex_view_array(
                    reinterpret_cast<double*>(in.data()), in.size());
                auto d_view = gsl_matrix_complex_diagonal(rv.raw());
                gsl_vector_complex_memcpy(&d_view.vector, &g_inpv.vector);
                return rv;
            };
            static cmat copy(cmat const & in){
                cmat rv(in.size(), 0.);
                gsl_matrix_complex_memcpy(rv.raw(), in.raw_const());
                return rv;
            };
            cmat(std::vector<double>& inp){
                std::size_t rsiz = inp.size() / 2;
                msiz = sqrt(rsiz);
                if(2 * msiz * msiz != inp.size())
                    throw std::logic_error("Size of input vector" 
                        " should be 2 square of integer number");
                matr = alloc(msiz);
                force_assign(inp.data());
            };
            gsl_complex& at_gsl(std::size_t const i, std::size_t const j){
                return *gsl_matrix_complex_ptr(matr.get(), i, j);
            };
            std::complex<double>& at(std::size_t const i, std::size_t const j){
                return reinterpret_cast<std::complex<double>&>(at_gsl(i,j));
            };
            std::complex<double>& at(const std::pair<std::size_t, std::size_t> ij){
                return at(ij.first, ij.second);
            };
            std::size_t size() const {
                return msiz;
            };
            static void gemm(  
                cmat const & a, 
                cmat const & b,
                cmat& c,
                const std::complex<double> alpha = {1., 0.}, 
                const std::complex<double> beta  = {0., 0.},
                CBLAS_TRANSPOSE_t tra = CblasNoTrans,
                CBLAS_TRANSPOSE_t trb = CblasNoTrans){
                    const auto  gsl_a = to_gsl_complex(alpha),
                                gsl_b = to_gsl_complex(beta);
                    gsl_blas_zgemm(
                        tra,
                        trb,
                        gsl_a,
                        a.raw_const(),
                        b.raw_const(),
                        gsl_b,
                        c.raw());
                };
            cmat operator*(cmat const & a) const {
                if(size() != a.size())
                    throw std::length_error("Invalid size of matrices");
                auto rv = cmat(size());
                gemm(*this, a, rv);
                return rv;
            };
            cmat operator+(cmat const & a) const {
                if(size() != a.size())
                    throw std::length_error("Invalid size of matrices");
                auto rv = cmat(size());
                gsl_matrix_complex_memcpy(rv.raw(), raw_const());
                gsl_matrix_complex_add(rv.raw(), a.raw_const());
                return rv;
            };
            cmat operator-(cmat const & a) const {
                if(size() != a.size())
                    throw std::length_error("Invalid size of matrices");
                auto rv = cmat(size());
                gsl_matrix_complex_memcpy(rv.raw(), raw_const());
                gsl_matrix_complex_sub(rv.raw(), a.raw_const());
                return rv;
            };
            cmat operator*(std::complex<double> const & a){
                auto rv = cmat(size());
                gsl_matrix_complex_memcpy(rv.raw(), raw_const());
                gsl_matrix_complex_scale(rv.raw(), 
                                        to_gsl_complex(a));
                return rv;
            };
            void print(void){
                for(std::size_t i = 0; i < msiz; i++){
                    for(std::size_t j = 0; j < msiz; j++){
                        std::cout << '(' << i << ',' << j 
                            << ") " << at(i, j) << '\t';
                    }
                    std::cout << std::endl;
                }
            };
    };

    class herm : public cmat{
        public:
            herm(cmat& inp, bool trust = false) : 
                cmat(inp){
                    if(!trust) make_herm();
                    if(!check())
                        throw std::logic_error("Not hermitian");
            };
            herm(herm& inp) : cmat(inp){};
            bool check(double atol = 1.e-4){
                bool rv = true;
                for(std::size_t r = 0; r < msiz; r++){
                    for(std::size_t c = r; c < msiz; c++){
                        auto dif = at(r, c) - std::conj(at(c, r));
                        rv &= (abs(dif)) < atol;
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
            std::vector<double> diagonalize(
                gsl_eigen_herm_workspace* ws = nullptr){
                    bool owner = (ws == nullptr);
                    ws = owner ? gsl_eigen_herm_alloc(size()) : ws;
                    std::vector<double> evals(size());
                    auto gsl_evals = 
                        gsl_vector_view_array(evals.data(), size());
                    gsl_eigen_herm(raw(), &gsl_evals.vector, ws);
                    if(owner)
                        gsl_eigen_herm_free(ws);
                    std::sort(evals.begin(), evals.end());
                    return evals;
            };
    };
};

#endif