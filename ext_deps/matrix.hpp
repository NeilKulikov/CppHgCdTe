#ifndef MATRIX
#define MATRIX

#include <vector>
#include <complex>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <memory>
#include <utility>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_complex_math.h>

namespace matrix{
    class cmat;
    class rmat;
    class herm;

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

    void drmat(gsl_matrix* inp){
        if(inp != nullptr)
            gsl_matrix_free(inp);
    };

    class rmat{
        protected:
            std::size_t msiz = 0;
            std::shared_ptr<gsl_matrix> matr = nullptr;
            std::shared_ptr<gsl_matrix> alloc(std::size_t s) const {
                return std::shared_ptr<gsl_matrix>
                    (gsl_matrix_calloc(s, s), drmat);
            };
            void force_assign(double* data){
                auto gb = new gsl_block{ msiz, data };
                auto gm = new gsl_matrix;
                *gm = *matr;
                gm->data = data;
                gm->block = gb;
                matr.~shared_ptr();
                matr = std::shared_ptr<gsl_matrix>(gm);
            };
            gsl_matrix* raw(){
                return matr.get();
            };
        public:
            gsl_matrix const * raw_const() const {
                return matr.get();
            };
            rmat(std::size_t size = 1){
                msiz = size;
                matr = alloc(size);
            };
            rmat(std::size_t size, double fill_value) : rmat(size)
                { gsl_matrix_set_all(matr.get(), fill_value); };
            rmat(rmat& inp) : msiz(inp.size()), matr(inp.matr){ };
            rmat(std::vector<double>& inp){
                msiz = static_cast<std::size_t>(sqrt(inp.size()));
                if(msiz * msiz != inp.size())
                    throw std::logic_error("Size of input vector should be square of integer number");
                matr = alloc(msiz);
                force_assign(inp.data());
            };
            rmat(const std::vector<double>& inp){
                msiz = static_cast<std::size_t>(sqrt(inp.size()));
                if(msiz * msiz != inp.size())
                    throw std::logic_error("Size of input vector" 
                    "should be square of integer number");
                const auto mat = gsl_matrix_const_view_array(
                    reinterpret_cast<const double*>(inp.data()), msiz, msiz);
                matr = alloc(msiz);
                gsl_matrix_memcpy(raw(), &(mat.matrix));
            };
            static rmat diagonal(std::vector<double>& in){
                rmat rv(in.size(), 0.);
                auto g_inpv = gsl_vector_view_array(
                    reinterpret_cast<double*>(in.data()), in.size());
                auto d_view = gsl_matrix_diagonal(rv.raw());
                gsl_vector_memcpy(&d_view.vector, &g_inpv.vector);
                return rv;
            };
            static rmat copy(rmat const & in){
                rmat rv(in.size(), 0.);
                gsl_matrix_memcpy(rv.raw(), in.raw_const());
                return rv;
            };
            double& at_gsl(std::size_t const i, std::size_t const j){
                return *gsl_matrix_ptr(matr.get(), i, j);
            };
            double& at(std::size_t const i, std::size_t const j){
                return at_gsl(i,j);
            };
            double& at(const std::pair<std::size_t, std::size_t> ij){
                return at(ij.first, ij.second);
            };
            std::size_t size() const {
                return msiz;
            };
            static void gemm(  
                rmat const & a, 
                rmat const & b,
                rmat& c,
                const double alpha = 1., 
                const double beta  = 0.,
                CBLAS_TRANSPOSE_t tra = CblasNoTrans,
                CBLAS_TRANSPOSE_t trb = CblasNoTrans){
                    if(a.size() != b.size())
                        throw std::length_error("Matrices should have equal size");
                    gsl_blas_dgemm(
                        tra,
                        trb,
                        alpha,
                        a.raw_const(),
                        b.raw_const(),
                        beta,
                        c.raw());
                };
            rmat dot(rmat const & a) const {
                if(size() != a.size())
                    throw std::length_error("Invalid size of matrices");
                auto rv = rmat(size());
                gemm(*this, a, rv);
                return rv;
            };
            rmat operator*(rmat const & a) const {
                return dot(a);
            };
            rmat operator+(rmat const & a) const {
                if(size() != a.size())
                    throw std::length_error("Invalid size of matrices");
                auto rv = rmat(size());
                gsl_matrix_memcpy(rv.raw(), raw_const());
                gsl_matrix_add(rv.raw(), a.raw_const());
                return rv;
            };
            rmat operator-(rmat const & a) const {
                if(size() != a.size())
                    throw std::length_error("Invalid size of matrices");
                auto rv = rmat(size());
                gsl_matrix_memcpy(rv.raw(), raw_const());
                gsl_matrix_sub(rv.raw(), a.raw_const());
                return rv;
            };
            rmat operator*(double const & a){
                auto rv = rmat(size());
                gsl_matrix_memcpy(rv.raw(), raw_const());
                gsl_matrix_scale(rv.raw(), a);
                return rv;
            };
            void print(std::ostream & ost = std::cout){
                for(std::size_t i = 0; i < msiz; i++){
                    for(std::size_t j = 0; j < msiz; j++){
                        ost << '_' << i << ',' << j << '_' << at(i, j) << "_\t";
                    }
                    std::cout << std::endl;
                }
            };
            void put_submatrix(rmat& inp,
                const std::pair<std::size_t, std::size_t> ij){
                    auto curv = gsl_matrix_submatrix(raw(), 
                        ij.first, ij.second, inp.size(), inp.size());
                    const auto inpv = inp.raw_const();
                    gsl_matrix_memcpy(&curv.matrix, inpv);
            };
            rmat inverse(gsl_permutation* perm = nullptr){
                bool owner = perm == nullptr;
                if(owner)
                    perm = gsl_permutation_alloc(size());
                auto cp = rmat::copy(*this);
                auto rv = rmat(size());
                int signum;
                gsl_linalg_LU_decomp(cp.raw(), perm, &signum);
                gsl_linalg_LU_invert(cp.raw_const(), perm, rv.raw());
                if(owner)
                    gsl_permutation_free(perm);
                return rv;
            };
            rmat transpose(void) const{
                rmat rv(size());
                gsl_matrix_transpose_memcpy(rv.raw(), raw_const());
                return rv;
            };
    };

    rmat dot(rmat const& a, rmat const& b){
        rmat rv(a.size());
        rmat::gemm(a, b, rv);
        return rv;
    };

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
            cmat(const std::vector< std::complex<double> >& inp){
                msiz = static_cast<std::size_t>(sqrt(inp.size()));
                if(msiz * msiz != inp.size())
                    throw std::logic_error("Size of input vector" 
                    "should be square of integer number");
                const auto mat = gsl_matrix_complex_const_view_array(
                    reinterpret_cast<const double*>(inp.data()), msiz, msiz);
                matr = alloc(msiz);
                gsl_matrix_complex_memcpy(raw(), &(mat.matrix));
            };
            static cmat diagonal(std::vector< std::complex<double> >& in){
                cmat rv(in.size(), 0.);
                auto g_inpv = gsl_vector_complex_view_array(
                    reinterpret_cast<double*>(in.data()), in.size());
                auto d_view = gsl_matrix_complex_diagonal(rv.raw());
                gsl_vector_complex_memcpy(&d_view.vector, &g_inpv.vector);
                return rv;
            };
            cmat(cmat const& in) : cmat(in.size()){
                gsl_matrix_complex_memcpy(raw(), in.raw_const());
            };
            static cmat copy(cmat const& in){
                return cmat(in);
            };
            static cmat real_copy(rmat& in){
                cmat rv(in.size());
                for(std::size_t i = 0; i < in.size(); i++){
                    for(std::size_t j = 0; j < in.size(); j++){
                        rv.at(i, j) = {in.at(i, j), 0.};
                    }
                }
                return rv;
            };
            static cmat imag_copy(rmat& in){
                cmat rv(in.size());
                for(std::size_t i = 0; i < in.size(); i++){
                    for(std::size_t j = 0; j < in.size(); j++){
                        rv.at(i, j) = {0., in.at(i, j)};
                    }
                }
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
                    if(a.size() != b.size())
                        throw std::length_error("Matrices should have equal size");
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
            cmat dot(cmat const & a) const {
                if(size() != a.size())
                    throw std::length_error("Invalid size of matrices");
                auto rv = cmat(size());
                gemm(*this, a, rv);
                return rv;
            };
            cmat operator*(cmat const & a) const {
                return dot(a);
            };
            cmat sum(cmat const & a) const {
                if(size() != a.size())
                    throw std::length_error("Invalid size of matrices");
                auto rv = cmat(size());
                gsl_matrix_complex_memcpy(rv.raw(), raw_const());
                gsl_matrix_complex_add(rv.raw(), a.raw_const());
                return rv;
            };
            cmat operator+ (cmat const & a) const {
                return sum(a);
            };
            cmat sub(cmat const & a) const {
                if(size() != a.size())
                    throw std::length_error("Invalid size of matrices");
                auto rv = cmat(size());
                gsl_matrix_complex_memcpy(rv.raw(), raw_const());
                gsl_matrix_complex_sub(rv.raw(), a.raw_const());
                return rv;
            };
            cmat operator- (cmat const & a) const {
                return sub(a);
            };
            cmat scale(std::complex<double> const & a){
                auto rv = cmat(size());
                gsl_matrix_complex_memcpy(rv.raw(), raw_const());
                gsl_matrix_complex_scale(rv.raw(), 
                                        to_gsl_complex(a));
                return rv;
            };
            cmat operator* (std::complex<double> const & a){
                return scale(a);
            };
            void print(std::ostream & ost = std::cout){
                for(std::size_t i = 0; i < msiz; i++){
                    for(std::size_t j = 0; j < msiz; j++){
                        ost << '_' << i << ',' << j << '_' << at(i, j) << "_\t";
                    }
                    std::cout << std::endl;
                }
            };
            void put_submatrix(cmat& inp,
                const std::pair<std::size_t, std::size_t> ij){
                    auto curv = gsl_matrix_complex_submatrix(raw(), 
                        ij.first, ij.second, inp.size(), inp.size());
                    const auto inpv = inp.raw_const();
                    gsl_matrix_complex_memcpy(&curv.matrix, inpv);
            };
            cmat inverse(gsl_permutation* perm = nullptr){
                bool owner = perm == nullptr;
                if(owner)
                    perm = gsl_permutation_alloc(size());
                auto cp = cmat::copy(*this);
                auto rv = cmat(size());
                int signum;
                gsl_linalg_complex_LU_decomp(cp.raw(), perm, &signum);
                gsl_linalg_complex_LU_invert(cp.raw_const(), perm, rv.raw());
                if(owner)
                    gsl_permutation_free(perm);
                return rv;
            };
            cmat transpose(void) const{
                cmat rv(size());
                gsl_matrix_complex_transpose_memcpy(rv.raw(), raw_const());
                return rv;
            };
            cmat conjugate(void) const{
                cmat rv = transpose();
                for(std::size_t i = 0; i < rv.size(); i++)
                    for(std::size_t j = 0; j < rv.size(); j++)
                        rv.at(i, j) = std::conj(rv.at(i, j));
                return rv;
            };
    };

    cmat dot(cmat const& a, cmat const& b){
        cmat rv(a.size());
        cmat::gemm(a, b, rv);
        return rv;
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
                    const bool owner = (ws == nullptr);
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
            std::pair< std::vector<double>, cmat> 
            diagonalize_v(
                gsl_eigen_hermv_workspace* ws = nullptr){
                    const bool owner = (ws == nullptr);
                    ws = owner ? gsl_eigen_hermv_alloc(size()) : ws;
                    std::vector<double> evals(size());
                    cmat evecs(size());
                    auto gsl_evals = 
                        gsl_vector_view_array(evals.data(), size());
                    gsl_eigen_hermv(raw(), &gsl_evals.vector, evecs.raw(), ws);
                    if(owner)
                        gsl_eigen_hermv_free(ws);
                    return {evals, evecs};
            };
    };
};

#endif