#ifndef MATRIX
#define MATRIX

#include <vector>
#include <complex>
#include <algorithm>

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
            std::size_t msiz = 0;
            gsl_matrix_complex* matr= nullptr;
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
            /*cmat(std::vector< std::complex<double> >& inp){
                msiz = static_cast<std::size_t>(sqrt(inp.size()));
                if(msiz * msiz != inp.size())
                    throw std::logic_error("Size of input vector should be square of integer number");
                auto gmv = gsl_matrix_complex_view_array(
                    reinterpret_cast<double*>(inp.data()), msiz, msiz);
                matr = &gmv.matrix;
            };*/
            cmat(std::vector<double>& inp){
                std::size_t rsiz = inp.size() / 2;
                msiz = static_cast<std::size_t>(sqrt(rsiz));
                if(2 * msiz * msiz != inp.size())
                    throw std::logic_error("Size of input vector should be 2 square of integer number");
                auto gmv = gsl_matrix_complex_view_array(
                    inp.data(), msiz, msiz);
                matr = &gmv.matrix;
            };
            gsl_complex& at_gsl(std::size_t const i, std::size_t const j){
                return *gsl_matrix_complex_ptr(matr, i, j);
            };
            std::complex<double>& at(std::size_t const i, std::size_t const j){
                return reinterpret_cast<std::complex<double>&>(at_gsl(i,j));
            };
            ~cmat(void){
                gsl_matrix_complex_free(matr);
            };
    };

    class herm : public cmat{
        public:
            bool check(double rtol = 1.e-4){
                bool rv = true;
                for(std::size_t i = 0; i < msiz; i++){
                    for(std::size_t j = i; j < msiz; j++){
                        auto dif = at(i, j) - std::conj(at(j, i));
                        rv &= (abs(dif) / abs(at(i, j))) < rtol;
                    }
                }
                return rv;
            };
            void make_herm(void){
                for(std::size_t i = 0; i < msiz; i++){
                    for(std::size_t j = i; j < msiz; j++){
                        at(j, i) = std::conj(at(i, j));
                    }
                }
            };
    };
};

#endif