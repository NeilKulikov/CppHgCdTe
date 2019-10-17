#ifndef VECTOR
#define VECTOR

#include <array>
#include <vector>
#include <complex>
#include <exception>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_complex.h>

#include <matrix.hpp>

namespace vector{
    std::vector< std::complex<double> > dot 
            (std::vector< std::complex<double> >& v, 
            matrix::cmat const & m){
        if(v.size() != m.size())
            throw std::length_error("Vector and matrix should have equal length");
        std::vector< std::complex<double> > rv(v.size(), {0., 0.});
        auto    icvv = gsl_vector_complex_view_array(
                    reinterpret_cast<double*>(v.data()), v.size()),
                rcvv = gsl_vector_complex_view_array(
                    reinterpret_cast<double*>(rv.data()), rv.size());
        gsl_blas_zgemv(
            CblasTrans,
            matrix::to_gsl_complex({1., 0.}),
            m.raw_const(),
            &(icvv.vector),
            matrix::to_gsl_complex({0., 0.}),
            &(rcvv.vector)
        );
        return rv;
    };

    std::vector< std::complex<double> > operator* 
            (std::vector< std::complex<double> >& v, 
            matrix::cmat const & m){
        return dot(v, m);
    };

    std::vector< std::complex<double> > dot  
            (matrix::cmat const & m,
            std::vector< std::complex<double> >& v){
        if(v.size() != m.size())
            throw std::length_error("Vector and matrix should have equal length");
        std::vector< std::complex<double> > rv(v.size(), {0., 0.});
        auto    icvv = gsl_vector_complex_view_array(
                    reinterpret_cast<double*>(v.data()), v.size()),
                rcvv = gsl_vector_complex_view_array(
                    reinterpret_cast<double*>(rv.data()), rv.size());
        gsl_blas_zgemv(
            CblasNoTrans,
            matrix::to_gsl_complex({1., 0.}),
            m.raw_const(),
            &(icvv.vector),
            matrix::to_gsl_complex({0., 0.}),
            &(rcvv.vector)
        );
        return rv;
    };

    std::vector< std::complex<double> > operator*  
            (matrix::cmat const & m,
            std::vector< std::complex<double> >& v){
        return dot(m, v);
    };

    std::complex<double> dot
            (std::vector< std::complex<double> >& a,
            std::vector< std::complex<double> >& b){
        if(a.size() != b.size())
            throw std::length_error("Vectors should have equal length");
        gsl_complex rv;
        const auto  acvv = gsl_vector_complex_view_array(
                            reinterpret_cast<double*>(a.data()), a.size()),
                    bcvv = gsl_vector_complex_view_array(
                            reinterpret_cast<double*>(b.data()), b.size());
        gsl_blas_zdotc(&(acvv.vector), &(bcvv.vector), &rv);
        return matrix::to_std_complex(rv);
    };

    std::complex<double> operator*  
            (std::vector< std::complex<double> >& a,
            std::vector< std::complex<double> >& b){
        return dot(a, b);
    };

    double norm(std::vector< std::complex<double> >& v){
        const auto cvv = gsl_vector_complex_view_array(
                            reinterpret_cast<double*>(v.data()), v.size());
        return gsl_blas_dznrm2(&(cvv.vector));
    };

    template<std::size_t size>
    std::array< std::complex<double>, size > dot 
            (std::array< std::complex<double>, size >& v, 
            matrix::cmat const & m){
        if(v.size() != m.size())
            throw std::length_error("Vector and matrix should have equal length");
        std::array< std::complex<double>, size > rv;
        auto    icvv = gsl_vector_complex_view_array(
                    reinterpret_cast<double*>(v.data()), v.size()),
                rcvv = gsl_vector_complex_view_array(
                    reinterpret_cast<double*>(rv.data()), rv.size());
        gsl_blas_zgemv(
            CblasTrans,
            matrix::to_gsl_complex({1., 0.}),
            m.raw_const(),
            &(icvv.vector),
            matrix::to_gsl_complex({0., 0.}),
            &(rcvv.vector)
        );
        return rv;
    };

    template<std::size_t size>
    std::array< std::complex<double>, size > operator* 
            (std::array< std::complex<double>, size >& v, 
            matrix::cmat const & m){
        return dot(v, m);
    };

    template<std::size_t size>
    std::array< std::complex<double>, size > dot  
            (matrix::cmat const & m,
            std::array< std::complex<double>, size >& v){
        if(v.size() != m.size())
            throw std::length_error("Vector and matrix should have equal length");
        std::array< std::complex<double>, size > rv;
        auto    icvv = gsl_vector_complex_view_array(
                    reinterpret_cast<double*>(v.data()), v.size()),
                rcvv = gsl_vector_complex_view_array(
                    reinterpret_cast<double*>(rv.data()), rv.size());
        gsl_blas_zgemv(
            CblasNoTrans,
            matrix::to_gsl_complex({1., 0.}),
            m.raw_const(),
            &(icvv.vector),
            matrix::to_gsl_complex({0., 0.}),
            &(rcvv.vector)
        );
        return rv;
    };

    template<std::size_t size>
    std::array< std::complex<double>, size > operator*  
            (matrix::cmat const & m,
            std::array< std::complex<double>, size >& v){
        return dot(m, v);
    };

    template<std::size_t size>
    std::complex<double> dot
            (std::array< std::complex<double>, size >& a,
            std::array< std::complex<double>, size >& b){
        if(a.size() != b.size())
            throw std::length_error("Vectors should have equal length");
        gsl_complex rv;
        const auto  acvv = gsl_vector_complex_view_array(
                            reinterpret_cast<double*>(a.data()), a.size()),
                    bcvv = gsl_vector_complex_view_array(
                            reinterpret_cast<double*>(b.data()), b.size());
        gsl_blas_zdotc(&(acvv.vector), &(bcvv.vector), &rv);
        return matrix::to_std_complex(rv);
    };

    template<std::size_t size>
    std::complex<double> operator*  
            (std::array< std::complex<double>, size >& a,
            std::array< std::complex<double>, size >& b){
        return dot(a, b);
    };

    template<std::size_t size>
    std::complex<double> dot
            (std::vector< std::complex<double>>& a,
            std::array< std::complex<double>, size >& b){
        if(a.size() != b.size())
            throw std::length_error("Vectors should have equal length");
        gsl_complex rv;
        const auto  acvv = gsl_vector_complex_view_array(
                            reinterpret_cast<double*>(a.data()), a.size()),
                    bcvv = gsl_vector_complex_view_array(
                            reinterpret_cast<double*>(b.data()), b.size());
        gsl_blas_zdotc(&(acvv.vector), &(bcvv.vector), &rv);
        return matrix::to_std_complex(rv);
    };

    template<std::size_t size>
    std::complex<double> dot
            (std::array< std::complex<double>, size >& a,
            std::vector< std::complex<double>>& b){
        return std::conj(dot(b, a));
    };

    template<std::size_t size>
    std::complex<double> operator*  
            (std::vector< std::complex<double>>& a,
            std::array< std::complex<double>, size >& b){
        return dot(a, b);
    };

    template<std::size_t size>
    std::complex<double> operator*
            (std::array< std::complex<double>, size >& a,
            std::vector< std::complex<double>>& b){
        return dot(a, b);
    };

    template<std::size_t size>
    double norm(std::array< std::complex<double>, size >& v){
        const auto cvv = gsl_vector_complex_view_array(
                            reinterpret_cast<double*>(v.data()), v.size());
        return gsl_blas_dznrm2(&(cvv.vector));
    };
};

#endif