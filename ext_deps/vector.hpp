#ifndef VECTOR
#define VECTOR

#include <array>
#include <vector>
#include <complex>
#include <exception>
#include <algorithm>
#include <functional>

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

    std::vector<double> dot 
            (std::vector<double>& v, 
            matrix::rmat const & m){
        if(v.size() != m.size())
            throw std::length_error("Vector and matrix should have equal length");
        std::vector<double> rv(v.size(), 0.);
        auto    icvv = gsl_vector_view_array(
                    reinterpret_cast<double*>(v.data()), v.size()),
                rcvv = gsl_vector_view_array(
                    reinterpret_cast<double*>(rv.data()), rv.size());
        gsl_blas_dgemv(
            CblasTrans,
            1.,
            m.raw_const(),
            &(icvv.vector),
            0.,
            &(rcvv.vector)
        );
        return rv;
    };

    std::vector<double> operator* 
            (std::vector<double>& v, 
            matrix::rmat const & m){
        return dot(v, m);
    };

    std::vector<double> dot  
            (matrix::rmat const & m,
            std::vector<double>& v){
        if(v.size() != m.size())
            throw std::length_error("Vector and matrix should have equal length");
        std::vector<double> rv(v.size(), 0.);
        auto    icvv = gsl_vector_view_array(
                    reinterpret_cast<double*>(v.data()), v.size()),
                rcvv = gsl_vector_view_array(
                    reinterpret_cast<double*>(rv.data()), rv.size());
        gsl_blas_dgemv(
            CblasNoTrans,
            1.,
            m.raw_const(),
            &(icvv.vector),
            0.,
            &(rcvv.vector)
        );
        return rv;
    };

    std::vector<double> operator*  
            (matrix::rmat const & m,
            std::vector<double>& v){
        return dot(m, v);
    };

    double dot
            (std::vector<double>& a,
            std::vector<double>& b){
        if(a.size() != b.size())
            throw std::length_error("Vectors should have equal length");
        double rv;
        const auto  acvv = gsl_vector_view_array(
                            reinterpret_cast<double*>(a.data()), a.size()),
                    bcvv = gsl_vector_view_array(
                            reinterpret_cast<double*>(b.data()), b.size());
        gsl_blas_ddot(&(acvv.vector), &(bcvv.vector), &rv);
        return rv;
    };

    double operator*  
            (std::vector<double>& a,
            std::vector<double>& b){
        return dot(a, b);
    };

    double norm(std::vector<double>& v){
        const auto cvv = gsl_vector_view_array(
                            reinterpret_cast<double*>(v.data()), v.size());
        return gsl_blas_dnrm2(&(cvv.vector));
    };

    template<std::size_t size>
    std::array<double, size > dot 
            (std::array<double, size >& v, 
            matrix::rmat const & m){
        if(v.size() != m.size())
            throw std::length_error("Vector and matrix should have equal length");
        std::array<double, size > rv;
        auto    icvv = gsl_vector_view_array(
                    reinterpret_cast<double*>(v.data()), v.size()),
                rcvv = gsl_vector_view_array(
                    reinterpret_cast<double*>(rv.data()), rv.size());
        gsl_blas_dgemv(
            CblasTrans,
            1.,
            m.raw_const(),
            &(icvv.vector),
            1.,
            &(rcvv.vector)
        );
        return rv;
    };

    template<std::size_t size>
    std::array<double, size > operator* 
            (std::array<double, size >& v, 
            matrix::rmat const & m){
        return dot(v, m);
    };

    template<std::size_t size>
    std::array<double, size > dot  
            (matrix::rmat const & m,
            std::array<double, size >& v){
        if(v.size() != m.size())
            throw std::length_error("Vector and matrix should have equal length");
        std::array<double, size > rv;
        auto    icvv = gsl_vector_view_array(
                    reinterpret_cast<double*>(v.data()), v.size()),
                rcvv = gsl_vector_view_array(
                    reinterpret_cast<double*>(rv.data()), rv.size());
        gsl_blas_dgemv(
            CblasNoTrans,
            1.,
            m.raw_const(),
            &(icvv.vector),
            0.,
            &(rcvv.vector)
        );
        return rv;
    };

    template<std::size_t size>
    std::array<double, size > operator*  
            (matrix::rmat const & m,
            std::array<double, size >& v){
        return dot(m, v);
    };

    template<std::size_t size>
    double dot
            (std::array<double, size >& a,
            std::array<double, size >& b){
        if(a.size() != b.size())
            throw std::length_error("Vectors should have equal length");
        double rv;
        const auto  acvv = gsl_vector_view_array(
                            reinterpret_cast<double*>(a.data()), a.size()),
                    bcvv = gsl_vector_view_array(
                            reinterpret_cast<double*>(b.data()), b.size());
        gsl_blas_ddot(&(acvv.vector), &(bcvv.vector), &rv);
        return rv;
    };

    template<std::size_t size>
    double operator*  
            (std::array<double, size >& a,
            std::array<double, size >& b){
        return dot(a, b);
    };

    template<std::size_t size>
    double dot
            (std::vector<double>& a,
            std::array<double, size >& b){
        if(a.size() != b.size())
            throw std::length_error("Vectors should have equal length");
        double rv;
        const auto  acvv = gsl_vector_view_array(
                            reinterpret_cast<double*>(a.data()), a.size()),
                    bcvv = gsl_vector_view_array(
                            reinterpret_cast<double*>(b.data()), b.size());
        gsl_blas_ddot(&(acvv.vector), &(bcvv.vector), &rv);
        return rv;
    };

    template<std::size_t size>
    double norm(std::array<double, size >& v){
        const auto cvv = gsl_vector_view_array(
                            reinterpret_cast<double*>(v.data()), v.size());
        return gsl_blas_dnrm2(&(cvv.vector));
    };

    std::vector<double> mul(
        std::vector<double> const& vec, 
        const double num){
            auto rv = vec;
            auto vv = gsl_vector_view_array(
                            reinterpret_cast<double*>(rv.data()), rv.size());
            gsl_blas_dscal(num, &(vv.vector));
            return rv;
    };

    std::vector<double> operator* (
        std::vector<double> const& vec, 
        const double num){
            return mul(vec, num);
    }; 
    std::vector<double> operator* ( 
        const double num,
        std::vector<double> const& vec){
            return mul(vec, num);
    };

    std::vector< std::complex<double> > mul(
        std::vector< std::complex<double> > const& vec, 
        const std::complex<double> num){
            auto rv = vec;
            auto vv = gsl_vector_complex_view_array(
                            reinterpret_cast<double*>(rv.data()), rv.size());
            gsl_blas_zscal(matrix::to_gsl_complex(num), &(vv.vector));
            return rv;
    };

    std::vector< std::complex<double> > operator* (
        std::vector< std::complex<double> > const& vec, 
        const std::complex<double> num){
            return mul(vec, num);
    }; 
    std::vector< std::complex<double> > operator* ( 
        const std::complex<double> num,
        std::vector< std::complex<double> > const& vec){
            return mul(vec, num);
    };

    std::vector< std::complex<double> > real_copy(
        std::vector<double> const& vec){
            std::vector< std::complex<double> > rv(vec.size());
            std::transform(
                vec.cbegin(),
                vec.cend(),
                rv.begin(),
                [](double x){ return std::complex<double>{x, 0.}; }
            );
            return rv;
    };

    std::vector< std::complex<double> > imag_copy(
        std::vector<double> const& vec){
            std::vector< std::complex<double> > rv(vec.size());
            std::transform(
                vec.cbegin(),
                vec.cend(),
                rv.begin(),
                [](double x){ return std::complex<double>{0., x}; }
            );
            return rv;
    };

    template<std::size_t size>
    std::array<double, size> mul(
        std::array<double, size> const& vec, 
        const double num){
            auto rv = vec;
            auto vv = gsl_vector_view_array(
                            reinterpret_cast<double*>(rv.data()), rv.size());
            gsl_blas_dscal(num, &(vv.vector));
            return rv;
    };

    template<std::size_t size>
    std::array<double, size> operator* (
        std::array<double, size> const& vec, 
        const double num){
            return mul(vec, num);
    }; 
    template<std::size_t size>
    std::array<double, size> operator* ( 
        const double num,
        std::array<double, size> const& vec){
            return mul(vec, num);
    };

    template<std::size_t size>
    std::array< std::complex<double>, size > mul(
        std::array< std::complex<double>, size > const& vec, 
        const std::complex<double> num){
            auto rv = vec;
            auto vv = gsl_vector_complex_view_array(
                            reinterpret_cast<double*>(rv.data()), rv.size());
            gsl_blas_zscal(matrix::to_gsl_complex(num), &(vv.vector));
            return rv;
    };

    template<std::size_t size>
    std::array< std::complex<double>, size > operator* (
        std::array< std::complex<double>, size > const& vec, 
        const std::complex<double> num){
            return mul(vec, num);
    }; 
    template<std::size_t size>
    std::array< std::complex<double>, size > operator* ( 
        const std::complex<double> num,
        std::array< std::complex<double>, size > const& vec){
            return mul(vec, num);
    };

    template<std::size_t size>
    std::array< std::complex<double>, size > real_copy(
        std::array<double, size> const& vec){
            std::array< std::complex<double>, size > rv;
            std::transform(
                vec.cbegin(),
                vec.cend(),
                rv.begin(),
                [](double x){ return std::complex<double>{x, 0.}; }
            );
            return rv;
    };

    template<std::size_t size>
    std::array< std::complex<double>, size > imag_copy(
        std::array<double, size> const& vec){
            std::array< std::complex<double>, size > rv;
            std::transform(
                vec.cbegin(),
                vec.cend(),
                rv.begin(),
                [](double x){ return std::complex<double>{0., x}; }
            );
            return rv;
    };
};

namespace vect{
        template<typename T, int dim>
        using vec = std::array<T, dim>;
        template<typename T, int dim>
        using vec_it = typename std::array<T, dim>::iterator;
        template<typename T, int dim>
        using vec_cit = typename std::array<T, dim>::const_iterator;
        template<typename T, int dim>
        using vec_range = std::pair< vec_it<T, dim>, vec_it<T, dim> >;
        template<typename T, int dim>
        using vec_crange = std::pair< vec_cit<T, dim>, vec_cit<T, dim> >;

        template<typename T, int dim>
        vec_range<T, dim> get_range(vec<T, dim>& arg){
            return {arg.begin(), arg.end()};
        };
        template<typename T, int dim>
        vec_crange<T, dim> get_crange(const vec<T, dim>& arg){
            return {arg.cbegin(), arg.cend()};
        };

        template<typename T>
        T asum(const T a, const T b){ return a + b; };
        template<typename T>
        T asub(const T a, const T b){ return a - b; };
        template<typename T>
        T amul(const T a, const T b){ return a * b; };
        template<typename T>
        T arat(const T a, const T b){ return a / b; };

        template<typename T>
        using atomic_op = const std::function<T(const T, const T)>;

        template<typename T, int dim>
        vec<T, dim> range_op(atomic_op<T>& func, const vec_crange<T, dim>& a, const vec_crange<T, dim>& b){
            vec<T, dim> out;
            std::transform(a.first, a.second, b.first, out.begin(), func);
            return out;
        };

        template<typename T, int dim>
        vec<T, dim> vec_op(atomic_op<T>& func, const vec<T, dim>& a, const vec<T, dim>& b){
            return range_op<T, dim>(func, get_crange<T, dim>(a), get_crange<T, dim>(b));
        };

        template<typename T, int dim>
        using vec_func = std::function<vec<T,dim>(const vec<T,dim>&, const vec<T,dim>&)>;
        template<typename T, int dim>
        using range_func = std::function<vec<T,dim>(const vec_crange<T,dim>&, const vec_crange<T,dim>&)>;

        template<typename T, int dim>
        vec_func<T, dim> vectorize(atomic_op<T>& func){
            return std::bind(vec_op<T, dim>, func, std::placeholders::_1, std::placeholders::_2);
        };

        template<typename T, int dim>
        range_func<T, dim> vectorize_range(atomic_op<T>& func){
            return std::bind(range_op<T, dim>, func, std::placeholders::_1, std::placeholders::_2);
        };

        template<typename T, int dim>
        const auto sum = vectorize<T, dim>(asum<T>);
        template<typename T, int dim>
        const auto sub = vectorize<T, dim>(asub<T>);
        template<typename T, int dim>
        const auto mul = vectorize<T, dim>(amul<T>);
        template<typename T, int dim>
        const auto rat = vectorize<T, dim>(arat<T>);

        template<typename T, int dim>
        const auto sum_range = vectorize_range<T, dim>(asum<T>);
        template<typename T, int dim>
        const auto sub_range = vectorize_range<T, dim>(asub<T>);
        template<typename T, int dim>
        const auto mul_range = vectorize_range<T, dim>(amul<T>);
        template<typename T, int dim>
        const auto rat_range = vectorize_range<T, dim>(arat<T>);
};

#endif