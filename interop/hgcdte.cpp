#include <vector>
#include <fstream>
#include <iostream>
#include <exception>

#include <cstring>

#include <model.hpp>
#include <matrix.hpp>
#include <hamiltonian.hpp>

#include "hgcdte.h"


void* make_model(size_t n, double* zs, double* xs){
    std::vector<double> z(zs, zs + n), 
                        x(xs, xs + n);
    auto rv = new materials::heterostruct(z, x);
    return reinterpret_cast<void*>(rv);
};

int del_model(void* md){
    auto mp = reinterpret_cast<materials::heterostruct*>(md);
    delete mp;
    return 0;
};

void* make_hcorea(void* model, size_t bsize, double accuracy){
    auto md = reinterpret_cast<materials::heterostruct*>(model);
    auto rv = new hamiltonian::hcore((*md), bsize, accuracy);
    return reinterpret_cast<void*>(rv);
};

int del_hcore(void* hc){
    auto hp = reinterpret_cast<hamiltonian::hcore*>(hc);
    delete hp;
    return 0;
};

void* make_hcore(void* model, size_t bsize){
    return make_hcorea(model, bsize, 1.e-7);
};
    
void* make_hinst(void* hcore, double kx, double ky){
    auto hc = reinterpret_cast<hamiltonian::hcore*>(hcore);
    std::pair<double, double> momentum{kx, ky};
    auto hf = hc->full_h(momentum);
    auto rv = new matrix::herm(hf);
    return reinterpret_cast<void*>(rv);
};

void* get_diag_ws(size_t size){
    auto rv = gsl_eigen_herm_alloc(size);
    return reinterpret_cast<void*>(rv);
};

int del_diag_ws(void* ws){
    auto wp = reinterpret_cast<gsl_eigen_herm_workspace*>(ws);
    gsl_eigen_herm_free(wp);
    return 0;
};

int del_hinst(void* hi){
    auto hp = reinterpret_cast<matrix::herm*>(hi);
    delete hp;
    return 0;
};

double* gen_eigen(void* hinst){
    return gen_eigena(hinst, nullptr);
};

double* gen_eigena(void* hinst, void* ws){
    auto hf = reinterpret_cast<matrix::herm*>(hinst);
    auto wp = reinterpret_cast<gsl_eigen_herm_workspace*>(ws);
    auto es = hf->diagonalize(wp); 
    auto rv = new double[hf->size()];
    for(size_t i = 0; i < es.size(); i++) rv[i] = es[i];
    return rv;
};

void print_hinst(void* hinst, char* name){
    std::fstream fs;
    fs.open(name, std::fstream::out);
    auto hi = reinterpret_cast<matrix::herm*>(hinst);
    hi->print(fs);
    fs.close();
};

double* get_matr(void* matr){
    auto hi = reinterpret_cast<matrix::cmat*>(matr);
    const auto size = hi->size();
    auto rv = new std::complex<double>[size * size];
    for(std::size_t r = 0; r < size; r++){
        for(std::size_t c = 0; c < size; c++)
            rv[r * size + c] = hi->at(r, c);
    }
    return reinterpret_cast<double*>(rv);
};

int make_model_c(void** rv, size_t n, double* zs, double* xs){
    try{
        *rv = make_model(n, zs, xs);
        return 0;
    }catch(std::exception& ex){
        std::cout << ex.what() << std::endl;
        return 1;
    };
};
int make_hcore_c(void** rv, void* model, size_t bsize){
    try{
        *rv = make_hcore(model, bsize);
        return 0;
    }catch(std::exception& ex){
        std::cout << ex.what() << std::endl;
        return 1;
    };
};
int make_hinst_c(void** rv, void* hcore, double kx, double ky){
    try{
        *rv = make_hinst(hcore, kx, ky);
        return 0;
    }catch(std::exception& ex){
        std::cout << ex.what() << std::endl;
        return 1;
    };
};
int gen_eigen_c(double** rv, void* hinst){
    try{
        *rv = gen_eigen(hinst);
        return 0;
    }catch(std::exception& ex){
        std::cout << ex.what() << std::endl;
        return 1;
    };
};



