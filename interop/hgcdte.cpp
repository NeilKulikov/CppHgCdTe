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

void* make_hcore(void* model, size_t bsize, double accuracy){
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
    return make_hcore(model, bsize, 1.e-7);
};
    
void* make_hinst(void* hcore, double kx, double ky){
    auto hc = reinterpret_cast<hamiltonian::hcore*>(hcore);
    std::pair<double, double> momentum{kx, ky};
    auto hf = hc->full_h(momentum);
    auto rv = new matrix::herm(hf);
    return reinterpret_cast<void*>(rv);
};

int del_hinst(void* hi){
    auto hp = reinterpret_cast<matrix::herm*>(hi);
    delete hp;
    return 0;
};

double* gen_eigen(void* hinst){
    auto hf = reinterpret_cast<matrix::herm*>(hinst);
    auto es = hf->diagonalize(); 
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



