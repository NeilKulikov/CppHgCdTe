#include <vector>

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

void* make_hcore(void* model, size_t bsize, double accuracy){
    auto md = reinterpret_cast<materials::heterostruct*>(model);
    auto rv = new hamiltonian::hcore((*md), bsize, accuracy);
    return reinterpret_cast<void*>(rv);
};

void* make_hcore(void* model, size_t bsize){
    return make_hcore(model, bsize, 1.e-5);
};
    
void* make_hinst(void* hcore, double kx, double ky){
    auto hc = reinterpret_cast<hamiltonian::hcore*>(hcore);
    std::pair<double, double> momentum{kx, ky};
    auto hf = hc->full_h(momentum);
    auto rv = new matrix::herm(hf);
    return reinterpret_cast<void*>(rv);
};

double* gen_eigen(void* hinst){
    auto hf = reinterpret_cast<matrix::herm*>(hinst);
    auto es = hf->diagonalize(); 
    auto rv = new double[hf->size()];
    for(size_t i = 0; i < es.size(); i++) rv[i] = es[i];
    return rv;
};


