#ifndef HGCDTE_C
#define HGCDTE_C

#include <stdlib.h>

extern "C"{
    void* make_model(size_t n, double* zs, double* xs);
    void* make_hcore(void* model, size_t bsize);
    void* make_hinst(void* hcore, double kx, double ky);
    double* gen_eigen(void* hinst);
};
#endif