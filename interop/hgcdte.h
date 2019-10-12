#ifndef HGCDTE_C
#define HGCDTE_C

#include <stdlib.h>

extern "C"{
    void* make_model(size_t n, double* zs, double* xs);
    int del_model(void* md);
    void* make_hcore(void* model, size_t bsize);
    void* make_hcorea(void* model, size_t bsize, double acc);
    int del_hcore(void* hc);
    void* make_hinst(void* hcore, double kx, double ky);
    int del_hinst(void* hc);
    double* gen_eigen(void* hinst);
    double* gen_eigena(void* hinst, void* ws);
    double* get_matr(void* matr);
    int del_diag_ws(void* ws);
    void* get_diag_ws(size_t size);
    int make_model_c(void** rv, size_t n, double* zs, double* xs);
    int make_hcore_c(void** rv, void* model, size_t bsize);
    int make_hinst_c(void** rv, void* hcore, double kx, double ky);
    int gen_eigen_c(double** rv, void* hinst);
};
#endif