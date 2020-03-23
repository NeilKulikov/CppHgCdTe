#ifndef HGCDTE_C
#define HGCDTE_C

#include <stdlib.h>

extern "C"{
    void* make_model(size_t n, double* zs, double* xs);
    int del_model(void* md);
    void* make_strain_model(size_t n, double* zs, double* xs, double bufx);
    void* make_strain_modela(size_t n, double* zs, double* xs, double bufx, size_t npoints);
    int del_strain_model(void* md);
    void* make_hcore(void* model, size_t bsize);
    void* make_hcorea(void* model, size_t bsize, double acc);
    int del_hcore(void* hc);
    void* make_rotator(double a, double b, double c);
    int del_rotator(void* rot);
    void* make_strain_hcore(void* model, size_t bsize);
    void* make_strain_hcorea(void* model, size_t bsize, double acc);
    int del_strain_hcore(void* hc);
    void* make_hinst(void* hcore, double kx, double ky);
    void* make_hinstr(void* hcore, double kx, double ky, void* rot);
    void* make_strain_hinst(void* hcore);
    void* make_strain_hinst_full(size_t n, double* zs, double* xs, double bufx);
    int del_hinst(void* hc);
    double* gen_eigen(void* hinst);
    double* gen_eigena(void* hinst, void* ws);
    double* get_matr(void* matr);
    int del_diag_ws(void* ws);
    void* get_diag_ws(size_t size);
    void* sum_hinsts(void* a, void* b);
    double* get_serial(void* hcore, void* strh, size_t n, double* ks);
};
#endif