#include <iostream>
#include <iostream>

#include <gsl/gsl_matrix.h>

int main(void){
    double dat[] = { 0., 1., 2., 3.};
    gsl_matrix* mat = gsl_matrix_alloc(50, 50);
    std::cout << mat->data << '\t' << mat->block->data << std::endl;

    return 0;
};