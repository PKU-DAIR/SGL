#ifndef __CUDAMATMUL_H__
#define __CUDAMATMUL_H__

void FloatCSRMulDense(float answer[], int data_nnz, float data[], int indices[], int indptr[], float mat[], int mat_row, int mat_col);

#endif