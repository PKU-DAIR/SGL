#ifndef __MATMUL_H__
#define __MATMUL_H__

void FloatCSRMulDenseRAW(float answer[], float data[], int indices[], int indptr[], float mat[], int mat_row, int mat_col);
void FloatCSRMulDenseOMP(float answer[], float data[], int indices[], int indptr[], float mat[], int mat_row, int mat_col);
void FloatCSRMulDenseAVX256(float answer[], float data[], int indices[], int indptr[], float mat[], int mat_row, int mat_col);
void FloatCSRMulDenseAVX128(float answer[], float data[], int indices[], int indptr[], float mat[], int mat_row, int mat_col);
void FloatCSRMulDenseAVX256OMP(float answer[], float data[], int indices[], int indptr[], float mat[], int mat_row, int mat_col);
void FloatCSRMulDenseAVX128OMP(float answer[], float data[], int indices[], int indptr[], float mat[], int mat_row, int mat_col);

void DoubleCSRMulDenseRAW(double answer[], double data[], int indices[], int indptr[], double mat[], int mat_row, int mat_col);
void DoubleCSRMulDenseOMP(double answer[], double data[], int indices[], int indptr[], double mat[], int mat_row, int mat_col);
void DoubleCSRMulDenseAVX256(double answer[], double data[], int indices[], int indptr[], double mat[], int mat_row, int mat_col);
void DoubleCSRMulDenseAVX128(double answer[], double data[], int indices[], int indptr[], double mat[], int mat_row, int mat_col);
void DoubleCSRMulDenseAVX256OMP(double answer[], double data[], int indices[], int indptr[], double mat[], int mat_row, int mat_col);
void DoubleCSRMulDenseAVX128OMP(double answer[], double data[], int indices[], int indptr[], double mat[], int mat_row, int mat_col);

#endif