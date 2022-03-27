#include <immintrin.h>
#include "matmul.h"

// nxn @ nxd
void FloatCSRMulDenseRAW(float answer[], float data[], int indices[], int indptr[], float mat[], int mat_row, int mat_col)
{
    for (int i = 0; i < mat_row; i++)
    {
        int st = indptr[i], ed = indptr[i + 1];
        int pre_elements_for_ans = i * mat_col;

        for (int j = st; j < ed; j++)
        {
            int pre_elements_for_mat = indices[j] * mat_col;
            float coefficient = data[j];

            for (int k = 0; k < mat_col; k++)
                answer[pre_elements_for_ans + k] = answer[pre_elements_for_ans + k] + coefficient * mat[pre_elements_for_mat + k];
        }
    }
}

void FloatCSRMulDenseOMP(float answer[], float data[], int indices[], int indptr[], float mat[], int mat_row, int mat_col)
{
#pragma omp parallel for
    for (int i = 0; i < mat_row; i++)
    {
        int st = indptr[i], ed = indptr[i + 1];
        int pre_elements_for_ans = i * mat_col;

        for (int j = st; j < ed; j++)
        {
            int pre_elements_for_mat = indices[j] * mat_col;
            float coefficient = data[j];

            for (int k = 0; k < mat_col; k++)
                answer[pre_elements_for_ans + k] = answer[pre_elements_for_ans + k] + coefficient * mat[pre_elements_for_mat + k];
        }
    }
}

void FloatCSRMulDenseAVX128(float answer[], float data[], int indices[], int indptr[], float mat[], int mat_row, int mat_col)
{
    for (int i = 0; i < mat_row; i++)
    {
        int st = indptr[i], ed = indptr[i + 1];
        int pre_elements_for_ans = i * mat_col;

        for (int j = st; j < ed; j++)
        {
            int pre_elements_for_mat = indices[j] * mat_col;
            float coefficient = data[j];

            int k = 0;
            for (; k < (int)(mat_col / 4); k++)
            {
                float *dst_ans = answer + (pre_elements_for_ans + 4 * k);
                float *dst_mat = mat + (pre_elements_for_mat + 4 * k);

                __m128 float_128_vec_ans = _mm_loadu_ps(dst_ans);
                __m128 float_128_vec_mat = _mm_load_ps(dst_mat);
                __m128 float_128_vec_coeffi = _mm_set1_ps(coefficient);

                // float_128_vec_mat = _mm_mul_ps(float_128_vec_coeffi, float_128_vec_mat);
                // float_128_vec_ans = _mm_add_ps(float_128_vec_ans, float_128_vec_mat);

                float_128_vec_ans = _mm_fmadd_ps(float_128_vec_coeffi, float_128_vec_mat, float_128_vec_ans);

                _mm_storeu_ps(dst_ans, float_128_vec_ans);
            }

            k = k * 4;
            for (; k < mat_col; k++)
                answer[pre_elements_for_ans + k] = answer[pre_elements_for_ans + k] + coefficient * mat[pre_elements_for_mat + k];
        }
    }
}

void FloatCSRMulDenseAVX256(float answer[], float data[], int indices[], int indptr[], float mat[], int mat_row, int mat_col)
{
    for (int i = 0; i < mat_row; i++)
    {
        int st = indptr[i], ed = indptr[i + 1];
        int pre_elements_for_ans = i * mat_col;

        for (int j = st; j < ed; j++)
        {
            int pre_elements_for_mat = indices[j] * mat_col;
            float coefficient = data[j];

            int k = 0;
            for (; k < (int)(mat_col / 8); k++)
            {
                float *dst_ans = answer + (pre_elements_for_ans + 8 * k);
                float *dst_mat = mat + (pre_elements_for_mat + 8 * k);

                __m256 float_256_vec_ans = _mm256_loadu_ps(dst_ans);
                __m256 float_256_vec_mat = _mm256_loadu_ps(dst_mat);
                __m256 float_256_vec_coeffi = _mm256_set1_ps(coefficient);

                // float_256_vec_mat = _mm256_mul_ps(float_256_vec_coeffi, float_256_vec_mat);
                // float_256_vec_ans = _mm256_add_ps(float_256_vec_ans, float_256_vec_mat);

                float_256_vec_ans = _mm256_fmadd_ps(float_256_vec_coeffi, float_256_vec_mat, float_256_vec_ans);

                _mm256_storeu_ps(dst_ans, float_256_vec_ans);
            }

            k = k * 8;
            for (; k < mat_col; k++)
                answer[pre_elements_for_ans + k] = answer[pre_elements_for_ans + k] + coefficient * mat[pre_elements_for_mat + k];
        }
    }
}

void FloatCSRMulDenseAVX128OMP(float answer[], float data[], int indices[], int indptr[], float mat[], int mat_row, int mat_col)
{
#pragma omp parallel for
    for (int i = 0; i < mat_row; i++)
    {
        int st = indptr[i], ed = indptr[i + 1];
        int pre_elements_for_ans = i * mat_col;

        for (int j = st; j < ed; j++)
        {
            int pre_elements_for_mat = indices[j] * mat_col;
            float coefficient = data[j];

            int k = 0;
            for (; k < (int)(mat_col / 4); k++)
            {
                float *dst_ans = answer + (pre_elements_for_ans + 4 * k);
                float *dst_mat = mat + (pre_elements_for_mat + 4 * k);

                __m128 float_128_vec_ans = _mm_loadu_ps(dst_ans);
                __m128 float_128_vec_mat = _mm_load_ps(dst_mat);
                __m128 float_128_vec_coeffi = _mm_set1_ps(coefficient);

                // float_128_vec_mat = _mm_mul_ps(float_128_vec_coeffi, float_128_vec_mat);
                // float_128_vec_ans = _mm_add_ps(float_128_vec_ans, float_128_vec_mat);

                float_128_vec_ans = _mm_fmadd_ps(float_128_vec_coeffi, float_128_vec_mat, float_128_vec_ans);

                _mm_storeu_ps(dst_ans, float_128_vec_ans);
            }

            k = k * 4;
            for (; k < mat_col; k++)
                answer[pre_elements_for_ans + k] = answer[pre_elements_for_ans + k] + coefficient * mat[pre_elements_for_mat + k];
        }
    }
}

void FloatCSRMulDenseAVX256OMP(float answer[], float data[], int indices[], int indptr[], float mat[], int mat_row, int mat_col)
{
#pragma omp parallel for
    for (int i = 0; i < mat_row; i++)
    {
        int st = indptr[i], ed = indptr[i + 1];
        int pre_elements_for_ans = i * mat_col;

        for (int j = st; j < ed; j++)
        {
            int pre_elements_for_mat = indices[j] * mat_col;
            float coefficient = data[j];

            int k = 0;
            for (; k < (int)(mat_col / 8); k++)
            {
                float *dst_ans = answer + (pre_elements_for_ans + 8 * k);
                float *dst_mat = mat + (pre_elements_for_mat + 8 * k);

                __m256 float_256_vec_ans = _mm256_loadu_ps(dst_ans);
                __m256 float_256_vec_mat = _mm256_loadu_ps(dst_mat);
                __m256 float_256_vec_coeffi = _mm256_set1_ps(coefficient);

                // float_256_vec_mat = _mm256_mul_ps(float_256_vec_coeffi, float_256_vec_mat);
                // float_256_vec_ans = _mm256_add_ps(float_256_vec_ans, float_256_vec_mat);

                float_256_vec_ans = _mm256_fmadd_ps(float_256_vec_coeffi, float_256_vec_mat, float_256_vec_ans);

                _mm256_storeu_ps(dst_ans, float_256_vec_ans);
            }

            k = k * 8;
            for (; k < mat_col; k++)
                answer[pre_elements_for_ans + k] = answer[pre_elements_for_ans + k] + coefficient * mat[pre_elements_for_mat + k];
        }
    }
}

void DoubleCSRMulDenseRAW(double answer[], double data[], int indices[], int indptr[], double mat[], int mat_row, int mat_col)
{
    for (int i = 0; i < mat_row; i++)
    {
        int st = indptr[i], ed = indptr[i + 1];
        int pre_elements_for_ans = i * mat_col;

        for (int j = st; j < ed; j++)
        {
            int pre_elements_for_mat = indices[j] * mat_col;
            double coefficient = data[j];

            for (int k = 0; k < mat_col; k++)
                answer[pre_elements_for_ans + k] = answer[pre_elements_for_ans + k] + coefficient * mat[pre_elements_for_mat + k];
        }
    }
}

void DoubleCSRMulDenseOMP(double answer[], double data[], int indices[], int indptr[], double mat[], int mat_row, int mat_col)
{
#pragma omp parallel for
    for (int i = 0; i < mat_row; i++)
    {
        int st = indptr[i], ed = indptr[i + 1];
        int pre_elements_for_ans = i * mat_col;

        for (int j = st; j < ed; j++)
        {
            int pre_elements_for_mat = indices[j] * mat_col;
            double coefficient = data[j];

            for (int k = 0; k < mat_col; k++)
                answer[pre_elements_for_ans + k] = answer[pre_elements_for_ans + k] + coefficient * mat[pre_elements_for_mat + k];
        }
    }
}

void DoubleCSRMulDenseAVX128(double answer[], double data[], int indices[], int indptr[], double mat[], int mat_row, int mat_col)
{
    for (int i = 0; i < mat_row; i++)
    {
        int st = indptr[i], ed = indptr[i + 1];
        int pre_elements_for_ans = i * mat_col;

        for (int j = st; j < ed; j++)
        {
            int pre_elements_for_mat = indices[j] * mat_col;
            double coefficient = data[j];

            int k = 0;
            for (; k < (int)(mat_col / 2); k++)
            {
                double *dst_ans = answer + (pre_elements_for_ans + 2 * k);
                double *dst_mat = mat + (pre_elements_for_mat + 2 * k);

                __m128d double_128_vec_ans = _mm_loadu_pd(dst_ans);
                __m128d double_128_vec_mat = _mm_load_pd(dst_mat);
                __m128d double_128_vec_coeffi = _mm_set1_pd(coefficient);

                // double_128_vec_mat = _mm_mul_pd(double_128_vec_coeffi, double_128_vec_mat);
                // double_128_vec_ans = _mm_add_pd(double_128_vec_ans, double_128_vec_mat);

                double_128_vec_ans = _mm_fmadd_pd(double_128_vec_coeffi, double_128_vec_mat, double_128_vec_ans);

                _mm_storeu_pd(dst_ans, double_128_vec_ans);
            }

            k = k * 2;
            for (; k < mat_col; k++)
                answer[pre_elements_for_ans + k] = answer[pre_elements_for_ans + k] + coefficient * mat[pre_elements_for_mat + k];
        }
    }
}

void DoubleCSRMulDenseAVX256(double answer[], double data[], int indices[], int indptr[], double mat[], int mat_row, int mat_col)
{
    for (int i = 0; i < mat_row; i++)
    {
        int st = indptr[i], ed = indptr[i + 1];
        int pre_elements_for_ans = i * mat_col;

        for (int j = st; j < ed; j++)
        {
            int pre_elements_for_mat = indices[j] * mat_col;
            double coefficient = data[j];

            int k = 0;
            for (; k < (int)(mat_col / 8); k++)
            {
                double *dst_ans = answer + (pre_elements_for_ans + 8 * k);
                double *dst_mat = mat + (pre_elements_for_mat + 8 * k);

                __m256d double_256_vec_ans = _mm256_loadu_pd(dst_ans);
                __m256d double_256_vec_mat = _mm256_loadu_pd(dst_mat);
                __m256d double_256_vec_coeffi = _mm256_set1_pd(coefficient);

                // double_256_vec_mat = _mm256_mul_pd(double_256_vec_coeffi, double_256_vec_mat);
                // double_256_vec_ans = _mm256_add_pd(double_256_vec_ans, double_256_vec_mat);

                double_256_vec_ans = _mm256_fmadd_pd(double_256_vec_coeffi, double_256_vec_mat, double_256_vec_ans);

                _mm256_storeu_pd(dst_ans, double_256_vec_ans);
            }

            k = k * 4;
            for (; k < mat_col; k++)
                answer[pre_elements_for_ans + k] = answer[pre_elements_for_ans + k] + coefficient * mat[pre_elements_for_mat + k];
        }
    }
}

void DoubleCSRMulDenseAVX128OMP(double answer[], double data[], int indices[], int indptr[], double mat[], int mat_row, int mat_col)
{
#pragma omp parallel for
    for (int i = 0; i < mat_row; i++)
    {
        int st = indptr[i], ed = indptr[i + 1];
        int pre_elements_for_ans = i * mat_col;

        for (int j = st; j < ed; j++)
        {
            int pre_elements_for_mat = indices[j] * mat_col;
            double coefficient = data[j];

            int k = 0;
            for (; k < (int)(mat_col / 2); k++)
            {
                double *dst_ans = answer + (pre_elements_for_ans + 2 * k);
                double *dst_mat = mat + (pre_elements_for_mat + 2 * k);

                __m128d double_128_vec_ans = _mm_loadu_pd(dst_ans);
                __m128d double_128_vec_mat = _mm_load_pd(dst_mat);
                __m128d double_128_vec_coeffi = _mm_set1_pd(coefficient);

                // double_128_vec_mat = _mm_mul_pd(double_128_vec_coeffi, double_128_vec_mat);
                // double_128_vec_ans = _mm_add_pd(double_128_vec_ans, double_128_vec_mat);

                double_128_vec_ans = _mm_fmadd_pd(double_128_vec_coeffi, double_128_vec_mat, double_128_vec_ans);

                _mm_storeu_pd(dst_ans, double_128_vec_ans);
            }

            k = k * 2;
            for (; k < mat_col; k++)
                answer[pre_elements_for_ans + k] = answer[pre_elements_for_ans + k] + coefficient * mat[pre_elements_for_mat + k];
        }
    }
}

void DoubleCSRMulDenseAVX256OMP(double answer[], double data[], int indices[], int indptr[], double mat[], int mat_row, int mat_col)
{
#pragma omp parallel for
    for (int i = 0; i < mat_row; i++)
    {
        int st = indptr[i], ed = indptr[i + 1];
        int pre_elements_for_ans = i * mat_col;

        for (int j = st; j < ed; j++)
        {
            int pre_elements_for_mat = indices[j] * mat_col;
            double coefficient = data[j];

            int k = 0;
            for (; k < (int)(mat_col / 8); k++)
            {
                double *dst_ans = answer + (pre_elements_for_ans + 8 * k);
                double *dst_mat = mat + (pre_elements_for_mat + 8 * k);

                __m256d double_256_vec_ans = _mm256_loadu_pd(dst_ans);
                __m256d double_256_vec_mat = _mm256_loadu_pd(dst_mat);
                __m256d double_256_vec_coeffi = _mm256_set1_pd(coefficient);

                // double_256_vec_mat = _mm256_mul_pd(double_256_vec_coeffi, double_256_vec_mat);
                // double_256_vec_ans = _mm256_add_pd(double_256_vec_ans, double_256_vec_mat);

                double_256_vec_ans = _mm256_fmadd_pd(double_256_vec_coeffi, double_256_vec_mat, double_256_vec_ans);

                _mm256_storeu_pd(dst_ans, double_256_vec_ans);
            }

            k = k * 4;
            for (; k < mat_col; k++)
                answer[pre_elements_for_ans + k] = answer[pre_elements_for_ans + k] + coefficient * mat[pre_elements_for_mat + k];
        }
    }
}