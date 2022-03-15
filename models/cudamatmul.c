#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <sys/time.h>         // clock

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

// nxn @ nxd
int FloatCSRMulDense(float answer[], int data_nnz, float data[], int indices[], int indptr[], float mat[], int mat_row, int mat_col)
{   
    int   A_num_rows      = mat_row;
    int   A_num_cols      = mat_row;
    int   A_nnz           = data_nnz;
    int   B_num_rows      = A_num_cols;
    int   B_num_cols      = mat_col;
    int   ldb             = B_num_rows;
    int   ldc             = A_num_rows;
    int   B_size          = ldb * B_num_cols;
    int   C_size          = ldc * B_num_cols;
    int   *hA_csrOffsets  = indptr;
    int   *hA_columns     = indices;
    float *hA_values      = data;
    float *hB             = mat;
    float *hC             = answer;

    float alpha           = 1.0f;
    float beta            = 0.0f;
    //--------------------------------------------------------------------------
    // Device memory management

    cudaStream_t stream1, stream2, stream3;
    CHECK_CUDA( cudaStreamCreate(&stream1) )
    CHECK_CUDA( cudaStreamCreate(&stream2) )
    CHECK_CUDA( cudaStreamCreate(&stream3) )

    int   *dA_csrOffsets, *dA_columns;
    float *dA_values, *dB, *dC;
    CHECK_CUDA( cudaHostAlloc((void**) &dA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int), cudaHostAllocMapped) )
    CHECK_CUDA( cudaHostAlloc((void**) &dA_columns, A_nnz * sizeof(int), cudaHostAllocMapped)    )
    CHECK_CUDA( cudaHostAlloc((void**) &dA_values,  A_nnz * sizeof(float), cudaHostAllocMapped)  )
    CHECK_CUDA( cudaHostAlloc((void**) &dB,         B_size * sizeof(float), cudaHostAllocMapped) )
    CHECK_CUDA( cudaHostAlloc((void**) &dC,         C_size * sizeof(float), cudaHostAllocMapped) )

    CHECK_CUDA( cudaMemcpyAsync(dA_csrOffsets, hA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice, stream1) )
    CHECK_CUDA( cudaMemcpyAsync(dA_columns, hA_columns, A_nnz * sizeof(int),
                           cudaMemcpyHostToDevice, stream1) )
    CHECK_CUDA( cudaMemcpyAsync(dA_values, hA_values, A_nnz * sizeof(float),
                           cudaMemcpyHostToDevice, stream2) )
    CHECK_CUDA( cudaMemcpyAsync(dB, hB, B_size * sizeof(float),
                           cudaMemcpyHostToDevice, stream2) )
    CHECK_CUDA( cudaMemcpyAsync(dC, hC, C_size * sizeof(float),
                           cudaMemcpyHostToDevice, stream2) )

    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    struct timeval startTime, endTime;
    gettimeofday(&startTime, 0);
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )

    gettimeofday(&endTime, 0);
    double timeuse = 1000000*(endTime.tv_sec - startTime.tv_sec) + endTime.tv_usec - startTime.tv_usec;
    printf("%lf\n", (double)(timeuse/1000));

    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, B_num_cols, dB,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, B_num_cols, dC,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )

    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_CSR_ALG2, &bufferSize) )
    CHECK_CUDA( cudaMallocHost(&dBuffer, bufferSize) )

    CHECK_CUSPARSE( cusparseSetStream(handle, stream3) )

    // execute SpMM
    CHECK_CUSPARSE( cusparseSpMM(handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                CUSPARSE_SPMM_CSR_ALG2, dBuffer) )

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )

    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpyAsync(hC, dC, C_size * sizeof(float),
                           cudaMemcpyDeviceToHost, stream3) )

    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFreeHost(dBuffer) )
    CHECK_CUDA( cudaFreeHost(dA_csrOffsets) )
    CHECK_CUDA( cudaFreeHost(dA_columns) )
    CHECK_CUDA( cudaFreeHost(dA_values) )
    CHECK_CUDA( cudaFreeHost(dB) )
    CHECK_CUDA( cudaFreeHost(dC) )

    CHECK_CUDA( cudaStreamDestroy(stream1) )
    CHECK_CUDA( cudaStreamDestroy(stream2) )
    CHECK_CUDA( cudaStreamDestroy(stream3) )

    return EXIT_SUCCESS;
}