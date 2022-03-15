import time
import numpy as np
import scipy.sparse as sp
import torch


sparse_mat = sp.rand(10000, 10000, density=0.001, format="csr", dtype=np.float32)
dense_mat = np.random.rand(10000, 500).astype(np.float32)


# scipy
st = time.time()
ans = sparse_mat.dot(dense_mat)
print(f'Scipy: {(time.time()-st):.4f}s')
print(ans)

# pytorch
data = sparse_mat.data
indices = sparse_mat.indices
indptr = sparse_mat.indptr
csr_matrix = torch.sparse_csr_tensor(indptr, indices, data, dtype=torch.float32)
dense_matrix = torch.from_numpy(dense_mat)

st = time.time()
ans = torch.spmm(csr_matrix, dense_matrix)
print(f'PyTorch: {(time.time()-st):.4f}s')
# print(ans)


# raw c++
import numpy.ctypeslib as ctl
from ctypes import c_int
ctl_lib = ctl.load_library("libmatmul.so", "./")

arr_1d_int = ctl.ndpointer(
    dtype=np.int32,
    ndim=1,
    flags="CONTIGUOUS"
)

arr_1d_float = ctl.ndpointer(
    dtype=np.float32,
    ndim=1,
    flags="CONTIGUOUS"
)

ctl_lib.FloatCSRMulDenseRAW.argtypes = [arr_1d_float, arr_1d_float, arr_1d_int, arr_1d_int, arr_1d_float, c_int, c_int]
ctl_lib.FloatCSRMulDenseRAW.restypes = None

answer = np.zeros((10000, 500)).astype(np.float32).flatten()
data = sparse_mat.data
indices = sparse_mat.indices
indptr = sparse_mat.indptr
mat = dense_mat.flatten()
mat_row, mat_col = dense_mat.shape

st = time.time()
ctl_lib.FloatCSRMulDenseRAW(answer, data, indices, indptr, mat, mat_row, mat_col)
print(f'Raw C: {(time.time()-st):.4f}s')


# OpenMP
ctl_lib.FloatCSRMulDenseOMP.argtypes = [arr_1d_float, arr_1d_float, arr_1d_int, arr_1d_int, arr_1d_float, c_int, c_int]
ctl_lib.FloatCSRMulDenseOMP.restypes = None

answer = np.zeros((10000, 500)).astype(np.float32).flatten()
data = sparse_mat.data
indices = sparse_mat.indices
indptr = sparse_mat.indptr
mat = dense_mat.flatten()
mat_row, mat_col = dense_mat.shape

st = time.time()
ctl_lib.FloatCSRMulDenseOMP(answer, data, indices, indptr, mat, mat_row, mat_col)
print(f'OpenMP C: {(time.time()-st):.4f}s')


# cuSPARSE
ctl_lib = ctl.load_library("libcudamatmul.so", "./")
ctl_lib.FloatCSRMulDense.argtypes = [arr_1d_float, c_int, arr_1d_float, arr_1d_int, arr_1d_int, arr_1d_float, c_int, c_int]
ctl_lib.FloatCSRMulDense.restypes = c_int

answer = np.zeros((10000, 500)).astype(np.float32).flatten()
data = sparse_mat.data
data_nnz = len(data)
indices = sparse_mat.indices
indptr = sparse_mat.indptr
mat = dense_mat.flatten()
mat_row, mat_col = dense_mat.shape

st = time.time()
ctl_lib.FloatCSRMulDense(answer, data_nnz, data, indices, indptr, mat, mat_row, mat_col)
print(f'CUDA C: {(time.time()-st):.4f}s')
print(answer.reshape(dense_mat.shape))
