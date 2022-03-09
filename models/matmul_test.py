import time
import numpy as np
import scipy.sparse as sp
import torch

sparse_mat = sp.rand(5000, 5000, density=0.1, format="csr", dtype=np.float32)
dense_mat = np.random.rand(5000, 1000).astype(np.float32)

# scipy
st = time.time()
ans = sparse_mat.dot(dense_mat)
print(f'Scipy: {(time.time() - st):.4f}s')
# print(ans)

# pytorch
data = sparse_mat.data
indices = sparse_mat.indices
indptr = sparse_mat.indptr
csr_matrix = torch.sparse_csr_tensor(indptr, indices, data, dtype=torch.float32)
dense_matrix = torch.from_numpy(dense_mat)

st = time.time()
ans = csr_matrix.matmul(dense_matrix)
print(f'PyTorch: {(time.time() - st):.4f}s')
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

answer = np.zeros((5000, 1000)).astype(np.float32).flatten()
data = sparse_mat.data
indices = sparse_mat.indices
indptr = sparse_mat.indptr
mat = dense_mat.flatten()
mat_row, mat_col = dense_mat.shape

st = time.time()
ctl_lib.FloatCSRMulDenseRAW(answer, data, indices, indptr, mat, mat_row, mat_col)
print(f'Raw C: {(time.time() - st):.4f}s')

# avx 256
ctl_lib.FloatCSRMulDenseAVX256.argtypes = [arr_1d_float, arr_1d_float, arr_1d_int, arr_1d_int, arr_1d_float, c_int,
                                           c_int]
ctl_lib.FloatCSRMulDenseAVX256.restypes = None

answer = np.zeros((5000, 1000)).astype(np.float32).flatten()
data = sparse_mat.data
indices = sparse_mat.indices
indptr = sparse_mat.indptr
mat = dense_mat.flatten()
mat_row, mat_col = dense_mat.shape

st = time.time()
ctl_lib.FloatCSRMulDenseAVX256(answer, data, indices, indptr, mat, mat_row, mat_col)
print(f'AVX 256 C: {(time.time() - st):.4f}s')

# avx 128
ctl_lib.FloatCSRMulDenseAVX128.argtypes = [arr_1d_float, arr_1d_float, arr_1d_int, arr_1d_int, arr_1d_float, c_int,
                                           c_int]
ctl_lib.FloatCSRMulDenseAVX128.restypes = None

answer = np.zeros((5000, 1000)).astype(np.float32).flatten()
data = sparse_mat.data
indices = sparse_mat.indices
indptr = sparse_mat.indptr
mat = dense_mat.flatten()
mat_row, mat_col = dense_mat.shape

st = time.time()
ctl_lib.FloatCSRMulDenseAVX128(answer, data, indices, indptr, mat, mat_row, mat_col)
print(f'AVX 128 C: {(time.time() - st):.4f}s')

# OpenMP
ctl_lib.FloatCSRMulDenseOMP.argtypes = [arr_1d_float, arr_1d_float, arr_1d_int, arr_1d_int, arr_1d_float, c_int, c_int]
ctl_lib.FloatCSRMulDenseOMP.restypes = None

answer = np.zeros((5000, 1000)).astype(np.float32).flatten()
data = sparse_mat.data
indices = sparse_mat.indices
indptr = sparse_mat.indptr
mat = dense_mat.flatten()
mat_row, mat_col = dense_mat.shape

st = time.time()
ctl_lib.FloatCSRMulDenseOMP(answer, data, indices, indptr, mat, mat_row, mat_col)
print(f'OpenMP C: {(time.time() - st):.4f}s')

# OpenMP + avx 256
ctl_lib.FloatCSRMulDenseAVX256OMP.argtypes = [arr_1d_float, arr_1d_float, arr_1d_int, arr_1d_int, arr_1d_float, c_int,
                                              c_int]
ctl_lib.FloatCSRMulDenseAVX256OMP.restypes = None

answer = np.zeros((5000, 1000)).astype(np.float32).flatten()
data = sparse_mat.data
indices = sparse_mat.indices
indptr = sparse_mat.indptr
mat = dense_mat.flatten()
mat_row, mat_col = dense_mat.shape

st = time.time()
ctl_lib.FloatCSRMulDenseAVX256OMP(answer, data, indices, indptr, mat, mat_row, mat_col)
print(f'OpenMP+AVX256 C: {(time.time() - st):.4f}s')

# OpenMP + avx 128
ctl_lib.FloatCSRMulDenseAVX128OMP.argtypes = [arr_1d_float, arr_1d_float, arr_1d_int, arr_1d_int, arr_1d_float, c_int,
                                              c_int]
ctl_lib.FloatCSRMulDenseAVX128OMP.restypes = None

answer = np.zeros((5000, 1000)).astype(np.float32).flatten()
data = sparse_mat.data
indices = sparse_mat.indices
indptr = sparse_mat.indptr
mat = dense_mat.flatten()
mat_row, mat_col = dense_mat.shape

st = time.time()
ctl_lib.FloatCSRMulDenseAVX128OMP(answer, data, indices, indptr, mat, mat_row, mat_col)
print(f'OpenMP+AVX128 C: {(time.time() - st):.4f}s')
