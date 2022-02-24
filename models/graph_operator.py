import scipy.sparse as sp

from models.base_op import GraphOp
from models.utils import sparse_mx_to_torch_sparse_tensor, adj_to_symmetric_norm


class LaplacianGraphOp(GraphOp):
    def __init__(self, prop_steps, r=0.5):
        super(LaplacianGraphOp, self).__init__(prop_steps)
        self.__r = r

    def _construct_adj(self, adj):
        if isinstance(adj, sp.csr_matrix):
            adj = adj.tocoo()
        elif not isinstance(adj, sp.coo_matrix):
            raise TypeError("The adjacency matrix must be scipy.sparse.coo_matrix/csr_matrix!")

        adj_normalized = adj_to_symmetric_norm(adj, self.__r)
        return sparse_mx_to_torch_sparse_tensor(adj_normalized)


class PprGraphOp(GraphOp):
    def __init__(self, prop_steps, r=0.5, alpha=0.15):
        super(PprGraphOp, self).__init__(prop_steps)
        self.__r = r
        self.__alpha = alpha

    def _construct_adj(self, adj):
        if isinstance(adj, sp.csr_matrix):
            adj = adj.tocoo()
        elif not isinstance(adj, sp.coo_matrix):
            raise TypeError("The adjacency matrix must be scipy.sparse.coo_matrix/csr_matrix!")

        adj_normalized = adj_to_symmetric_norm(adj, self.__r)
        adj_normalized = (1 - self.__alpha) * adj_normalized + self.__alpha * sp.eye(adj.shape[0])
        return sparse_mx_to_torch_sparse_tensor(adj_normalized)


class MotifGraphOp(GraphOp):
    pass