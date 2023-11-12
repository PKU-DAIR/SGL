import scipy.sparse as sp

from sgl.operators.base_op import GraphOp
from sgl.operators.utils import adj_to_symmetric_norm


class LaplacianGraphOp(GraphOp):
    def __init__(self, prop_steps=-1, r=0.5, add_self_loops=True):
        super(LaplacianGraphOp, self).__init__(prop_steps)
        self.__r = r
        self.__add_self_loops = add_self_loops

    def _construct_adj(self, adj):
        if isinstance(adj, sp.csr_matrix):
            adj = adj.tocoo()
        elif not isinstance(adj, sp.coo_matrix):
            raise TypeError("The adjacency matrix must be a scipy.sparse.coo_matrix/csr_matrix!")
        
        adj_normalized = adj_to_symmetric_norm(adj, self.__r, self.__add_self_loops)
        return adj_normalized.tocsr()
