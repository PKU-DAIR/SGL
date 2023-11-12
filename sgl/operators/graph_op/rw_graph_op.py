import scipy.sparse as sp

from sgl.operators.base_op import GraphOp
from sgl.operators.utils import adj_to_row_norm


class RwGraphOP(GraphOp):
    def __init__(self, prop_steps=-1):
        super(RwGraphOP, self).__init__(prop_steps)

    def _construct_adj(self, adj):
        if isinstance(adj, sp.csr_matrix):
            adj = adj.tocoo()
        elif not isinstance(adj, sp.coo_matrix):
            raise TypeError("The adjacency matrix must be a scipy.sparse.coo_matrix/csr_matrix!")
        
        adj_normalized = adj_to_row_norm(adj)
        return adj_normalized.tocsr()
