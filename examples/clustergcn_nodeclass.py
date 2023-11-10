import argparse
import networkx as nx
import torch.nn.functional as F
from sgl.dataset import Planetoid
from sgl.models.homo import ClusterGCN
from sgl.tasks import NodeClassification_Sampling


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Run .")
    parser.add_argument("--clustering_method",
                        nargs = "?",
                        default = "random",
                        choices = ["random", "metis"],
	                help = "Clustering method for graph decomposition. Default is the random procedure.")

    parser.add_argument("--epochs",
                        type = int,
                        default = 200,
	                help = "Number of training epochs. Default is 200.")

    parser.add_argument("--seed",
                        type = int,
                        default = 42,
	                help = "Random seed for train_test split. Default is 42.")

    parser.add_argument("--dropout",
                        type = float,
                        default = 0.5,
	                help = "Dropout parameter. Default is 0.5.")

    parser.add_argument("--learning_rate",
                        type = float,
                        default = 0.01,
	                help = "Learning rate. Default is 0.01.")

    parser.add_argument("--test_ratio",
                        type = float,
                        default = 0.9,
	                help = "Test data ratio. Default is 0.1.")

    parser.add_argument("--cluster_number",
                        type = int,
                        default = 10,
                        help = "Number of clusters extracted. Default is 10.")
    args = parser.parse_args()
    device = 'cuda:0'
    dataset = Planetoid("cora", "/home/ssq/test_data/", f"clustergcn_{args.cluster_number}")
    model = ClusterGCN(nx.from_scipy_sparse_matrix(dataset.adj), dataset.x.numpy(), dataset.y.unsqueeze(1).numpy(), device, dataset.num_features, 128, dataset.num_classes, args.clustering_method, args.cluster_number, args.test_ratio)
    test_acc = NodeClassification_Sampling(dataset, model, lr=0.1, weight_decay=5e-5, epochs=30, device=device, loss_fn=F.nll_loss, train_batch_size=1, eval_batch_size=1).test_acc
