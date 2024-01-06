import argparse
from sgl.dataset import Ogbn
from sgl.models.homo import GAMLP
from sgl.tasks import NodeClassification

if __name__ == "__main__":
    parser = argparse.ArgumentParser("GMLP")
    parser.add_argument("--device", type=int, default=0, help="GPU ID or CPU (-1)")
    parser.add_argument("--hidden-dim", type=int, default=512, help="dimension of hidden layer")
    parser.add_argument("--num-layers", type=int, default=3, help="number of layers")
    parser.add_argument("--dataset_root", type=str, default="data/", help="dataset path")
    args = parser.parse_args()

    dataset = Ogbn("products", args.dataset_root, "official")
    model = GAMLP(prop_steps=3, feat_dim=dataset.num_features, output_dim=dataset.num_classes,
                  hidden_dim=args.hidden_dim, num_layers=args.num_layers)

    device = f"cuda:{args.device}"
    test_acc = NodeClassification(dataset, model, lr=0.1, weight_decay=5e-5, epochs=200, device=device, train_batch_size=100000, eval_batch_size=200000).test_acc
