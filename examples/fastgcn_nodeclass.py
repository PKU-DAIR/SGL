import argparse
import torch.nn.functional as F
from sgl.dataset import Planetoid
from sgl.models.homo import FastGCN, GraphSAGE, VanillaGCN
from sgl.tasks import NodeClassification_Sampling


if __name__ == "__main__":
    parser = argparse.ArgumentParser("FastGCN")
    parser.add_argument(
        "--hidden", type=int, default=128, help="dimension of hidden layer"
    )
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout")
    parser.add_argument(
        "--layer_sizes", type=str, default="128-128", help="sampling sizes per layer"
    )
    args = parser.parse_args()
    device = "cuda:0"
    dataset = Planetoid("cora", "/home/ssq/test_data/", "official")
    # model = FastGCN(
    #     dataset,
    #     hidden_dim=args.hidden,
    #     output_dim=dataset.num_classes,
    #     dropout=args.dropout,
    #     device=device,
    #     inductive=False,
    #     prob_type="uniform"
    # )
    # test_acc = NodeClassification_Sampling(
    #     dataset,
    #     model,
    #     lr=0.1,
    #     weight_decay=5e-5,
    #     epochs=20,
    #     device=device,
    #     loss_fn=F.nll_loss,
    #     train_batch_size=256,
    #     eval_batch_size=256,
    # ).test_acc
    # print(f"final test acc: {test_acc}")
    model = GraphSAGE(
        dataset,
        hidden_dim=args.hidden,
        output_dim=dataset.num_classes,
        dropout=args.dropout,
        device=device,
    )
    test_acc = NodeClassification_Sampling(
        dataset,
        model,
        lr=0.1,
        weight_decay=5e-5,
        epochs=20,
        device=device,
        loss_fn=F.nll_loss,
        train_batch_size=64,
        eval_batch_size=64,
    ).test_acc
    print(f"final test acc: {test_acc}")
    # model = VanillaGCN(
    #     dataset,
    #     hidden_dim=args.hidden,
    #     output_dim=dataset.num_classes,
    #     dropout=args.dropout,
    #     device=device,
    # )
    # test_acc = NodeClassification_Sampling(
    #     dataset,
    #     model,
    #     lr=0.1,
    #     weight_decay=5e-5,
    #     epochs=20,
    #     device=device,
    #     loss_fn=F.nll_loss
    # ).test_acc
    # print(f"final test acc: {test_acc}")
