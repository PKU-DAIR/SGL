import yaml
import argparse
import sgl.dataset as Dataset
from sgl.models.homo import ClusterGCN
from sgl.sampler import ClusterGCNSampler
from sgl.tasks import NodeClassification_Sampling


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "ClusterGCNSampler-Models.")
    parser.add_argument(
        "--device", type=int, default=0, help="gpu device id or cpu (-1)"
    )
    parser.add_argument(
        "--config_path", type=str, default="./configs/clustergcn.yml", help="save path of the configuration file"
    )
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_path, "rb"))
    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"
    dataset_kwargs = config["dataset"]
    cluster_number = config["sampler"]["cluster_number"]
    classname = dataset_kwargs.pop("classname")
    dataset = getattr(Dataset, classname)(**dataset_kwargs)
    sampler_kwargs = config["sampler"]
    sampler = ClusterGCNSampler(dataset, **sampler_kwargs)
    model_kwargs = config["model"]
    model_kwargs.update({"device": device})
    model = ClusterGCN(sampler, nfeat=dataset.num_features, nclass=dataset.num_classes, **model_kwargs)
    task_kwargs = config["task"]
    task_kwargs.update({"device": device})
    task_kwargs.update({"graph_number": cluster_number})
    test_acc = NodeClassification_Sampling(dataset, model, **task_kwargs).test_acc
