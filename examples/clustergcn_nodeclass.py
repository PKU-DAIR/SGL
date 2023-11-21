import yaml
import argparse
import sgl.dataset as Dataset
from sgl.models.homo import ClusterGCN
import sgl.sampler as Sampler
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
    sampler_kwargs = config["sampler"]
    model_kwargs = config["model"]
    task_kwargs = config["task"]
    
    classname = dataset_kwargs.pop("classname")
    dataset = getattr(Dataset, classname)(**dataset_kwargs)
    train_sampler_kwargs = sampler_kwargs["train"]
    train_sampler_kwargs.update({"save_dir": dataset.processed_dir})
    train_cluster_number = train_sampler_kwargs["cluster_number"]
    task_kwargs.update({"train_graph_number": train_cluster_number})
    train_sampler = ClusterGCNSampler(dataset, **train_sampler_kwargs)
    if "eval" in sampler_kwargs:
        eval_sampler_kwargs = sampler_kwargs["eval"]
        eval_sampler_name = eval_sampler_kwargs["name"]
        if eval_sampler_name == "ClusterGCNSampler":
            eval_sampler_kwargs.update({"save_dir": dataset.processed_dir})
            eval_cluster_number = eval_sampler_kwargs["cluster_number"]
            task_kwargs.update({"eval_graph_number": eval_cluster_number})
            eval_sampler = ClusterGCNSampler(dataset, **eval_sampler_kwargs)
        else:
            eval_sampler = getattr(Sampler, eval_sampler_name)(dataset.adj, **eval_sampler_kwargs)
    else:
        eval_sampler = None
    model_kwargs.update({"device": device})
    model = ClusterGCN(train_sampler, eval_sampler, nfeat=dataset.num_features, nclass=dataset.num_classes, **model_kwargs)
    task_kwargs.update({"device": device})
    test_acc = NodeClassification_Sampling(dataset, model, **task_kwargs).test_acc
