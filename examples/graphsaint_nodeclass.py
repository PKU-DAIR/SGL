import yaml
import argparse

import sgl.dataset as Dataset
from sgl.models.homo import GraphSAINT

import sgl.sampler as Sampler
from sgl.sampler import GraphSAINTSampler
from sgl.tasks import NodeClassification_Sampling

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphSaint-Models.")
    parser.add_argument(
        "--device", type=int, default=0, help="gpu device id or cpu (-1)"
    )
    parser.add_argument(
        "--config_path", type=str, default="./configs/graphsaint.yml", help="save path of the configuration file"
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

    train_sampler = GraphSAINTSampler(dataset, **train_sampler_kwargs)
    if "eval" in sampler_kwargs:
        eval_sampler_kwargs = sampler_kwargs["eval"]
        eval_sampler_name = eval_sampler_kwargs["name"]
        if eval_sampler_name == "ClusterGCNSampler":
            eval_sampler_kwargs.update({"save_dir": dataset.processed_dir})
            eval_cluster_number = eval_sampler_kwargs["cluster_number"]
            task_kwargs.update({"eval_graph_number": eval_cluster_number})
            eval_sampler = GraphSAINTSampler(dataset, **eval_sampler_kwargs)
        else:
            eval_sampler = getattr(Sampler, eval_sampler_name)(dataset.adj, **eval_sampler_kwargs)
    else:
        eval_sampler = None

    model_kwargs.update({"device": device})
    model = GraphSAINT(dataset, train_sampler, eval_sampler, **model_kwargs)
    task_kwargs.update({"device": device})
    task_kwargs.update({"loss_fn": model.loss_fn})
    test_acc = NodeClassification_Sampling(dataset, model, **task_kwargs).test_acc
