import yaml
import argparse

import sgl.dataset as Dataset
import sgl.models.homo as HomoModels
import sgl.sampler as Sampler
from sgl.tasks import NodeClassification_Sampling


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Sampler-Models")
    parser.add_argument(
        "--device", type=int, default=0, help="gpu device id or cpu (-1)"
    )
    parser.add_argument(
        "--config_path", type=str, default="./configs/fastgcn.yml", help="save path of the configuration file"
    )
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_path, "rb"))
    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"
    dataset_kwargs = config["dataset"]
    classname = dataset_kwargs.pop("classname")
    dataset = getattr(Dataset, classname)(**dataset_kwargs)
    sampler_kwargs = config["sampler"]
    if "inductive" in sampler_kwargs.keys():
        inductive = sampler_kwargs.pop("inductive")
    else:
        inductive = False
    sampler_name = sampler_kwargs.pop("name")
    sampler = getattr(Sampler, sampler_name)(dataset.adj[dataset.train_idx, :][:, dataset.train_idx] if inductive else dataset.adj, **sampler_kwargs)
    model_kwargs = config["model"]
    model_name = model_kwargs.pop("name")
    model_kwargs.update({"device": device})
    model = getattr(HomoModels, model_name)(dataset, sampler, **model_kwargs)
    task_kwargs = config["task"]
    task_kwargs.update({"device": device})
    test_acc = NodeClassification_Sampling(dataset, model, **task_kwargs).test_acc
    print(f"final test acc: {test_acc}")