import yaml
import argparse
import scipy.sparse as sp

import sgl.tasks as Tasks
import sgl.dataset as Dataset
import sgl.sampler as Sampler
from sgl.models.homo.gda import SampleMixup


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Sampler-Models")
    parser.add_argument(
        "--device", type=int, default=0, help="gpu device id or cpu (-1)"
    )
    parser.add_argument(
        "--config_path", type=str, default="./configs/SampleMixup.yml", help="save path of the configuration file"
    )
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_path, "rb"))
    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"
    dataset_kwargs = config["dataset"]
    task_kwargs = config["task"]
    classname = dataset_kwargs.pop("classname")
    dataset = getattr(Dataset, classname)(**dataset_kwargs)
    adj_matrix = dataset.adj
    if isinstance(adj_matrix, sp.coo_matrix) is False:
        adj_matrix = sp.coo_matrix(adj_matrix)
    adj_matrix.setdiag(0)
    adj_matrix = adj_matrix.tocsr()
    training_sampler_kwargs = config["sampler"]["training"]
    if "inductive" in training_sampler_kwargs.keys():
        inductive = training_sampler_kwargs.pop("inductive")
    else:
        inductive = False
    task_kwargs.update({"inductive": inductive})
    training_sampler_name = training_sampler_kwargs.pop("name")
    training_sampler_kwargs.update({"save_dir": dataset.processed_dir})
    training_sampler = getattr(Sampler, training_sampler_name)(adj_matrix[dataset.train_idx, :][:, dataset.train_idx] if inductive else adj_matrix, **training_sampler_kwargs)
    if "eval" in config["sampler"].keys():
        eval_sampler_kwargs = config["sampler"]["eval"]
        eval_sampler_name = eval_sampler_kwargs.pop("name")
        eval_sampler_kwargs.update({"save_dir": dataset.processed_dir})
        eval_sampler = getattr(Sampler, eval_sampler_name)(adj_matrix, **eval_sampler_kwargs)
    else:
        eval_sampler = None
    model_kwargs = config["model"]
    model = SampleMixup(training_sampler, eval_sampler, in_dim=dataset.num_features, n_classes=dataset.num_classes, **model_kwargs)
    task_kwargs.update({"device": device})
    task_kwargs.update({"loss_fn": model.loss_fn})
    task_name = task_kwargs.pop("name")
    test_acc = getattr(Tasks, task_name)(dataset, model, **task_kwargs).test_acc
    print(f"final test acc: {test_acc}")