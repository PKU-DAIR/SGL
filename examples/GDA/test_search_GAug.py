import torch
import argparse
import torch.nn.functional as F
from openbox import Optimizer

import sgl.dataset as Dataset
from sgl.search.gda_hpo.search_config import BaseGDAConfigManager

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HPO-GAug-Model.")
    parser.add_argument("--device", type=int, default=0, help="gpu device id or cpu(-1)")
    parser.add_argument("--dataset_classname", type=str, default="Planetoid", help="class name of the dataset")
    parser.add_argument("--name", type=str, default="cora", help="dataset name")
    parser.add_argument("--root", type=str, default="/home/ssq/test_data/", help="root dir for dataset")
    parser.add_argument("--gnn_type", type=str, default="gcn", choices=["gcn", "gsage", "gat"], help="gnn backbone")
    parser.add_argument("--not_gae", action="store_true", default=False, help="whether not to use gae")
    parser.add_argument("--minibatch", action="store_true", default=False, help="whether to use minibatch")
    parser.add_argument("--pretrain_batch_size", type=int, default=-1, help="batch size when pretraining ep net")
    parser.add_argument("--train_batch_size", type=int, default=-1, help="batch size when training")
    parser.add_argument("--model", type=str, default="GAugO", choices=["GAugO", "GAugM"], help="choose the target mnodel")
    parser.add_argument("--num_logits", type=int, default=10, help="number of candidate edge logits")
    parser.add_argument("--runs_per_config", type=int, default=5, help="repeat runs for each configuration")
    parser.add_argument("--max_patience", type=int, default=50, help="patience for early stop")
    args = parser.parse_args()
    device = f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu"
    dataset = getattr(Dataset, args.dataset_classname)(name=args.name, root=args.root)
    pretrain_batch_size = args.pretrain_batch_size if args.pretrain_batch_size > 0 else None
    train_batch_size = args.train_batch_size if args.train_batch_size > 0 else None
    if args.model == "GAugO":
        model_keys = ["in_dim", "hidden_dim", "emb_size", "n_classes", "n_layers", "dropout", "gnn_type", "activation", "temperature", "gae", "alpha", "feat_norm", "sample_type", "minibatch", "n_heads"]
        task_keys = ["dataset", "model", "lr", "weight_decay", "epochs", "device", "beta", "warmup", "max_patience", "pretrain_ep", "pretrain_nc", "runs", "verbose", "seed", "pretrain_batch_size", "train_batch_size", "ep_lr"]
        const_model_kwargs = dict(in_dim=dataset.num_features, n_classes=dataset.num_classes, gnn_type=args.gnn_type, activation=F.relu, gae=not args.not_gae, minibatch=args.minibatch, emb_size=32, n_layers=2, dropout=0.5, feat_norm="row")
        const_task_kwargs = dict(dataset=dataset, epochs=200, device=device, max_patience=args.max_patience, pretrain_batch_size=pretrain_batch_size, train_batch_size=train_batch_size, runs=args.runs_per_config, verbose=False, lr=0.01, weight_decay=0.0005)
        Reals = dict(alpha=dict(lower=0, upper=1, default_value=0.4, q=0.01), temperature=dict(lower=0.1, upper=2.1, default_value=1.5, q=0.1), beta=dict(lower=0, upper=4, default_value=2, q=0.1))
        if pretrain_batch_size is not None:
            Reals.update(ep_lr=dict(lower=0.001, upper=0.01, default_value=0.002, q=0.001))
        else:
            const_task_kwargs.update(ep_lr=0.01)
        Categoricals = dict(hidden_dim=dict(choices=[32, 64, 128, 256], default_value=128))
        Ints = dict(warmup=dict(lower=0, upper=10, default_value=2, q=1), pretrain_ep=dict(lower=5, upper=300, default_value=100, q=5), pretrain_nc=dict(lower=5, upper=300, default_value=100, q=5))
        hier_params = dict(Real=Reals, Categorical=Categoricals, Int=Ints)
        configer = BaseGDAConfigManager(args.model, f"NodeClassification{args.model}", model_keys, task_keys, const_model_kwargs, const_task_kwargs, hier_params)
    else:
        model_keys = ["in_dim", "hidden_dim", "n_classes", "n_layers", "gnn_type", "rm_pct", "add_pct", "choose_idx", "gae", "dropout", "activation", "feat_norm", "n_heads"]
        task_keys = ["dataset", "model", "lr", "weight_decay", "epochs", "device", "max_patience", "runs", "verbose", "seed"]
        const_model_kwargs = dict(in_dim=dataset.num_features, n_classes=dataset.num_classes, gnn_type=args.gnn_type, activation=F.relu, gae=not args.not_gae, n_layers=2, dropout=0.5, feat_norm="row")
        const_task_kwargs = dict(dataset=dataset, epochs=200, device=device, max_patience=args.max_patience, runs=args.runs_per_config, verbose=False, lr=0.01, weight_decay=0.0005)
        Categoricals = dict(hidden_dim=dict(choices=[32, 64, 128, 256], default_value=128))
        Ints = dict(choose_idx=dict(lower=1, upper=args.num_logits, default_value=1, q=1), rm_pct=dict(lower=0, upper=80, default_value=20, q=1), add_pct=dict(lower=0, upper=80, default_value=20, q=1))
        hier_params = dict(Categorical=Categoricals, Int=Ints)
        configer = BaseGDAConfigManager(args.model, f"NodeClassification{args.model}", model_keys, task_keys, const_model_kwargs, const_task_kwargs, hier_params)

    opt = Optimizer(configer._configFunction,
                    configer._configSpace(),
                    num_objectives=1,
                    num_constraints=0,
                    max_runs=400,
                    surrogate_type="prf",
                    acq_type='ei',
                    acq_optimizer_type='local_random',
                    initial_runs=20,
                    task_id='quick_start',
                    random_state=1)
    
    history = opt.run()
    print(history)