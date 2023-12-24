import torch
import argparse
from openbox import Optimizer

import sgl.dataset as Dataset
from sgl.search.gda_hpo.GAug_search_config import GAugOConfigManager, GAugMConfigManager

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
        configer = GAugOConfigManager(dataset, args.gnn_type, not args.not_gae, device, minibatch=args.minibatch, pretrain_batch_size=pretrain_batch_size, train_batch_size=train_batch_size, runs=args.runs_per_config, max_patience=args.max_patience)
    else:
        configer = GAugMConfigManager(dataset, args.gnn_type, not args.not_gae, device, args.num_logits, runs=args.runs_per_config, max_patience=args.max_patience)

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