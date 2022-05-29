import argparse
from openbox.optimizer.generic_smbo import SMBO

from sgl.dataset.planetoid import Planetoid
from sgl.search.search_config_dist import ConfigManagerDist

def main():
    initial_arch = [2, 0, 1, 2, 3, 0, 0]
    dataset = Planetoid("cora", "./", "official")

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=2, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--lr', default=1e-2, type=float, metavar='LR',
                        help='learning rate')
    parser.add_argument('--wd', default=5e-4, type=float, metavar='WD',
                        help='weight decay')                        
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--hidden', default=128, type=int, metavar='N',
                        help='hidden_dims')
    parser.add_argument('--batch', default=128, type=int, metavar='N',
                        help='batch_size')
    args = parser.parse_args()

    configer = ConfigManagerDist(initial_arch)
    configer._setParameters(dataset, args)

    dim = 7
    bo = SMBO(configer._configFunction,
            configer._configSpace(),
            num_objs=2,
            num_constraints=0,
            max_runs=3500,
            surrogate_type='prf',
            acq_type='ehvi',
            acq_optimizer_type='local_random',
            initial_runs=2*(dim+1),
            init_strategy='sobol',
            ref_point=[-1, 0.00001],
            time_limit_per_trial=5000,
            task_id='quick_start',
            random_state=1)
    history = bo.run()
    return history

if __name__ == '__main__':
    res = main()
    print(res)
