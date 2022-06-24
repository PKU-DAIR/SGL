import argparse
from sgl.dataset import Planetoid
from sgl.models.homo import SGCDist
from sgl.tasks import NodeClassificationDist

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

dataset = Planetoid("pubmed", "./", "official")
model = SGCDist(prop_steps=3, feat_dim=dataset.num_features, output_dim=dataset.num_classes)
test_acc = NodeClassificationDist(dataset, model)._execute(args)
