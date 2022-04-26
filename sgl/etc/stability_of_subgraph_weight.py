import numpy as np

from sgl.dataset import Dblp
from sgl.etc.auto_select_edge_type_for_nars import GenerateSubgraphList, OneTrialWithSubgraphList

SUBGRAPH_COMBINATIONS = 20
NUM_REPEAT = 5
SUBGRAPH_CONFIG = [(3, 2), (3, 3)]
NUM_EPOCHS_TO_FIND_WEIGHT = 20


def main():
    dataset = Dblp(root='.', path_of_zip='./dataset/DBLP_processed.zip')
    with open('subgraph_weight.txt', 'w') as output:
        for i in range(SUBGRAPH_COMBINATIONS):
            output.write(f'Subgraph combination {i + 1}\n')
            subgraph_list = GenerateSubgraphList(dataset, SUBGRAPH_CONFIG)
            for j in range(NUM_REPEAT):
                output.write(f'\tIteration {j + 1}\n')
                test_acc, _, subgraph_weight = OneTrialWithSubgraphList(
                    dataset, subgraph_list, NUM_EPOCHS_TO_FIND_WEIGHT)
                index_sort = np.argsort(subgraph_weight.detach().numpy())
                output.write(f'\t\tTest accuracy: {test_acc}\n')
                output.write(f'\t\tSubgraph weight: {index_sort}\n\n')
                output.flush()


if __name__ == '__main__':
    main()
