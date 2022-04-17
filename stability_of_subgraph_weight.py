from auto_select_edge_type_for_nars import SMALL_NUM_EPOCHS, \
    GenerateSubgraphList, OneTrialWithSubgraphList
from dataset.dblp import Dblp

SUBGRAPH_COMBINATIONS = 20
NUM_REPEAT = 5
SUBGRAPH_CONFIG = [(3, 2), (3, 3)]


def main():
    dataset = Dblp(root='.', path_of_zip='./dataset/DBLP_processed.zip')
    with open('subgraph_weight.txt', 'w') as output:
        for i in range(SUBGRAPH_COMBINATIONS):
            output.write(f'Subgraph combination {i+1}\n')
            subgraph_list = GenerateSubgraphList(dataset, SUBGRAPH_CONFIG)
            for j in range(NUM_REPEAT):
                output.write(f'\tIteration {j+1}\n')
                test_acc, _, subgraph_weight = OneTrialWithSubgraphList(
                    dataset, subgraph_list, SMALL_NUM_EPOCHS)
                output.write(f'\t\tTest accuracy: {test_acc}\n')
                output.write(f'\t\tSubgraph weight: {subgraph_weight}\n\n')
                output.flush()


if __name__ == '__main__':
    main()
