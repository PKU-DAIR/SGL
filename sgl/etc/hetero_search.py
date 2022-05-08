import time
import torch

from sgl.dataset import Dblp
from sgl.models.hetero import NARS_SIGN, Fast_NARS_SGC_WithLearnableWeights
from sgl.tasks import HeteroNodeClassification
from sgl.utils import GpuWithMaxFreeMem

# Hyperparameters
PROP_STEPS = 3
HIDDEN_DIM = 256
NUM_LAYERS = 2
NUM_EPOCHS = 10
LR = 0.01
WEIGHT_DECAY = 0.0
BATCH_SIZE = 10000


def OneTrial(dataset, random_subgraph_num: int, subgraph_edge_type_num: int) -> float:
    predict_class = dataset.TYPE_OF_NODE_TO_PREDICT
    model = Fast_NARS_SGC_WithLearnableWeights(prop_steps=PROP_STEPS,
                                               feat_dim=dataset.data.num_features[predict_class],
                                               output_dim=dataset.data.num_classes[predict_class],
                                               hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                                               random_subgraph_num=random_subgraph_num)

    device = torch.device(
        f"cuda:{GpuWithMaxFreeMem()}" if torch.cuda.is_available() else "cpu")
    classification = HeteroNodeClassification(dataset, predict_class, model,
                                              lr=LR, weight_decay=WEIGHT_DECAY,
                                              epochs=NUM_EPOCHS, device=device,
                                              train_batch_size=BATCH_SIZE,
                                              eval_batch_size=BATCH_SIZE,
                                              random_subgraph_num=random_subgraph_num,
                                              subgraph_edge_type_num=subgraph_edge_type_num,
                                              seed=int(time.time()))

    test_acc = classification.test_acc
    raw_weight = classification.subgraph_weight
    weight_sum = raw_weight.sum()
    normalized_weight = raw_weight / weight_sum
    print(normalized_weight)

    return test_acc


def main():
    dataset = Dblp(root='.', path_of_zip='./dataset/DBLP_processed.zip')

    with open('search_result.txt', 'w') as output:
        for random_subgraph_num in range(2, 20):
            for subgraph_edge_type_num in range(2, dataset.edge_type_cnt):
                for iteration in range(3):
                    output.writelines(
                        f'''random_subgraph_num: {random_subgraph_num}, ''' +
                        f'''subgraph_edge_type_num: {subgraph_edge_type_num}, ''' +
                        f'''iteration:{iteration}''' + '\t')
                    test_acc = OneTrial(
                        dataset, random_subgraph_num, subgraph_edge_type_num)
                    output.writelines(f'test accuracy: {test_acc}\n')
                    output.flush()


if __name__ == '__main__':
    main()
