import torch

from dataset.ogbn_mag import OgbnMag
from dataset.dblp import Dblp
from models.hetero_SGAP_models import NARS_SIGN, NARS_SIGN_WeightSharedAcrossFeatures, NARS_SGC_WithLearnableWeights, Fast_NARS_SGC_WithLearnableWeights
from tasks.node_classification import HeteroNodeClassification
from auto_choose_gpu import GpuWithMaxFreeMem

NUM_EPOCHS = 100

dataset = Dblp(root='.', path_of_zip='./dataset/DBLP_processed.zip')
#dataset = OgbnMag(name="mag", root="./")
predict_class = dataset.TYPE_OF_NODE_TO_PREDICT


def OneTrial(random_subgraph_num: int, subgraph_edge_type_num: int) -> float:
    model = Fast_NARS_SGC_WithLearnableWeights(prop_steps=3,
                                               feat_dim=dataset.data.num_features[predict_class],
                                               num_classes=dataset.data.num_classes[predict_class],
                                               hidden_dim=256, num_layers=2,
                                               random_subgraph_num=random_subgraph_num)

    device = torch.device(
        f"cuda:{GpuWithMaxFreeMem()}" if torch.cuda.is_available() else "cpu")
    test_acc = HeteroNodeClassification(dataset, predict_class, model,
                                        lr=0.01, weight_decay=0, epochs=NUM_EPOCHS, device=device,
                                        train_batch_size=10000, eval_batch_size=10000,
                                        random_subgraph_num=random_subgraph_num,
                                        subgraph_edge_type_num=subgraph_edge_type_num).test_acc

    print(model.subgraph_weight)

    return test_acc


with open('search_result.txt', 'w') as output:
    for random_subgraph_num in range(2, 20):
        for subgraph_edge_type_num in range(2, dataset.edge_type_cnt):
            for iteration in range(3):
                output.writelines(
                    f'''random_subgraph_num: {random_subgraph_num}, ''' +
                    f'''subgraph_edge_type_num: {subgraph_edge_type_num}, ''' +
                    f'''iteration:{iteration}''' + '\t')
                test_acc = OneTrial(random_subgraph_num,
                                    subgraph_edge_type_num)
                output.writelines(f'test accuracy: {test_acc}\n')
                output.flush()
