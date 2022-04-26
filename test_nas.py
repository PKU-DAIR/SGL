import numpy as np
import torch

from dataset import Planetoid
from models.search_models import SearchModel
from search.auto_search import SearchManager

dataset = Planetoid("cora", "./", "official")


def AutoSearch(arch):
    model = SearchModel(arch, dataset.num_features, int(dataset.num_classes), 64)
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    acc_res, time_res = SearchManager(dataset, model, lr=0.01, weight_decay=5e-4, epochs=200, device=device)._execute()
    result = dict()
    result['objs'] = np.stack([-acc_res, time_res], axis=-1)
    return result


res = AutoSearch([2, 0, 1, 2, 3, 0, 0])
print(res)

import math
from openbox.core.base import Observation
from openbox.optimizer.generic_smbo import SMBO
from openbox.utils.constants import MAXINT, SUCCESS, FAILED, TIMEOUT
from openbox.utils.limit import time_limit, TimeoutException
from openbox.utils.util_funcs import get_result
from openbox.utils.config_space import ConfigurationSpace, UniformFloatHyperparameter, CategoricalHyperparameter, \
    UniformIntegerHyperparameter

# Define Configuration Space
config_space = ConfigurationSpace()
prop_steps = UniformIntegerHyperparameter("prop_steps", 1, 10, default_value=3)
prop_types = UniformIntegerHyperparameter("prop_types", 0, 1)
mesg_types = UniformIntegerHyperparameter("mesg_types", 0, 8, default_value=2)
num_layers = UniformIntegerHyperparameter("num_layers", 1, 10, default_value=2)
post_steps = UniformIntegerHyperparameter("post_steps", 1, 10, default_value=0)
post_types = UniformIntegerHyperparameter("post_types", 0, 1)
pmsg_types = UniformIntegerHyperparameter("pmsg_types", 0, 5)
config_space.add_hyperparameters([prop_steps, prop_types, mesg_types, num_layers, post_steps, post_types, pmsg_types])


def SearchTarget(config_space):
    arch = [2, 0, 1, 2, 3, 0, 0]
    arch[0] = config_space['prop_steps']
    arch[1] = config_space['prop_types']
    arch[2] = config_space['mesg_types']
    arch[3] = config_space['num_layers']
    arch[4] = config_space['post_steps']
    arch[5] = config_space['post_types']
    arch[6] = config_space['pmsg_types']
    result = AutoSearch(arch)
    return result


dim = 7
bo = SMBO(SearchTarget,
          config_space,
          num_objs=2,
          num_constraints=0,
          max_runs=3500,
          surrogate_type='prf',
          acq_type='ehvi',
          acq_optimizer_type='local_random',
          initial_runs=2 * (dim + 1),
          init_strategy='sobol',
          ref_point=[-1, 0.00001],
          time_limit_per_trial=5000,
          task_id='quick_start',
          random_state=1)
history = bo.run()
print(history)
