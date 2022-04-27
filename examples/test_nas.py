import torch
from openbox.optimizer.generic_smbo import SMBO

from sgl.dataset.planetoid import Planetoid
from sgl.search.search_config import ConfigManager

dataset = Planetoid("cora", "./", "official")
device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")

## Define Initial Arch and Configuration
initial_arch = [2, 0, 1, 2, 3, 0, 0]
configer = ConfigManager(initial_arch)
configer._setParameters(dataset, device, 128, 200, 1e-2, 5e-4)

## Define Search Parameters
dim = 7
bo = SMBO(configer._configFunction,
          configer._configSpace(),
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

## Search
history = bo.run()
print(history)
