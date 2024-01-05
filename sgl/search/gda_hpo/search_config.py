from typing import List
from openbox import space as osp

import sgl.models.homo.gda as GDAModel
import sgl.tasks as Task

class BaseGDAConfigManager:
    def __init__(self, gda_model_name: str, task_name: str, model_keys: List[str], task_keys: List[str], const_model_kwargs: dict, const_task_kwargs: dict, hier_params: dict):
        self._gda_model_name = gda_model_name 
        self._task_name = task_name
        self._model_keys = model_keys 
        self._task_keys = task_keys
        self._const_model_kwargs = const_model_kwargs 
        self._const_task_kwargs = const_task_kwargs
        self._config_space = osp.Space()
        self._setupSpace(hier_params)
    
    def _configTarget(self, params: dict):
        model_kwargs, task_kwargs = self._const_model_kwargs.copy(), self._const_task_kwargs.copy()
        for p_name, p_value in params.items():
            if p_name in self._model_keys:
                model_kwargs.update({p_name: p_value})
            elif p_name in self._task_keys:
                task_kwargs.update({p_name: p_value})
            else:
                raise ValueError(f"Get unexpected parameter {p_name}")
        model = getattr(GDAModel, self._gda_model_name)(**model_kwargs)
        task = getattr(Task, self._task_name)(model=model, **task_kwargs)
        acc_res = task._execute()

        return dict(objectives=[-acc_res])
    
    def _configSpace(self):
        return self._config_space
    
    def _configFunction(self, config_space: osp.Configuration):
        params = config_space.get_dictionary().copy()
        result = self._configTarget(params)
        return result

    def _setupSpace(self, hier_params: dict):
        for cls, variables in hier_params.items():
            """
            cls: str, variable class, Real, Int, Constant
            variables: dict, key = variable name (e.g., alpha, temperature), 
                             value = variable property (e.g., lower=0, upper=1, default_value=0.4, q=0.01)
            """
            variable_list = []
            for var_name, var_kwargs in variables.items():
                variable_list.append(getattr(osp, cls)(var_name, **var_kwargs))
            self._config_space.add_variables(variable_list)