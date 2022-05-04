import numpy as np
<<<<<<< Updated upstream
from sgl.search.search_models import SearchModel
from sgl.search.auto_search import SearchManager
=======
>>>>>>> Stashed changes
from openbox.utils.config_space import ConfigurationSpace, UniformIntegerHyperparameter

from sgl.search.auto_search import SearchManager
from sgl.search.search_models import SearchModel


class ConfigManager():
    def __init__(self, arch, prop_steps=[1, 10], prop_types=[1, 4], mesg_types=[0, 8], num_layers=[1, 10],
                 post_steps=[1, 10], post_types=[1, 4], pmsg_types=[0, 5]):
        super(ConfigManager, self).__init__()

        self.__initial_arch = arch
        self.__config_space = ConfigurationSpace()
        self.__prop_steps = UniformIntegerHyperparameter("prop_steps", prop_steps[0], prop_steps[1])
        self.__prop_types = UniformIntegerHyperparameter("prop_types", prop_types[0], prop_types[1])
        self.__mesg_types = UniformIntegerHyperparameter("mesg_types", mesg_types[0], mesg_types[1])
        self.__num_layers = UniformIntegerHyperparameter("num_layers", num_layers[0], num_layers[1])
        self.__post_steps = UniformIntegerHyperparameter("post_steps", post_steps[0], post_steps[1])
        self.__post_types = UniformIntegerHyperparameter("post_types", post_types[0], post_types[1])
        self.__pmsg_types = UniformIntegerHyperparameter("pmsg_types", pmsg_types[0], pmsg_types[1])
        self.__config_space.add_hyperparameters(
            [self.__prop_steps, self.__prop_types, self.__mesg_types, self.__num_layers, self.__post_steps,
             self.__post_types, self.__pmsg_types])

    def _setParameters(self, dataset, device, hiddim, epochs, lr, wd):
        self.__dataset = dataset
        self.__device = device
        self.__hiddim = hiddim
        self.__epochs = epochs
        self.__lr = lr
        self.__wd = wd

    def _configSpace(self):
        return self.__config_space

    def _configTarget(self, arch):
        model = SearchModel(arch, self.__dataset.num_features, int(self.__dataset.num_classes), self.__hiddim)
        acc_res, time_res = SearchManager(self.__dataset, model, lr=self.__lr, weight_decay=self.__wd,
                                          epochs=self.__epochs, device=self.__device)._execute()
        result = dict()
        result['objs'] = np.stack([-acc_res, time_res], axis=-1)
        return result

    def _configFunction(self, config_space):
        self.__initial_arch[0] = config_space['prop_steps']
        self.__initial_arch[1] = config_space['prop_types']
        self.__initial_arch[2] = config_space['mesg_types']
        self.__initial_arch[3] = config_space['num_layers']
        self.__initial_arch[4] = config_space['post_steps']
        self.__initial_arch[5] = config_space['post_types']
        self.__initial_arch[6] = config_space['pmsg_types']
        result = self._configTarget(self.__initial_arch)
        return result
