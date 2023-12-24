from openbox import space as osp

class BaseGDAConfigManager():
    def __init__(self):
        super(BaseGDAConfigManager, self).__init__()
        self.__config_space = None
    
    def _configTarget(self, params):
        raise NotImplementedError
    
    def _configFunction(self, config_space: osp.Configuration):
        params = config_space.get_dictionary().copy()
        result = self._configTarget(params)
        return result
