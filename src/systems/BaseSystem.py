from abc import ABC, abstractmethod
import numpy as np


class BaseSystem(ABC):
    @abstractmethod
    def __init__(self):
        pass


    @abstractmethod
    def getParameters(self):
        pass


    def addGrid(self, grid):
        self.grid = dict()
        for parameter_name, parameter_values in grid.items():
            if not hasattr(self, parameter_name):
                raise RuntimeError(f"Class {self.__class__.__name__} doesn't have parameter {parameter_name}")

            start_param         = getattr(self, parameter_name)
            left_side_interval  = parameter_values['interval'][0]
            right_side_interval = parameter_values['interval'][1]

            self.grid[parameter_name] = np.arange(start_param - left_side_interval, start_param + right_side_interval + 1, parameter_values['step'])


    def getParametersPlaces(self):
        n = 0
        self.param_places = []
        for param_name in self.param_names:
            param_value = getattr(self, param_name)
            if isinstance(param_value, (list, tuple)):
                param_len = len(param_value)
            else:
                param_len = 1
            self.param_places.append([n, param_len])
            n = n + param_len
        return self.param_places


    def getParametersToChange(self):
        params_to_change = []
        for param_name, _ in self.grid.items():
            param_index = self.param_names.index(param_name)
            params_to_change.append(param_index)
        return params_to_change


    def flatten_params(self, params):
        flat_list = []
        for param in params:
            if isinstance(param, (list, tuple)):
                flat_list.extend(self.flatten_params(param))
            else:
                flat_list.append(param)
        return flat_list

