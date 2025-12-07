from src.systems.BaseSystem import BaseSystem
from numba import cuda
import numpy as np
import math


class MultiPeak(BaseSystem):
    """
    Двухмерная динамическая система со множеством локальных экстремумов.
    """
    module_name = 'src.systems.MultiPeak'

    def __init__(self, amplitude: float, frequency: float):
        self.param_names = ['amplitude', 'frequency']
        self.amplitude = amplitude
        self.frequency = frequency

    def getSystem(self, t, X):
        x, y = X
        A = self.amplitude
        f = self.frequency

        dx = A * np.sin(f * x) * np.cos(f * y)
        dy = A * np.cos(f * x) * np.sin(f * y)

        return [dx, dy]

    def getParameters(self):
        return [self.amplitude, self.frequency]


@cuda.jit
def rhs_jit(X, params, dX):
    amplitude = params[0]
    frequency = params[1]

    x = X[0]
    y = X[1]

    dx = amplitude * math.sin(frequency * x) * math.cos(frequency * y)
    dy = amplitude * math.cos(frequency * x) * math.sin(frequency * y)

    dX[0] = dx
    dX[1] = dy
