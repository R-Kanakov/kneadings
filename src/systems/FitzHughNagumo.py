from .BaseSystem import BaseSystem
from numba import cuda


class FitzHughNagumo(BaseSystem):
    """"FitzHugh-Nagumo system"""
    module_name = 'src.systems.FitzHughNagumo'


    def __init__(self, a, b, S):
        assert a > 0., "'a' must be greater than zero!"
        self.a = a
        assert b > 0., "'b' must be greater than zero!"
        self.b = b
        # S - внешнее воздействие:
        self.S = S


    def getParameters(self):
        params = self.flatten_params((self.a, self.b))
        return params


    def getSystem(self, t, X):
        x, y = X
        a, b, S = self.a, self.b, self.S
        dx = x - 1/3 * x ** 3 - y + S
        dy = x + a - b * y
        return [dx, dy]


    def setParams(self, paramDict):
        for key in paramDict:
            if hasattr(self, key):
                setattr(self, key, paramDict[key])
            else:
                raise KeyError(f"System has no parameter '{key}'")


@cuda.jit
def rhs_jit(X, a, b, S, dX):
    x = X[0]
    y = X[1]

    dx = x - (1.0 / 3.0) * x ** 3 - y + S
    dy = x + a - b * y

    dX[0] = dx
    dX[1] = dy
