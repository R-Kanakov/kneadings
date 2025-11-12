from .FitzHughNagumo import FitzHughNagumo
from .FitzHughNagumo import rhs_jit as rhs_jit_fhn
from typing import Union
from itertools import chain
from numba import cuda
import numpy as np


class ThreeConnectedNeurons:
    """"Three Connected FitzHugh-Nagumo system"""
    a   : float
    b   : float
    tau1: Union[int, float]
    tau2: Union[int, float]
    v   : Union[int, float]
    S   : Union[list[int], list[float]]
    G   : Union[list[list[int]], list[list[float]]]
    module_name = 'src.systems.ThreeConnectedNeurons'

    def __init__(self, a, b, tau1, tau2, v, S, G):
        self.param_names = ['a', 'b', 'tau1', 'tau2', 'v', 'S', 'G']

        assert isinstance(a, float), "'a' must be float"
        assert isinstance(b, float), "'b' must be float"

        assert isinstance(tau1, float) or isinstance(tau1, int), "'tau1' must be float or int"
        assert tau1 >= 0., "'tau1' must be greater than zero!" # TODO: check for non-positive taus
        assert isinstance(tau2, float) or isinstance(tau1, int), "'tau2' must be float or int"
        assert tau2 >= 0., "'tau2' must be greater than zero!"

        assert isinstance(v, float) or isinstance(v, int), "'v' must be float or int"
        assert isinstance(S, list) and all(isinstance(x, float) for x in S) or \
               isinstance(S, list) and all(isinstance(x, int)   for x in S), \
               "'S' must be list of floats or ints"
        assert (isinstance(G, list) and all(isinstance(sublist, list) and all(isinstance(x, float) for x in sublist) for sublist in G)) or \
               (isinstance(G, list) and all(isinstance(sublist, list) and all(isinstance(x, int)   for x in sublist) for sublist in G)), \
               "'G' must be list of lists of floats or ints"

        self.a = a
        self.b = b
        self.tau1 = tau1
        self.tau2 = tau2
        # потенциал реверсии
        self.v = v
        # матрица 3Х3 коэффициентов синапт. связи между нейронами
        self.G = G
        # G[0][0], G[1][1], G[2][2] = 0, 0, 0
        # у каждого i - го нейрона свой внешний стимул (Si)
        # S = [S1, S2, S3]
        self.S = S


    def getSystem(self, t, X):
        # X  = [x1, y1, z1, x2, y2, z2, x3, y3, z3]
        dz = [0, 0, 0]
        dx = [0, 0, 0]
        dy = [0, 0, 0]
        a, b, S = self.a, self.b, self.S
        tau1, tau2, v = self.tau1, self.tau2, self.v
        G = self.G

        funs = [FitzHughNagumo(a, b, s) for s in S]

        xs = X[0::3]
        ys = X[1::3]
        zs = X[2::3]

        fhsRhs = [f.getSystem(0.0, xy) for f, xy in zip(funs, zip(xs, ys))]
        # tau * dzi = сумма по j от 1 до 3( g(ij) * F(xj) ) - zi
        for i in range(0, 3):
            gij = 0
            for j in range(0, 3):
                gij += np.heaviside(xs[j], 0) * G[i][j]
            dz[i] = (gij - zs[i]) / tau2

        for i, rhs in enumerate(fhsRhs):
            dx[i] = (rhs[0] - zs[i] * (xs[i] - v)) / tau1
            dy[i] = rhs[1]

        # это вектор, содержащий значения [dx1, dy1, dz1, dx2, dy2, dz2, dx3, dy3, dz3]
        dX = list(chain.from_iterable(zip(dx, dy, dz)))
        return dX


    def addGrid(self, grid):
        self.grid = dict()
        for parameter_name, parameter_values in grid.items():
            if not hasattr(self, parameter_name):
                raise RuntimeError(f"Class {self.__class__.__name__} doesn't have parameter {parameter_name}")

            start_param         = getattr(self, parameter_name)
            left_side_interval  = parameter_values['interval'][0]
            right_side_interval = parameter_values['interval'][0]

            self.grid[parameter_name] = np.arange(start_param - left_side_interval, start_param + right_side_interval + 1, parameter_values['step'])


    # Helper function to get parameters. Maybe should be in some util.py file
    def flatten_params(self, params):
        flat_list = []
        for param in params:
            if isinstance(param, (list, tuple)):
                flat_list.extend(self.flatten_params(param))
            else:
                flat_list.append(param)
        return flat_list


    def getParameters(self):
        params = self.flatten_params((self.a, self.b, self.tau1, self.tau2, self.v, self.S, self.G))
        return params


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


    def event_extr(self, t, X):
        rhs = self.getSystem(t, X)
        return rhs[1::3]


@cuda.jit
def rhs_jit(X, params, dX):
    # X expected to be length 9: [x1, y1, z1, x2, y2, z2, x3, y3, z3]
    # dX same length 9 output
    a = params[0]
    b = params[1]
    tau1 = params[2]
    tau2 = params[3]
    v = params[4]
    S = params[5:7]
    G = params[8:16]

    xs = cuda.local.array(3, dtype=np.float64)
    ys = cuda.local.array(3, dtype=np.float64)
    zs = cuda.local.array(3, dtype=np.float64)

    dx = cuda.local.array(3, dtype=np.float64)
    dy = cuda.local.array(3, dtype=np.float64)
    dz = cuda.local.array(3, dtype=np.float64)

    for i in range(3):
        xs[i] = X[3 * i]
        ys[i] = X[3 * i + 1]
        zs[i] = X[3 * i + 2]

    for i in range(3):
        local_dX = cuda.local.array(2, dtype=np.float64)
        xy = cuda.local.array(2, dtype=np.float64)
        xy[0] = xs[i]
        xy[1] = ys[i]
        rhs_jit_fhn(xy, a, b, S[i], local_dX)
        dx[i] = (local_dX[0] - zs[i] * (xs[i] - v)) / tau1
        dy[i] = local_dX[1]

    for i in range(3):
        gij = 0.
        for j in range(3):
            heav = 1. if xs[j] >= 0. else 0.
            gij += heav * G[i * 3 + j]
        dz[i] = (gij - zs[i]) / tau2

    for i in range(3):
        dX[3 * i] = dx[i]
        dX[3 * i + 1] = dy[i]
        dX[3 * i + 2] = dz[i]