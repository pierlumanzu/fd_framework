import numpy as np
from gurobipy import Model, GRB, GurobiError

from nsma.direction_solvers.descent_direction.dds import DDS
from nsma.direction_solvers.gurobi_settings import GurobiSettings

from problems.extended_problem import ExtendedProblem


class LMQN_DD(DDS, GurobiSettings):

    def __init__(self, gurobi_method: int, gurobi_verbose: bool):
        DDS.__init__(self)
        GurobiSettings.__init__(self, gurobi_method, gurobi_verbose)

    def compute_direction(self, problem: ExtendedProblem, Jac: np.array, x_p: np.array = None, HG_results: np.array = None):
        assert x_p is None
        assert HG_results is not None

        m, n = Jac.shape

        P = np.dot(Jac, HG_results)
        reset = False

        if np.isinf(P).any() or np.isnan(P).any():
            return np.zeros(n), 0, np.zeros(m), reset

        if np.min(np.linalg.eigvals(P)) < -5 * np.finfo(np.float32).eps ** 0.5:
            if self._gurobi_verbose:
                print('Reset')
            reset = True
            P = np.dot(Jac, Jac.T)

        try:
            model = Model("Limited Memory Quasi-Newton Descent Direction")

            model.setParam("OutputFlag", self._gurobi_verbose)
            model.setParam("Method", self._gurobi_method)

            lam = model.addMVar(m, lb=0, ub=np.inf, name="lambda")

            obj = 1/2 * (lam @ P @ lam)
            model.setObjective(obj)

            model.addConstr(np.ones(m) @ lam == 1, name='Lambda constraint')

            model.update()

            for i in range(m):
                lam[i].start = 0.

            model.optimize()

            if model.Status == GRB.OPTIMAL:
                return -np.dot(HG_results, lam.x).flatten(), -model.getObjective().getValue(), lam.x, reset
            else:
                return np.zeros(n), 0, np.zeros(m), reset

        except GurobiError:
            if self._gurobi_verbose:
                print('Gurobi Error')
            return np.zeros(n), 0, np.zeros(m), reset

        except AttributeError:
            if self._gurobi_verbose:
                print('Attribute Error')
            return np.zeros(n), 0, np.zeros(m), reset
