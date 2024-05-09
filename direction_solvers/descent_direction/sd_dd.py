import numpy as np
from gurobipy import Model, GRB, GurobiError

from nsma.direction_solvers.descent_direction.dds import DDS
from nsma.direction_solvers.gurobi_settings import GurobiSettings

from problems.extended_problem import ExtendedProblem


class SD_DD(DDS, GurobiSettings):

    def __init__(self, gurobi_method: int, gurobi_verbose: bool):
        DDS.__init__(self)
        GurobiSettings.__init__(self, gurobi_method, gurobi_verbose)

    def compute_direction(self, problem: ExtendedProblem, Jac: np.array, x_p: np.array = None):
        assert x_p is not None

        m, n = Jac.shape

        if np.isinf(Jac).any() or np.isnan(Jac).any():
            return np.zeros(n), 0
        
        if len(Jac) == 1:
            return -Jac.flatten(), -1/2 * np.linalg.norm(Jac.flatten())**2
        
        elif len(Jac) == 2:
            prod_jac_1_2 = np.dot(Jac[0, :], Jac[1, :])
            squared_norm_jac_1 = np.linalg.norm(Jac[0, :])**2

            if prod_jac_1_2 >= squared_norm_jac_1:
                lambda_1 = 1
            
            else:
                squared_norm_jac_2 = np.linalg.norm(Jac[1, :])**2

                if prod_jac_1_2 >= squared_norm_jac_2:
                    lambda_1 = 0
                
                else:
                    lambda_1 = (squared_norm_jac_2 - prod_jac_1_2) / (squared_norm_jac_1 - 2 * prod_jac_1_2 + squared_norm_jac_2)

            d_p = -(lambda_1 * Jac[0, :] + (1-lambda_1) * Jac[1, :])

            return d_p, np.amax(np.dot(Jac, d_p)) + 1/2 * np.linalg.norm(d_p) ** 2

        else: 
            try:
                model = Model("Steepest Descent Direction")

                model.setParam("OutputFlag", self._gurobi_verbose)
                model.setParam("Method", self._gurobi_method)

                z = model.addMVar(n, lb=-np.inf, ub=np.inf, name="z")
                beta = model.addMVar(1, lb=-np.inf, ub=0., name="beta")

                obj = beta + 0.5 * (z - x_p) @ (z - x_p)
                model.setObjective(obj)

                for j in range(m):
                    model.addConstr(Jac[j, :] @ (z - x_p) <= beta, name='Jacobian constraint n°{}'.format(j))

                model.update()

                for i in range(n):
                    z[i].start = float(x_p[i])
                beta.start = 0.

                model.optimize()

                if model.Status == GRB.OPTIMAL:
                    return z.x - x_p, model.getObjective().getValue()
                else:
                    return np.zeros(n), 0

            except GurobiError:
                if self._gurobi_verbose:
                    print('Gurobi Error')
                return np.zeros(n), 0

            except AttributeError:
                if self._gurobi_verbose:
                    print('Attribute Error')
                return np.zeros(n), 0
