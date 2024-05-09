import numpy as np

from algorithms.gradient_based.fd import FD
from problems.extended_problem import ExtendedProblem


class FD_SD(FD):

    def __init__(self,
                 max_iter: int, max_time: float, max_f_evals: int,
                 verbose: bool, verbose_interspace: int,
                 plot_pareto_front: bool, plot_pareto_solutions: bool, plot_dpi: int,
                 theta_tol: float, qth_quantile: float,
                 gurobi_method: int, gurobi_verbose: bool,
                 eps_hv: float,
                 ALS_alpha_0: float, ALS_delta: float, ALS_beta: float, ALS_min_alpha: float):

        FD.__init__(self,
                    max_iter, max_time, max_f_evals,
                    verbose, verbose_interspace,
                    plot_pareto_front, plot_pareto_solutions, plot_dpi,
                    theta_tol, qth_quantile,
                    gurobi_method, gurobi_verbose,
                    None, None, eps_hv,
                    ALS_alpha_0, ALS_delta, ALS_beta, ALS_min_alpha)

    def compute_direction(self, problem: ExtendedProblem, x_p: np.array, point_idx: int):
        return np.zeros(problem.n), 0, {}
