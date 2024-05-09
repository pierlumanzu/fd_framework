import numpy as np

from algorithms.gradient_based.fd import FD
from direction_solvers.direction_solver_factory import DirectionSolverFactory
from problems.extended_problem import ExtendedProblem


class FD_N(FD):

    def __init__(self,
                 max_iter: int, max_time: float, max_f_evals: int,
                 verbose: bool, verbose_interspace: int,
                 plot_pareto_front: bool, plot_pareto_solutions: bool, plot_dpi: int,
                 theta_tol: float, qth_quantile: float,
                 gurobi_method: int, gurobi_verbose: bool,
                 rho: float, gamma_1: float, gamma_2: float, eps_hv: float,
                 ALS_alpha_0: float, ALS_delta: float, ALS_beta: float, ALS_min_alpha: float):

        FD.__init__(self,
                    max_iter, max_time, max_f_evals,
                    verbose, verbose_interspace,
                    plot_pareto_front, plot_pareto_solutions, plot_dpi,
                    theta_tol, qth_quantile,
                    gurobi_method, gurobi_verbose,
                    gamma_1, gamma_2, eps_hv,
                    ALS_alpha_0, ALS_delta, ALS_beta, ALS_min_alpha)

        self.__newton_direction_solver = DirectionSolverFactory.get_direction_calculator('N_DD', gurobi_method, gurobi_verbose)
        self.__rho = rho

    def check_hessians_eigenvalues(self, Hess: np.array):
        eigenvals, _ = np.linalg.eig(Hess)
        
        min_eigenvals = np.amin(eigenvals.real, axis=1)
        multipliers = np.where(min_eigenvals > self.__rho, 0, - min_eigenvals + self.__rho)

        if min(min_eigenvals + multipliers) < self.__rho:
            return None
    
        return Hess + multipliers[:, np.newaxis, np.newaxis] * np.eye(Hess.shape[1])

    def compute_direction(self, problem: ExtendedProblem, x_p: np.array, point_idx: int):
        J_p = problem.evaluate_functions_jacobian(x_p)
        self.add_to_stopping_condition_current_value('max_f_evals', problem.n)

        H_p = self.check_hessians_eigenvalues(problem.evaluate_functions_hessians(x_p))
        self.add_to_stopping_condition_current_value('max_f_evals', problem.n**2)

        if H_p is not None:
            d_n, theta_n = self.__newton_direction_solver.compute_direction(problem, J_p, x_p, H_p)
        else:
            d_n, theta_n = np.zeros(problem.n), 0

        return d_n, theta_n, {'J_p': J_p}
