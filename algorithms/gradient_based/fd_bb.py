import numpy as np

from algorithms.gradient_based.fd import FD
from direction_solvers.direction_solver_factory import DirectionSolverFactory
from problems.extended_problem import ExtendedProblem


class FD_BB(FD):

    def __init__(self,
                 max_iter: int, max_time: float, max_f_evals: int,
                 verbose: bool, verbose_interspace: int,
                 plot_pareto_front: bool, plot_pareto_solutions: bool, plot_dpi: int,
                 theta_tol: float, qth_quantile: float,
                 gurobi_method: int, gurobi_verbose: bool,
                 alpha_min: float, alpha_max: float, gamma_1: float, gamma_2: float, eps_hv: float,
                 ALS_alpha_0: float, ALS_delta: float, ALS_beta: float, ALS_min_alpha: float):

        FD.__init__(self,
                    max_iter, max_time, max_f_evals,
                    verbose, verbose_interspace,
                    plot_pareto_front, plot_pareto_solutions, plot_dpi,
                    theta_tol, qth_quantile,
                    gurobi_method, gurobi_verbose,
                    gamma_1, gamma_2, eps_hv,
                    ALS_alpha_0, ALS_delta, ALS_beta, ALS_min_alpha)

        self.__bb_direction_solver = DirectionSolverFactory.get_direction_calculator('BB_DD', gurobi_method, gurobi_verbose)
        self.__alpha_min = alpha_min
        self.__alpha_max = alpha_max

    def eval_alpha(self, p_ini: np.array, p_fin: np.array, J_ini: np.array, J_fin: np.array):
        m, _ = J_ini.shape

        s_p = p_fin - p_ini
        y_p = J_fin - J_ini

        norm_s_p = np.linalg.norm(s_p)
        scalar_prods = y_p @ s_p

        alpha_p = np.full(m, self.__alpha_min) 

        sp_greater = np.where(scalar_prods > 1e-6)[0]
        alpha_p[sp_greater] = np.maximum(self.__alpha_min, np.minimum(scalar_prods[sp_greater] / norm_s_p ** 2, self.__alpha_max))

        sp_smaller = np.where(scalar_prods < -1e-6)[0]
        alpha_p[sp_smaller] = np.maximum(self.__alpha_min, np.minimum(np.linalg.norm(y_p[sp_smaller], axis=1) / norm_s_p, self.__alpha_max))

        return alpha_p

    def initialize_points_dict(self, problem: ExtendedProblem, len_p_list: int):
        self._points_dict['alpha_list'] = np.ones((len_p_list, problem.m))

    def update_points_dict(self, problem: ExtendedProblem, old_x_p: np.array, x_p: np.array, J_p: np.array, efficient_points_idx: np.array, args: dict):
        self._points_dict['alpha_list'] = np.concatenate((self._points_dict['alpha_list'][efficient_points_idx, :], self.eval_alpha(old_x_p, x_p, args['J_p'], J_p).reshape(1, -1)), axis=0)

    def add_new_element_to_points_dict(self, problem: ExtendedProblem, efficient_points_idx: np.array):
        self._points_dict['alpha_list'] = np.concatenate((self._points_dict['alpha_list'][efficient_points_idx, :], np.ones((1, problem.m))), axis=0)
        
    def compute_direction(self, problem: ExtendedProblem, x_p: np.array, point_idx: int):
        J_p = problem.evaluate_functions_jacobian(x_p)
        self.add_to_stopping_condition_current_value('max_f_evals', problem.n)

        d_bb, theta_bb = self.__bb_direction_solver.compute_direction(problem, J_p, x_p, self._previous_points_dict['alpha_list'][point_idx, :])
        return d_bb, theta_bb, {'J_p': J_p}
