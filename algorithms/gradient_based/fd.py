import numpy as np
from abc import ABC, abstractmethod
from itertools import chain, combinations
from pymoo.indicators.hv import HV
import time

from nsma.algorithms.gradient_based.gradient_based_algorithm import GradientBasedAlgorithm
from nsma.algorithms.genetic.genetic_utils.general_utils import calc_crowding_distance
from nsma.general_utils.pareto_utils import pareto_efficient

from direction_solvers.direction_solver_factory import DirectionSolverFactory
from line_searches.line_search_factory import LineSearchFactory
from problems.extended_problem import ExtendedProblem


class FD(GradientBasedAlgorithm, ABC):

    def __init__(self,
                 max_iter: int, max_time: float, max_f_evals: int,
                 verbose: bool, verbose_interspace: int,
                 plot_pareto_front: bool, plot_pareto_solutions: bool, plot_dpi: int,
                 theta_tol: float, qth_quantile: float,
                 gurobi_method: int, gurobi_verbose: bool,
                 gamma_1: float, gamma_2: float, eps_hv: float,
                 ALS_alpha_0: float, ALS_delta: float, ALS_beta: float, ALS_min_alpha: float):

        GradientBasedAlgorithm.__init__(self,
                                        max_iter, max_time, max_f_evals,
                                        verbose, verbose_interspace,
                                        plot_pareto_front, plot_pareto_solutions, plot_dpi,
                                        theta_tol,
                                        True, gurobi_method, gurobi_verbose,
                                        ALS_alpha_0, ALS_delta, ALS_beta, ALS_min_alpha, name_ALS='BoundconstrainedFrontALS')

        self._direction_solver = DirectionSolverFactory.get_direction_calculator('SD_DD', gurobi_method, gurobi_verbose)
        self._single_point_line_search = LineSearchFactory.get_line_search('MOALS', ALS_alpha_0, ALS_delta, ALS_beta, ALS_min_alpha)

        self.add_stopping_condition('hv_rel_diff', eps_hv, np.inf, smaller_value_required=True, equal_required=True)

        self._qth_quantile = qth_quantile
        self._gamma_1 = gamma_1
        self._gamma_2 = gamma_2

        self._points_dict = {}
        self._previous_points_dict = {}
        
    def search(self, p_list: np.array, f_list: np.array, problem: ExtendedProblem):

        self.update_stopping_condition_current_value('max_time', time.time())

        efficient_points_idx = pareto_efficient(f_list)
        f_list = f_list[efficient_points_idx, :]
        p_list = p_list[efficient_points_idx, :]
        self.show_figure(p_list, f_list)

        for idx_p, p in enumerate(p_list):
            d_p, theta_p = self._direction_solver.compute_direction(problem, problem.evaluate_functions_jacobian(p), p)
            self.add_to_stopping_condition_current_value('max_f_evals', problem.n)

            d_list = d_p.reshape(1, -1) if idx_p == 0 else np.concatenate((d_list, d_p.reshape(1, -1)), axis=0)
            theta_list = np.array([theta_p]) if idx_p == 0 else np.concatenate((theta_list, np.array([theta_p])))

        self.initialize_points_dict(problem, len(p_list))

        crowding_quantile = np.inf  

        while not self.evaluate_stopping_conditions():

            self.output_data(f_list, crowding_quantile=crowding_quantile, hv_rel_diff=self.get_stopping_condition_current_value('hv_rel_diff'))
            self.add_to_stopping_condition_current_value('max_iter', 1)

            previous_p_list = np.copy(p_list)
            previous_f_list = np.copy(f_list)
            previous_d_list = np.copy(d_list)
            previous_theta_list = np.copy(theta_list)
            
            self._previous_points_dict = self._points_dict.copy()
                
            crowding_list = calc_crowding_distance(previous_f_list)
            is_finite_idx = np.isfinite(crowding_list)
            crowding_quantile = np.quantile(crowding_list[is_finite_idx], self._qth_quantile) if len(crowding_list[is_finite_idx]) > 0 else np.inf

            if len(p_list) == 1:
                sorted_idx = np.array([0])
            else:
                argmin_theta_list = np.argmin(theta_list)
                sorted_idx_crowding_list = np.flip(np.argsort(crowding_list))
                sorted_idx = np.concatenate((
                    np.array([argmin_theta_list]), 
                    np.delete(sorted_idx_crowding_list, np.where(sorted_idx_crowding_list == argmin_theta_list))
                ))

            previous_p_list = previous_p_list[sorted_idx, :]
            previous_f_list = previous_f_list[sorted_idx, :]
            previous_d_list = previous_d_list[sorted_idx, :]
            previous_theta_list = previous_theta_list[sorted_idx]
            crowding_list = crowding_list[sorted_idx]

            for ppd_key in self._previous_points_dict.keys():
                self._previous_points_dict[ppd_key] = self._previous_points_dict[ppd_key][sorted_idx]        

            for point_idx in range(len(previous_p_list)):

                if self.evaluate_stopping_conditions():
                    break
                if self.exists_dominating_point(previous_f_list[point_idx, :], f_list):
                    continue

                x_p = previous_p_list[point_idx, :]
                f_p = previous_f_list[point_idx, :]
                
                d_sd, theta_sd = previous_d_list[point_idx, :], previous_theta_list[point_idx]

                if not self.evaluate_stopping_conditions() and theta_sd < self._theta_tol:
                
                    d_fd, theta_fd, outs = self.compute_direction(problem, x_p, point_idx)
 
                    if not self.evaluate_stopping_conditions():
                        
                        if theta_fd < self._theta_tol:

                            assert 'J_p' in outs.keys()

                            max_scalar_prod = np.amax(np.dot(outs['J_p'], d_fd))

                            norm_dfd = np.linalg.norm(d_fd)
                            norm_dsd = np.linalg.norm(d_sd)

                            if max_scalar_prod <= - self._gamma_1 * norm_dsd**2 and norm_dfd <= self._gamma_2 * norm_dsd:
                                outs['ud_string'] = 'fd'
                                new_x_p, new_f_p, _, ls_f_evals = self._single_point_line_search.search(problem, x_p, f_p, d_fd, max_scalar_prod)
                            else:
                                outs['ud_string'] = 'sd'
                                new_x_p, new_f_p, _, ls_f_evals = self._single_point_line_search.search(problem, x_p, f_p, d_sd, theta_sd - 1/2 * norm_dsd ** 2) 
                        
                        else:
                            outs['ud_string'] = 'sd'
                            new_x_p, new_f_p, _, ls_f_evals = self._single_point_line_search.search(problem, x_p, f_p, d_sd, theta_sd - 1/2 * np.linalg.norm(d_sd) ** 2) 
                            
                        self.add_to_stopping_condition_current_value('max_f_evals', ls_f_evals)

                        if not self.evaluate_stopping_conditions() and new_x_p is not None:

                            efficient_points_idx = self.fast_non_dominated_filter(f_list, new_f_p.reshape(1, -1))

                            p_list = np.concatenate((p_list[efficient_points_idx, :],
                                                     new_x_p.reshape(1, -1)), axis=0)

                            f_list = np.concatenate((f_list[efficient_points_idx, :],
                                                     new_f_p.reshape(1, -1)), axis=0)
                            
                            old_x_p = np.copy(x_p)

                            x_p = np.copy(new_x_p)
                            f_p = np.copy(new_f_p)
                            
                            J_p = problem.evaluate_functions_jacobian(x_p)
                            self.add_to_stopping_condition_current_value('max_f_evals', problem.n)

                            d_z, theta_z = self._direction_solver.compute_direction(problem, J_p, new_x_p)
                            d_list = np.concatenate((d_list[efficient_points_idx, :], d_z.reshape(1, -1)), axis=0)
                            theta_list = np.concatenate((theta_list[efficient_points_idx], np.array([theta_z])))
                            
                            self.update_points_dict(problem, old_x_p, x_p, J_p, efficient_points_idx, outs) 
                
                elif not self.evaluate_stopping_conditions():
                    J_p = problem.evaluate_functions_jacobian(x_p)
                    self.add_to_stopping_condition_current_value('max_f_evals', problem.n)
               
                for I_k in self.objectives_powerset(problem.m):

                    if self.evaluate_stopping_conditions() or self.exists_dominating_point(f_p, f_list) or crowding_list[point_idx] < crowding_quantile:
                        break

                    partial_d_p, partial_theta_p = self._direction_solver.compute_direction(problem, J_p[I_k, ], x_p)

                    if not self.evaluate_stopping_conditions() and partial_theta_p < self._theta_tol:

                        new_x_p, new_f_p, _, ls_f_evals = self._line_search.search(problem, x_p, f_list, partial_d_p, 0., I=np.arange(problem.m))
                        self.add_to_stopping_condition_current_value('max_f_evals', ls_f_evals)

                        if not self.evaluate_stopping_conditions() and new_x_p is not None:

                            efficient_points_idx = self.fast_non_dominated_filter(f_list, new_f_p.reshape(1, -1))

                            p_list = np.concatenate((p_list[efficient_points_idx, :],
                                                     new_x_p.reshape(1, -1)), axis=0)

                            f_list = np.concatenate((f_list[efficient_points_idx, :],
                                                     new_f_p.reshape(1, -1)), axis=0)

                            d_p, theta_p = self._direction_solver.compute_direction(problem, problem.evaluate_functions_jacobian(new_x_p), new_x_p)
                            self.add_to_stopping_condition_current_value('max_f_evals', problem.n)
                            d_list = np.concatenate((d_list[efficient_points_idx, :], d_p.reshape(1, -1)), axis=0)
                            theta_list = np.concatenate((theta_list[efficient_points_idx], np.array([theta_p])))

                            self.add_new_element_to_points_dict(problem, efficient_points_idx)

            self.update_stopping_condition_current_value('hv_rel_diff', self.check_hypervolume(previous_f_list, f_list))

            self.show_figure(p_list, f_list)

        self.output_data(f_list, crowding_quantile=crowding_quantile, hv_rel_diff=self.get_stopping_condition_current_value('hv_rel_diff'))
        self.close_figure()

        return p_list, f_list, time.time() - self.get_stopping_condition_current_value('max_time'), self.get_stopping_condition_current_value('hv_rel_diff')

    @staticmethod
    def objectives_powerset(m: int):
        s = list(range(m))        
        return list(chain.from_iterable(combinations(s, r) for r in range(1, len(s))))

    @staticmethod
    def exists_dominating_point(f: np.array, f_list: np.array):

        if np.isnan(f).any():
            return True

        n_obj = len(f)

        f = np.reshape(f, (1, n_obj))
        dominance_matrix = f_list - f

        return (np.logical_and(np.sum(dominance_matrix <= 0, axis=1) == n_obj, np.sum(dominance_matrix < 0, axis=1) > 0)).any()

    @staticmethod
    def check_hypervolume(previous_f_list: np.array, f_list: np.array):

        hv_ref_point = np.amax((np.amax(previous_f_list, axis=0), np.amax(f_list, axis=0)), axis=0)
        HV_eval = HV(ref_point=hv_ref_point)
        
        v_i = HV_eval(previous_f_list)
        v_f = HV_eval(f_list)

        if v_i < 1e-6 or (len(previous_f_list) == 1 and len(f_list) > len(previous_f_list)):
            return np.inf
        else:
            return (v_f - v_i) / v_i

    def initialize_points_dict(self, problem: ExtendedProblem, len_p_list: int):
        pass

    def update_points_dict(self, problem: ExtendedProblem, old_x_p: np.array, x_p: np.array, J_p: np.array, efficient_points_idx: np.array, args: dict):
        pass

    def add_new_element_to_points_dict(self, problem: ExtendedProblem, efficient_points_idx: np.array):
        pass

    @abstractmethod
    def compute_direction(self, problem: ExtendedProblem, x_p: np.array, point_idx: int):
        return