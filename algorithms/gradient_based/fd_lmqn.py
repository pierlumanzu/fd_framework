import numpy as np

from algorithms.gradient_based.fd import FD
from direction_solvers.direction_solver_factory import DirectionSolverFactory
from problems.extended_problem import ExtendedProblem


class FD_LMQN(FD):

    def __init__(self,
                 max_iter: int, max_time: float, max_f_evals: int,
                 verbose: bool, verbose_interspace: int,
                 plot_pareto_front: bool, plot_pareto_solutions: bool, plot_dpi: int,
                 theta_tol: float, qth_quantile: float,
                 gurobi_method: int, gurobi_verbose: bool,
                 max_cor: int, gamma_1: float, gamma_2: float, eps_hv: float,
                 ALS_alpha_0: float, ALS_delta: float, ALS_beta: float, ALS_min_alpha: float):

        FD.__init__(self,
                    max_iter, max_time, max_f_evals,
                    verbose, verbose_interspace,
                    plot_pareto_front, plot_pareto_solutions, plot_dpi,
                    theta_tol, qth_quantile,
                    gurobi_method, gurobi_verbose,
                    gamma_1, gamma_2, eps_hv,
                    ALS_alpha_0, ALS_delta, ALS_beta, ALS_min_alpha)

        self.__lm_qnwt_direction_solver = DirectionSolverFactory.get_direction_calculator('LMQN_DD', gurobi_method, gurobi_verbose)
        self.__max_cor = max_cor

    def hg_procedure(self, Jac: np.array, s_p: np.array, u_p: np.array, ro_p: np.array):
        m, n = Jac.shape
        n_current_memory_vectors = np.sum(np.sum(np.isnan(s_p), axis=0) == 0)
        
        q = Jac

        alpha = np.empty((m, n_current_memory_vectors))

        for i in range(n_current_memory_vectors - 1, -1, -1):
            alpha[:, i] = ro_p[i] * np.dot(q, s_p[:, i])
            q = q - np.dot(alpha[:, i][:, None], u_p[:, i][None, :])

        r = np.dot(np.array([1.]) * np.eye(n), q.T)

        for i in range(n_current_memory_vectors):
            beta = ro_p[i] * np.dot(u_p[:, i][None, :], r)
            r = r + (np.dot(s_p[:, i][:, None], alpha[:, i][None, :] - beta))

        return r

    def update_memory_vectors(self, p_ini: np.array, p_fin: np.array, J_ini: np.array, J_fin: np.array, lam: np.array, s_p: np.array, u_p: np.array, ro_p: np.array, reset: bool, n: int):
        
        if reset:
            s_p = np.full_like(s_p, np.nan)
            u_p = np.full_like(u_p, np.nan)
            ro_p = np.full_like(ro_p, np.nan)
        
        n_current_memory_vectors = np.sum(np.sum(np.isnan(s_p), axis=0) == 0)

        if n_current_memory_vectors == self.__max_cor:
            s_p = np.roll(s_p, -1, axis=1)
            u_p = np.roll(u_p, -1, axis=1)
            ro_p = np.roll(ro_p, -1)
            n_current_memory_vectors = n_current_memory_vectors - 1

        s_p[:, n_current_memory_vectors] = p_fin - p_ini
        u_p[:, n_current_memory_vectors] = np.dot(lam, J_fin - J_ini)

        ro_tmp_den = np.dot(u_p[:, n_current_memory_vectors], s_p[:, n_current_memory_vectors])
        if ro_tmp_den <= 0:
            u_p[:, n_current_memory_vectors] = np.dot(lam, J_fin[np.argmax(np.dot(J_fin, s_p[:, n_current_memory_vectors])), :] - J_ini)
            ro_tmp_den = np.dot(u_p[:, n_current_memory_vectors], s_p[:, n_current_memory_vectors])
        ro_p[n_current_memory_vectors] = 1 / ro_tmp_den

        return s_p, u_p, ro_p

    def initialize_points_dict(self, problem: ExtendedProblem, len_p_list: int):
        self._points_dict['s_list'] = np.full((len_p_list, problem.n, self.__max_cor), np.nan)
        self._points_dict['u_list'] = np.full((len_p_list, problem.n, self.__max_cor), np.nan)
        self._points_dict['ro_list'] = np.full((len_p_list, self.__max_cor), np.nan)

    def update_points_dict(self, problem: ExtendedProblem, old_x_p: np.array, x_p: np.array, J_p: np.array, efficient_points_idx: np.array, args: dict):
        new_s_p, new_u_p, new_ro_p = self.update_memory_vectors(old_x_p, x_p, 
                                                                args['J_p'], J_p, 
                                                                args['lam'], 
                                                                self._previous_points_dict['s_list'][args['point_idx'], :, :], 
                                                                self._previous_points_dict['u_list'][args['point_idx'], :, :], 
                                                                self._previous_points_dict['ro_list'][args['point_idx'], :], 
                                                                args['reset'] or args['ud_string'] == 'sd', 
                                                                problem.n) 

        self._points_dict['s_list'] = np.concatenate((self._points_dict['s_list'][efficient_points_idx, :, :], new_s_p[np.newaxis, :, :]), axis=0)
        self._points_dict['u_list'] = np.concatenate((self._points_dict['u_list'][efficient_points_idx, :, :], new_u_p[np.newaxis, :, :]), axis=0)
        self._points_dict['ro_list'] = np.concatenate((self._points_dict['ro_list'][efficient_points_idx, :], new_ro_p.reshape(1, -1)), axis=0)

    def add_new_element_to_points_dict(self, problem: ExtendedProblem, efficient_points_idx: np.array):
        self._points_dict['s_list'] = np.concatenate((self._points_dict['s_list'][efficient_points_idx, :, :], np.full((1, problem.n, self.__max_cor), np.nan)), axis=0)
        self._points_dict['u_list'] = np.concatenate((self._points_dict['u_list'][efficient_points_idx, :, :], np.full((1, problem.n, self.__max_cor), np.nan)), axis=0)
        self._points_dict['ro_list'] = np.concatenate((self._points_dict['ro_list'][efficient_points_idx, :], np.full((1, self.__max_cor), np.nan)), axis=0)
        
    def compute_direction(self, problem: ExtendedProblem, x_p: np.array, point_idx: int):
        J_p = problem.evaluate_functions_jacobian(x_p)
        self.add_to_stopping_condition_current_value('max_f_evals', problem.n)

        d_qn, theta_qn, l_qn, reset = self.__lm_qnwt_direction_solver.compute_direction(problem, 
                                                                                        J_p, 
                                                                                        None, 
                                                                                        self.hg_procedure(J_p, 
                                                                                                          self._previous_points_dict['s_list'][point_idx, :, :], 
                                                                                                          self._previous_points_dict['u_list'][point_idx, :, :], 
                                                                                                          self._previous_points_dict['ro_list'][point_idx, :]))
        return d_qn, theta_qn, {'point_idx': point_idx, 'J_p': J_p, 'lam': l_qn, 'reset': reset}
