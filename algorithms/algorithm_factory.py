import numpy as np

from algorithms.gradient_based.fd_sd import FD_SD
from algorithms.gradient_based.fd_n import FD_N
from algorithms.gradient_based.fd_lmqn import FD_LMQN
from algorithms.gradient_based.fd_bb import FD_BB


class AlgorithmFactory:

    @staticmethod
    def get_algorithm(algorithm_name: str, **kwargs):

        general_settings = kwargs['general_settings']
        algorithms_settings = kwargs['algorithms_settings']

        if 'FD' in algorithm_name:
            FD_settings = algorithms_settings['FD']
            DDS_settings = kwargs['DDS_settings']
            ALS_settings = kwargs['ALS_settings']

            if algorithm_name == 'FD_SD':
                return FD_SD(general_settings['max_iter'],
                             general_settings['max_time'],
                             general_settings['max_f_evals'],
                             general_settings['verbose'],
                             general_settings['verbose_interspace'],
                             general_settings['plot_pareto_front'],
                             general_settings['plot_pareto_solutions'],
                             general_settings['plot_dpi'],
                             FD_settings['theta_tol'],
                             FD_settings['qth_quantile'],
                             DDS_settings['gurobi_method'],
                             DDS_settings['gurobi_verbose'],
                             FD_settings['eps_hv'],
                             ALS_settings['alpha_0'],
                             ALS_settings['delta'],
                             ALS_settings['beta'],
                             ALS_settings['min_alpha'])
        
            elif algorithm_name == 'FD_N':
                FD_N_settings = algorithms_settings[algorithm_name]

                return FD_N(general_settings['max_iter'],
                            general_settings['max_time'],
                            general_settings['max_f_evals'],
                            general_settings['verbose'],
                            general_settings['verbose_interspace'],
                            general_settings['plot_pareto_front'],
                            general_settings['plot_pareto_solutions'],
                            general_settings['plot_dpi'],
                            FD_settings['theta_tol'],
                            FD_settings['qth_quantile'],
                            DDS_settings['gurobi_method'],
                            DDS_settings['gurobi_verbose'],
                            FD_N_settings['rho'],
                            FD_settings['gamma_1'],
                            FD_settings['gamma_2'],
                            FD_settings['eps_hv'],
                            ALS_settings['alpha_0'],
                            ALS_settings['delta'],
                            ALS_settings['beta'],
                            ALS_settings['min_alpha'])

            elif algorithm_name == 'FD_LMQN':
                FD_LMQN_settings = algorithms_settings[algorithm_name]

                return FD_LMQN(general_settings['max_iter'],
                               general_settings['max_time'],
                               general_settings['max_f_evals'],
                               general_settings['verbose'],
                               general_settings['verbose_interspace'],
                               general_settings['plot_pareto_front'],
                               general_settings['plot_pareto_solutions'],
                               general_settings['plot_dpi'],
                               FD_settings['theta_tol'],
                               FD_settings['qth_quantile'],
                               DDS_settings['gurobi_method'],
                               DDS_settings['gurobi_verbose'],
                               FD_LMQN_settings['max_cor'],
                               FD_settings['gamma_1'],
                               FD_settings['gamma_2'],
                               FD_settings['eps_hv'],
                               ALS_settings['alpha_0'],
                               ALS_settings['delta'],
                               ALS_settings['beta'],
                               ALS_settings['min_alpha'])
        
            elif algorithm_name == 'FD_BB':
                FD_BB_settings = algorithms_settings[algorithm_name]

                return FD_BB(general_settings['max_iter'],
                             general_settings['max_time'],
                             general_settings['max_f_evals'],
                             general_settings['verbose'],
                             general_settings['verbose_interspace'],
                             general_settings['plot_pareto_front'],
                             general_settings['plot_pareto_solutions'],
                             general_settings['plot_dpi'],
                             FD_settings['theta_tol'],
                             FD_settings['qth_quantile'],
                             DDS_settings['gurobi_method'],
                             DDS_settings['gurobi_verbose'],
                             FD_BB_settings['alpha_min'],
                             FD_BB_settings['alpha_max'],
                             FD_settings['gamma_1'],
                             FD_settings['gamma_2'],
                             FD_settings['eps_hv'],
                             ALS_settings['alpha_0'],
                             ALS_settings['delta'],
                             ALS_settings['beta'],
                             ALS_settings['min_alpha'])

        raise NotImplementedError
