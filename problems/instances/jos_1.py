import numpy as np
import tensorflow as tf

from problems.extended_problem import ExtendedProblem

'''
For more details about the JOS problems, the user is referred to 

Jin, Y., Olhofer, M., Sendhoff, B.: Dynamic weighted aggregation for evo-
lutionary multi-objective optimization: Why does it work and how? In:
Proceedings of the Genetic and Evolutionary Computation Conference,
pp. 1042–1049 (2001).
'''


class JOS_1(ExtendedProblem):

    def __init__(self, n: int):
        assert n >= 1

        ExtendedProblem.__init__(self, n)

        self.set_objectives([
            tf.reduce_sum([self._z[i] ** 2 for i in range(self.n)]) / self.n,
            tf.reduce_sum([(self._z[i] - 2) ** 2 for i in range(self.n)]) / self.n
        ])

        self.filtered_lb_for_ini = -100 * np.ones(self.n)
        self.filtered_ub_for_ini = 100 * np.ones(self.n)

    @staticmethod
    def family_name():
        return 'JOS'

    @staticmethod
    def name():
        return 'JOS_1'
