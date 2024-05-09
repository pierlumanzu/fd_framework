from direction_solvers.descent_direction.sd_dd import SD_DD
from direction_solvers.descent_direction.n_dd import N_DD
from direction_solvers.descent_direction.lmqn_dd import LMQN_DD
from direction_solvers.descent_direction.bb_dd import BB_DD


class DirectionSolverFactory:

    @staticmethod
    def get_direction_calculator(direction_type: str, gurobi_method: int, gurobi_verbose: bool):

        if direction_type == 'SD_DD':
            return SD_DD(gurobi_method, gurobi_verbose)

        elif direction_type == 'N_DD':
            return N_DD(gurobi_method, gurobi_verbose)

        elif direction_type == 'LMQN_DD':
            return LMQN_DD(gurobi_method, gurobi_verbose)

        elif direction_type == 'BB_DD':
            return BB_DD(gurobi_method, gurobi_verbose)

        raise NotImplementedError
