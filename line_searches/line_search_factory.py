from line_searches.armijo_type.mo_als import MOALS


class LineSearchFactory:

    @staticmethod
    def get_line_search(line_search_type: str, alpha_0: float, delta: float, beta: float, min_alpha: float):

        if line_search_type == 'MOALS':
            return MOALS(alpha_0, delta, beta, min_alpha)

        raise NotImplementedError
