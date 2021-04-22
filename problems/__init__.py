from problems.tsp.problem_tsp import TSP
from problems.vrp.problem_vrp import CVRP


def load_problem(name):
    problem = {
        'tsp': TSP,
        'cvrp': CVRP
    }.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem