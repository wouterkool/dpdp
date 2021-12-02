import random
import numpy as np
from scipy.spatial.distance import cdist


# Taken from https://github.com/qcappart/hybrid-cp-rl-solver/blob/master/src/problem/tsptw/environment/tsptw.py
def generate_random_instance(n_city, grid_size, max_tw_gap, max_tw_size,
                             is_integer_instance, seed, fast=True, da_silva_style=False):
    """
    :param n_city: number of cities
    :param grid_size: x-pos/y-pos of cities will be in the range [0, grid_size]
    :param max_tw_gap: maximum time windows gap allowed between the cities constituing the feasible tour
    :param max_tw_size: time windows of cities will be in the range [0, max_tw_size]
    :param is_integer_instance: True if we want the distances and time widows to have integer values
    :param seed: seed used for generating the instance. -1 means no seed (instance is random)
    :return: a feasible TSPTW instance randomly generated using the parameters
    """

    rand = random.Random()

    if seed != -1:
        rand.seed(seed)

    x_coord = [rand.uniform(0, grid_size) for _ in range(n_city)]
    y_coord = [rand.uniform(0, grid_size) for _ in range(n_city)]
    coord = np.array([x_coord, y_coord]).transpose()

    if fast:  # Improved code but could (theoretically) give different results with rounding?
        travel_time = cdist(coord, coord)
        if is_integer_instance:
            travel_time = travel_time.round().astype(np.int)
    else:
        travel_time = []
        for i in range(n_city):

            dist = [float(np.sqrt((x_coord[i] - x_coord[j]) ** 2 + (y_coord[i] - y_coord[j]) ** 2))
                    for j in range(n_city)]

            if is_integer_instance:
                dist = [round(x) for x in dist]

            travel_time.append(dist)

    random_solution = list(range(1, n_city))
    rand.shuffle(random_solution)

    random_solution = [0] + random_solution

    time_windows = np.zeros((n_city, 2))
    time_windows[0, :] = [0, 1000 * grid_size]

    total_dist = 0
    for i in range(1, n_city):

        prev_city = random_solution[i-1]
        cur_city = random_solution[i]

        cur_dist = travel_time[prev_city][cur_city]

        tw_lb_min = time_windows[prev_city, 0] + cur_dist
        total_dist += cur_dist

        if da_silva_style:
            # Style by Da Silva and Urrutia, 2010, "A VNS Heuristic for TSPTW"
            rand_tw_lb = rand.uniform(total_dist - max_tw_size / 2, total_dist)
            rand_tw_ub = rand.uniform(total_dist, total_dist + max_tw_size / 2)
        else:
            # Cappart et al. style 'propagates' the time windows resulting in little overlap / easier instances
            rand_tw_lb = rand.uniform(tw_lb_min, tw_lb_min + max_tw_gap)
            rand_tw_ub = rand.uniform(rand_tw_lb, rand_tw_lb + max_tw_size)

        if is_integer_instance:
            rand_tw_lb = np.floor(rand_tw_lb)
            rand_tw_ub = np.ceil(rand_tw_ub)

        time_windows[cur_city, :] = [rand_tw_lb, rand_tw_ub]

    if is_integer_instance:
        time_windows = time_windows.astype(np.int)

    # Don't store travel time since it takes up much
    return coord[0], coord[1:], time_windows, grid_size