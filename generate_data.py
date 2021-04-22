import argparse
import os
import numpy as np
from utils.data_utils import check_extension, save_dataset


def generate_tsp_data(dataset_size, tsp_size):
    return np.random.uniform(size=(dataset_size, tsp_size, 2)).tolist()


def generate_vrp_data(dataset_size, vrp_size, distribution, seed):
    if distribution == 'nazari':
        return generate_nazari_vrp_data(dataset_size, vrp_size)
    else:
        return generate_uchoa_vrp_data(dataset_size, vrp_size, distribution, seed)
    assert False, f"Unknown VRP distribution: {distribution}"


def generate_nazari_vrp_data(dataset_size, vrp_size):
    CAPACITIES = {
        10: 20.,
        20: 30.,
        50: 40.,
        100: 50.
    }
    return list(zip(
        np.random.uniform(size=(dataset_size, 2)).tolist(),  # Depot location
        np.random.uniform(size=(dataset_size, vrp_size, 2)).tolist(),  # Node locations
        np.random.randint(1, 10, size=(dataset_size, vrp_size)).tolist(),  # Demand, uniform integer 1 ... 9
        np.full(dataset_size, CAPACITIES[vrp_size]).tolist()  # Capacity, same for whole dataset
    ))


def generate_uchoa_vrp_data(dataset_size, vrp_size, distribution, seed):
    import torch
    from problems.vrp.generate_uchoa_data import generate_uchoa_instances

    torch.manual_seed(seed)
    depot, loc, demand, capacity, depot_types, customer_types, grid_size = generate_uchoa_instances(
        dataset_size, vrp_size, distribution)
    return list(zip(
        depot.tolist(),
        loc.tolist(),
        demand.tolist(),
        capacity.tolist(),
        depot_types.tolist(),
        customer_types.tolist(),
        grid_size.tolist()
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")
    parser.add_argument("--name", type=str, required=True, help="Name to identify dataset")
    parser.add_argument("--problem", type=str, default='all',
                        help="Problem, 'tsp', 'vrp' or 'all' to generate all")
    parser.add_argument('--data_distribution', type=str, default='all',
                        help="Distributions to generate for problem, default 'all'.")
    parser.add_argument("--dataset_size", type=int, default=10000, help="Size of the dataset")
    parser.add_argument('--graph_sizes', type=int, nargs='+', default=[100],
                        help="Sizes of problem instances (default 100)")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")

    opts = parser.parse_args()

    assert opts.filename is None or (opts.problem != 'all' and len(opts.graph_sizes) == 1), \
        "Can only specify filename when generating a single dataset"

    distributions_per_problem = {
        'tsp': [None],
        'vrp': ['nazari', 'uchoa'],
    }
    if opts.problem == 'all':
        problems = distributions_per_problem
    else:
        problems = {
            opts.problem:
                distributions_per_problem[opts.problem]
                if opts.data_distribution == 'all'
                else [opts.data_distribution]
        }
    assert opts.filename is None or len(problems[opts.problem]) == 1, "Can only specify single distribution when generating single dataset"

    for problem, distributions in problems.items():
        for distribution in distributions or [None]:
            for graph_size in opts.graph_sizes:

                datadir = os.path.join(opts.data_dir, problem)
                os.makedirs(datadir, exist_ok=True)

                if opts.filename is None:
                    filename = os.path.join(datadir, "{}{}{}_{}_seed{}.pkl".format(
                        problem,
                        "_{}".format(distribution) if distribution is not None else "",
                        graph_size, opts.name, opts.seed))
                else:
                    filename = check_extension(opts.filename)

                assert opts.f or not os.path.isfile(check_extension(filename)), \
                    "File already exists! Try running with -f option to overwrite."

                np.random.seed(opts.seed)
                if problem == 'tsp':
                    dataset = generate_tsp_data(opts.dataset_size, graph_size)
                elif problem == 'vrp':
                    dataset = generate_vrp_data(
                        opts.dataset_size, graph_size, distribution=distribution, seed=opts.seed)
                else:
                    assert False, "Unknown problem: {}".format(problem)

                print(dataset[0])

                save_dataset(dataset, filename)
