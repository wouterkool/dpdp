import argparse
import os
import numpy as np
import re
from utils.data_utils import check_extension, load_dataset, save_dataset
from subprocess import check_call
import time
from datetime import timedelta
from utils.functions import run_all_in_pool
from problems.vrp.vrp_baseline import get_lkh_executable
from problems.tsptw.problem_tsptw import get_rounded_distance_matrix


def solve_lkh_log(executable, directory, name, depot, loc, timew,
                  grid_size=1, runs=1, makespan=False, disable_cache=False, only_cache=False):

    # lkhms = LKH that optimizes for makespan (total time including waiting) instead of driving time
    alg_name = 'lkh' if not makespan else 'lkhms'
    basename = "{}.{}{}".format(name, alg_name, runs)
    problem_filename = os.path.join(directory, "{}.tsptw".format(basename))
    tour_filename = os.path.join(directory, "{}.tour".format(basename))
    output_filename = os.path.join(directory, "{}.pkl".format(basename))
    param_filename = os.path.join(directory, "{}.par".format(basename))
    log_filename = os.path.join(directory, "{}.log".format(basename))

    try:
        # May have already been run

        # Use rounded euclidean distances following Cappart et al.
        coord = np.vstack((depot[None], loc))
        dist = get_rounded_distance_matrix(coord)
        if os.path.isfile(output_filename) and not disable_cache:
            tour, duration = load_dataset(output_filename)
        elif not only_cache:
            write_tsptwlib(problem_filename, dist, timew, name=name)

            params = {
                "PROBLEM_FILE": problem_filename,
                "OUTPUT_TOUR_FILE": tour_filename,
                "RUNS": runs,
                "SEED": 1234
            }
            if makespan:
                params['MAKESPAN'] = "YES"
            write_lkh_par(param_filename, params)

            with open(log_filename, 'w') as f:
                start = time.time()
                check_call([executable, param_filename], stdout=f, stderr=f)
                duration = time.time() - start

            tour = read_tsptwlib(tour_filename, n=len(timew))

            save_dataset((tour, duration), output_filename)
        else:
            raise Exception("No cached solution found")

        return calc_tsptw_cost(dist, timew, tour), tour, duration

    except Exception as e:
        print("Exception occured")
        print(e)
        return None


def calc_tsptw_cost(dist, timew, tour, makespan=False):
    assert len(tour) == len(timew) - 1
    assert (np.sort(tour) == np.arange(len(timew) - 1) + 1).all(), "All nodes must be visited once!"

    # Validate timewindow constraints
    t = timew[0, 0]
    assert t == 0
    d = 0
    prev = 0
    for n in tour:
        l, u = timew[n]
        t = max(t + dist[prev, n], l)
        d += dist[prev, n]
        assert t <= u, f"Time window violated for node {n}: {t} is not in ({l, u})"
        prev = n
    d += dist[prev, 0]
    t = t + dist[prev, 0]
    assert t <= timew[0, 1]
    return t if makespan else d


def write_lkh_par(filename, parameters):
    default_parameters = {  # Use none to include as flag instead of kv
        "SPECIAL": None,
        "MAX_TRIALS": 10000,
        "RUNS": 10,
        "TRACE_LEVEL": 1,
        "SEED": 0
    }
    with open(filename, 'w') as f:
        for k, v in {**default_parameters, **parameters}.items():
            if v is None:
                f.write("{}\n".format(k))
            else:
                f.write("{} = {}\n".format(k, v))


def read_tsptwlib(filename, n):
    with open(filename, 'r') as f:
        tour = []
        dimension = 0
        started = False
        for line in f:
            if started:
                loc = int(line)
                if loc == -1:
                    break
                tour.append(loc)
            if line.startswith("DIMENSION"):
                dimension = int(line.split(" ")[-1])

            if line.startswith("TOUR_SECTION"):
                started = True

    assert len(tour) == dimension
    tour = np.array(tour).astype(int) - 1  # Subtract 1 as depot is 1 and should be 0
    tour[tour > n] = 0  # Any nodes above the number of nodes there are is also depot
    assert tour[0] == 0  # Tour should start with depot
    # Now remove duplicates, remove if node is equal to next one (cyclic)
    tour = tour[tour != np.roll(tour, -1)]
    assert tour[-1] != 0  # Tour should not end with depot
    return tour[1:].tolist()


def write_tsptwlib(filename, dist, timew, name="problem"):


    with open(filename, 'w') as f:
        f.write("\n".join([
            "{} : {}".format(k, v)
            for k, v in (
                ("NAME", name),
                ("TYPE", "TSPTW"),
                ("DIMENSION", len(timew)),
                ("EDGE_WEIGHT_TYPE", "EXPLICIT"),
                ("EDGE_WEIGHT_FORMAT", "FULL_MATRIX")
            )
        ]))
        f.write("\n")
        f.write("EDGE_WEIGHT_SECTION\n")
        f.write("\n".join([
            " ".join(map(str, row))
            for row in dist
        ]))
        f.write("\n")
        f.write("TIME_WINDOW_SECTION\n")
        f.write("\n".join([
            "{}\t{}\t{}".format(i + 1, l, u)
            for i, (l, u) in enumerate(timew)
        ]))
        f.write("\n")
        f.write("DEPOT_SECTION\n")
        f.write("1\n")
        f.write("-1\n")
        f.write("EOF\n")


def write_tsptw_edges(filename, mask, num_depots=None):
    n = mask.shape[0] - 1
    if num_depots is None:
        num_depots = n

    # Bit weird but this is how LKH arranges multiple depots
    depots = np.arange(num_depots)  # 0 to n_depots - 1
    depots[1:] += n  # Add num nodes, so depots are (0, n + 1, n + 2, ...)

    with open(filename, 'w') as f:
        # Note: mask is already upper triangular
        # First row is connections to depot, we should replicate these for 'all depots'
        depot_edges = np.flatnonzero(mask[0])
        # TODO remove since this is temporary
        depot_edges = np.arange(n) + 1 # 1 to n, connect depot to every node to ensure feasibility
        frm, to = mask[1:, 1:].nonzero()
        # Add one for stripping of depot, nodes are 1 ... n
        frm, to = frm + 1, to + 1

        num_nodes = n + num_depots
        num_edges = num_depots * len(depot_edges) + len(frm)

        f.write(f"{num_nodes} {num_edges}\n")

        f.write("\n".join([
            f"{depot} {node}"
            for depot in depots
            for node in depot_edges
        ] + [
            f"{f} {t}"
            for f, t in zip(frm, to)
        ]))

        f.write("\n")

        f.write("EOF\n")


def solve_gvns_log(executable, directory, name, depot, loc, timew,
                   grid_size=1, runs=30, disable_cache=False, only_cache=False):
    alg_name = 'gvns'
    basename = "{}.{}{}".format(name, alg_name, runs)
    problem_filename = os.path.join(directory, "{}.tsptw".format(basename))
    output_filename = os.path.join(directory, "{}.pkl".format(basename))
    log_filename = os.path.join(directory, "{}.log".format(basename))

    try:
        # May have already been run

        # Use rounded euclidean distances following Cappart et al.
        coord = np.vstack((depot[None], loc))
        if os.path.isfile(output_filename) and not disable_cache:
            tour, duration = load_dataset(output_filename)
        elif not only_cache:
            write_tsptw(problem_filename, coord, timew, name=name)

            with open(log_filename, 'w') as f:
                start = time.time()
                seed = 1234
                check_call([executable, str(seed), str(runs), problem_filename], stdout=f, stderr=f)
                duration = time.time() - start

            tour = read_tsptw(log_filename, n=len(timew))

            save_dataset((tour, duration), output_filename)
        else:
            raise Exception("No cached solution found")

        dist = get_rounded_distance_matrix(coord)
        return calc_tsptw_cost(dist, timew, tour), tour, duration

    except Exception as e:
        print("Exception occured")
        print(e)
        return None


def read_tsptw(filename, n):
    with open(filename, 'r') as f:
        tour = []
        started = False
        for line in f:
            if line.startswith("tour="):
                started = True
            elif started and line.strip() != "":
                tour = [int(v) for v in line.strip(" -\n").split(" - ")]
                break

    assert len(tour) == n
    tour = np.array(tour).astype(int)
    assert (tour < n).all()
    assert tour[0] == 0  # Tour should start with depot
    # Now remove duplicates, remove if node is equal to next one (cyclic)
    tour = tour[tour != np.roll(tour, -1)]
    assert tour[-1] != 0  # Tour should not end with depot
    return tour[1:].tolist()


def get_gvns_executable():
    cwd = os.path.abspath(os.path.join("problems", "tsptw", "TSPTW"))
    file = os.path.join(cwd, "Run")
    if not os.path.isfile(file):
        check_call("make", cwd=cwd)
        assert os.path.isfile(file)
    return file


def write_tsptw(filename, coord, timew, name="problem"):
    with open(filename, 'w') as f:
        f.write(f"!! {name}\n")
        f.write('CUST NO.   XCOORD.   YCOORD.    DEMAND   READY TIME   DUE DATE   SERVICE TIME\n')
        for i, ((x, y), (l, u)) in enumerate(zip(coord, timew)):
            f.write('{:>5d} {:>10.6f} {:>10.6f} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f}\n'.format(i + 1, x, y, 0, l, u, 0))




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("method", help="Name of the method to evaluate, 'lkh' or 'gvns'")
    parser.add_argument("datasets", nargs='+', help="Filename of the dataset(s) to evaluate")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument("--cpus", type=int, help="Number of CPUs to use, defaults to all cores")
    parser.add_argument('--disable_cache', action='store_true', help='Disable caching')
    parser.add_argument('--only_cache', action='store_true',
                        help='Only get results from cache, fail instance if no cached solution available')
    parser.add_argument('--progress_bar_mininterval', type=float, default=0.1, help='Minimum interval')
    parser.add_argument('-n', type=int, help="Number of instances to process")
    parser.add_argument('--offset', type=int, help="Offset where to start processing")
    parser.add_argument('--results_dir', default='results', help="Name of results directory")
    parser.add_argument('--allow_failure', action='store_true', help='Allow failed runs')

    opts = parser.parse_args()

    assert opts.o is None or len(opts.datasets) == 1, "Cannot specify result filename with more than one dataset"

    for dataset_path in opts.datasets:

        assert os.path.isfile(check_extension(dataset_path)), "File does not exist!"

        dataset_basename, ext = os.path.splitext(os.path.split(dataset_path)[-1])

        if opts.o is None:
            results_dir = os.path.join(opts.results_dir, "tsptw", dataset_basename)
            os.makedirs(results_dir, exist_ok=True)

            out_file = os.path.join(results_dir, "{}{}{}-{}{}".format(
                dataset_basename,
                "offs{}".format(opts.offset) if opts.offset is not None else "",
                "n{}".format(opts.n) if opts.n is not None else "",
                opts.method, ext
            ))
        else:
            out_file = opts.o

        assert opts.f or not os.path.isfile(
            out_file), "File already exists! Try running with -f option to overwrite."

        match = re.match(r'^([a-z_]+)(\d*)$', opts.method)
        assert match
        method = match[1]
        runs = 1 if match[2] == '' else int(match[2])

        if opts.o is None:
            target_dir = os.path.join(results_dir, "{}-{}".format(
                dataset_basename,
                opts.method
            ))
        else:
            target_dir, _ = os.path.splitext(opts.o)
        assert opts.f or not os.path.isdir(target_dir), \
            "Target dir already exists! Try running with -f option to overwrite."

        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)

        dataset = load_dataset(dataset_path)

        if method in ("lkh", "lkhms"):
            executable = get_lkh_executable()

            use_multiprocessing = False
            makespan = method == "lkhms"

            def run_func(args):
                directory, name, *args = args
                depot, loc, timew, grid_size = args

                return solve_lkh_log(
                    executable,
                    directory, name,
                    depot, loc, timew, grid_size,
                    runs=runs, makespan=makespan,
                    disable_cache=opts.disable_cache,
                    only_cache=opts.only_cache
                )

            # Note: only processing n items is handled by run_all_in_pool
            results, parallelism = run_all_in_pool(
                run_func,
                target_dir, dataset, opts, use_multiprocessing=use_multiprocessing,
            )
        elif method == 'gvns':
            executable = get_gvns_executable()

            use_multiprocessing = False

            def run_func(args):
                directory, name, *args = args
                depot, loc, timew, grid_size = args

                return solve_gvns_log(
                    executable,
                    directory, name,
                    depot, loc, timew, grid_size,
                    runs=runs,
                    disable_cache=opts.disable_cache,
                    only_cache=opts.only_cache
                )

            # Note: only processing n items is handled by run_all_in_pool
            results, parallelism = run_all_in_pool(
                run_func,
                target_dir, dataset, opts, use_multiprocessing=use_multiprocessing,
            )
        else:
            assert False, "Unknown method: {}".format(opts.method)

        results_stat = results
        if opts.allow_failure:
            results_stat = [res for res in results if res is not None]
            print("Failed {} of {} instances, only showing statistics for {} solved instances".format(len(results) - len(results_stat), len(results), len(results_stat)))

        if len(results_stat) > 0:
            costs, tours, durations = zip(*results_stat)  # Not really costs since they should be negative
            print("Average cost: {} +- {}".format(np.mean(costs), 2 * np.std(costs) / np.sqrt(len(costs))))
            print("Average serial duration: {} +- {}".format(
                np.mean(durations), 2 * np.std(durations) / np.sqrt(len(durations))))
            print("Average parallel duration: {}".format(np.mean(durations) / parallelism))
            print("Calculated total duration: {}".format(timedelta(seconds=int(np.sum(durations) / parallelism))))
        else:
            print("Not printing statistics since no instances were solved")
        # Save all results!
        save_dataset((results, parallelism), out_file)
