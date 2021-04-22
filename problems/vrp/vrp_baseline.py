import argparse
import os
import math
import numpy as np
import re
from utils.data_utils import check_extension, load_dataset, save_dataset, load_heatmaps
from subprocess import check_call, check_output
from urllib.parse import urlparse
import tempfile
import time
from datetime import timedelta
from utils.functions import run_all_in_pool


def evaluate_dp(depot, loc, demand, vehicle_capacity, heatmap, beam_size, collapse, score_function, heatmap_threshold, knn, verbose, device=None):
    import torch
    from dp import Graph, StreamingTopK, run_dp

    # coord = torch.from_numpy(np.concatenate((depot[None], loc), 0)).to(device)
    coord = torch.cat((torch.tensor(depot)[None], torch.tensor(loc)), 0).to(device)
    demand = torch.tensor(demand).to(device)

    graph = Graph.get_graph(
        coord, score_function=score_function, heatmap=heatmap, heatmap_threshold=heatmap_threshold, knn=knn, quantize_cost_dtype=torch.int32,
        demand=demand, vehicle_capacity=vehicle_capacity,
        start_node=0, node_score_weight=1.0, node_score_dist_to_start_weight=0.1
    )

    add_potentials = graph.edge_weight is not None
    assert add_potentials == ("potential" in score_function.split("_"))

    candidate_queue = StreamingTopK(
        beam_size,
        dtype=graph.score.dtype if graph.score is not None else graph.cost.dtype,
        verbose=verbose,
        payload_dtypes=(torch.int32, torch.int16),  # parent = max 1e9, action = max 2e3 (for VRP with 1000 nodes)
        device=coord.device,
        alloc_size_factor=10. if beam_size <= int(1e6) else 2.,  # up to 1M we can easily allocate 10x so 10MB
        kthvalue_method='sort'  # Other methods may increase performance but are experimental / buggy
    )

    start = time.time()
    mincost_dp_qt, solution = run_dp(graph, candidate_queue, return_solution=True, collapse=collapse,
           beam_device=coord.device, bound_first=True,  # Always bound first #is_vrp or beam_size >= int(1e7),
           sort_beam_by='group_idx', trace_device='cpu',
           verbose=verbose, add_potentials=add_potentials
    )

    duration = time.time() - start
    if solution is None:
        print("Unable to find solution!")
        cost = None
    else:
        # TODO check
        tour = solution.cpu().numpy()
        mincost_dp = graph.dequantize_cost(mincost_dp_qt).item()
        cost = calc_vrp_cost(depot, loc, tour)
        assert np.allclose(cost, mincost_dp)

    return cost, tour, duration


def get_lkh_executable(url="http://www.akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.4.tgz"):

    cwd = os.path.abspath(os.path.join("problems", "vrp", "lkh"))
    os.makedirs(cwd, exist_ok=True)

    file = os.path.join(cwd, os.path.split(urlparse(url).path)[-1])
    filedir = os.path.splitext(file)[0]

    if not os.path.isdir(filedir):
        print("{} not found, downloading and compiling".format(filedir))

        check_call(["wget", url], cwd=cwd)
        assert os.path.isfile(file), "Download failed, {} does not exist".format(file)
        check_call(["tar", "xvfz", file], cwd=cwd)

        assert os.path.isdir(filedir), "Extracting failed, dir {} does not exist".format(filedir)
        check_call("make", cwd=filedir)
        os.remove(file)

    executable = os.path.join(filedir, "LKH")
    assert os.path.isfile(executable)
    return os.path.abspath(executable)


def solve_lkh(executable, depot, loc, demand, capacity):
    with tempfile.TemporaryDirectory() as tempdir:
        problem_filename = os.path.join(tempdir, "problem.vrp")
        output_filename = os.path.join(tempdir, "output.tour")
        param_filename = os.path.join(tempdir, "params.par")

        starttime = time.time()
        write_vrplib(problem_filename, depot, loc, demand, capacity)
        params = {"PROBLEM_FILE": problem_filename, "OUTPUT_TOUR_FILE": output_filename}
        write_lkh_par(param_filename, params)
        output = check_output([executable, param_filename])
        result = read_vrplib(output_filename, n=len(demand))
        duration = time.time() - starttime
        return result, output, duration


def solve_lkh_log(executable, directory, name, depot, loc, demand, capacity,
                  grid_size=1, mask=None, runs=1, unlimited_routes=False, disable_cache=False, only_cache=False):

    # lkhu = LKH with unlimited routes/salesmen
    alg_name = 'lkhu' if unlimited_routes else 'lkh'
    basename = "{}.{}{}".format(name, alg_name, runs)
    problem_filename = os.path.join(directory, "{}.vrp".format(basename))
    tour_filename = os.path.join(directory, "{}.tour".format(basename))
    output_filename = os.path.join(directory, "{}.pkl".format(basename))
    param_filename = os.path.join(directory, "{}.par".format(basename))
    log_filename = os.path.join(directory, "{}.log".format(basename))
    if mask is not None:
        edges_filename = os.path.join(directory, "{}.edges".format(basename))

    try:
        # May have already been run
        if os.path.isfile(output_filename) and not disable_cache:
            tour, duration = load_dataset(output_filename)
        elif not only_cache:
            write_vrplib(problem_filename, depot, loc, demand, capacity, grid_size, name=name)

            params = {
                "PROBLEM_FILE": problem_filename,
                "OUTPUT_TOUR_FILE": tour_filename,
                "RUNS": runs,
                "SEED": 1234
            }
            if unlimited_routes:
                # Option unlimited_routes is False by default for backwards compatibility
                # By default lkh computes some bound for the number of salesmen which can result in infeasible solutions
                params['SALESMEN'] = len(loc)
                params['MTSP_MIN_SIZE'] = 0  # We should allow for empty routes
            if mask is not None:
                # assert unlimited_routes, "For now, edges only work with unlimited routes so we now num routes a priori"
                if unlimited_routes:
                    num_depots = len(loc)
                else:
                    # Standard LKH computes a bound
                    num_depots = math.ceil(sum(demand) / capacity)
                write_vrp_edges(edges_filename, mask, num_depots=num_depots)
                params['EDGE_FILE'] = edges_filename
                # Next to the predicted edges, we should not have additional edges (nearest neighbours etc.)
                params['MAX_CANDIDATES'] = 0
            write_lkh_par(param_filename, params)

            with open(log_filename, 'w') as f:
                start = time.time()
                check_call([executable, param_filename], stdout=f, stderr=f)
                duration = time.time() - start

            tour = read_vrplib(tour_filename, n=len(demand))

            save_dataset((tour, duration), output_filename)
        else:
            raise Exception("No cached solution found")

        return calc_vrp_cost(depot, loc, tour), tour, duration

    except Exception as e:
        print("Exception occured")
        print(e)
        return None


def calc_vrp_cost(depot, loc, tour):
    assert (np.sort(tour)[-len(loc):] == np.arange(len(loc)) + 1).all(), "All nodes must be visited once!"
    # TODO validate capacity constraints
    loc_with_depot = np.vstack((np.array(depot)[None, :], np.array(loc)))
    sorted_locs = loc_with_depot[np.concatenate(([0], tour, [0]))]
    return np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1).sum()


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


def read_vrplib(filename, n):
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


def write_vrplib(filename, depot, loc, demand, capacity, grid_size, name="problem"):

    with open(filename, 'w') as f:
        f.write("\n".join([
            "{} : {}".format(k, v)
            for k, v in (
                ("NAME", name),
                ("TYPE", "CVRP"),
                ("DIMENSION", len(loc) + 1),
                ("EDGE_WEIGHT_TYPE", "EUC_2D"),
                ("CAPACITY", capacity)
            )
        ]))
        f.write("\n")
        f.write("NODE_COORD_SECTION\n")
        f.write("\n".join([
            "{}\t{}\t{}".format(i + 1, int(x / grid_size * 100000 + 0.5), int(y / grid_size * 100000 + 0.5))  # VRPlib does not take floats
            #"{}\t{}\t{}".format(i + 1, x, y)
            for i, (x, y) in enumerate([depot] + loc)
        ]))
        f.write("\n")
        f.write("DEMAND_SECTION\n")
        f.write("\n".join([
            "{}\t{}".format(i + 1, d)
            for i, d in enumerate([0] + demand)
        ]))
        f.write("\n")
        f.write("DEPOT_SECTION\n")
        f.write("1\n")
        f.write("-1\n")
        f.write("EOF\n")


def write_vrp_edges(filename, mask, num_depots=None):
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("method", help="Name of the method to evaluate, 'lkh', 'dpdp' or 'dpbs'")
    parser.add_argument("datasets", nargs='+', help="Filename of the dataset(s) to evaluate")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument("--cpus", type=int, help="Number of CPUs to use, defaults to all cores")
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--disable_cache', action='store_true', help='Disable caching')
    parser.add_argument('--only_cache', action='store_true',
                        help='Only get results from cache, fail instance if no cached solution available')
    parser.add_argument('--progress_bar_mininterval', type=float, default=0.1, help='Minimum interval')
    parser.add_argument('-n', type=int, help="Number of instances to process")
    parser.add_argument('--offset', type=int, help="Offset where to start processing")
    parser.add_argument('--results_dir', default='results', help="Name of results directory")
    parser.add_argument('--allow_failure', action='store_true', help='Allow failed runs')
    # When providing a heatmap, will sparsify the input
    parser.add_argument('--heatmap', default=None, help="Heatmaps to use")
    parser.add_argument('--heatmap_threshold', type=float, default=1e-5, help="Min threshold for heatmaps")
    parser.add_argument('--heatmap_max_edges', type=int, default=None, help="Max number of edges for heatmaps")
    parser.add_argument('--score_function', type=str, default='heatmap_potential',
                        help="Policy/score function to use to select beam: 'cost', 'heatmap' or 'heatmap_potential'")
    parser.add_argument('--beam_size', type=int, default=None, help="Beam size with DPDP")
    parser.add_argument('--knn', type=int, default=-1, help="Use K-nearest neighbor graph with DPDP")
    parser.add_argument('--verbose', action='store_true', help='Set to show verbose output')

    opts = parser.parse_args()

    assert opts.o is None or len(opts.datasets) == 1, "Cannot specify result filename with more than one dataset"
    assert opts.heatmap is None or len(opts.datasets) == 1, "Cannot specify heatmap with more than one dataset"

    assert opts.heatmap is None or not opts.f or opts.o is not None, \
        "Must specify output filename when using heatmap with overwrite to prevent accidental overwrite"

    for dataset_path in opts.datasets:

        assert os.path.isfile(check_extension(dataset_path)), "File does not exist!"

        dataset_basename, ext = os.path.splitext(os.path.split(dataset_path)[-1])

        if opts.o is None:
            results_dir = os.path.join(opts.results_dir, "vrp", dataset_basename)
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

        raw_dataset = load_dataset(dataset_path)

        if opts.heatmap is not None:
            heatmaps = load_heatmaps(opts.heatmap, symmetric=True)
            if method == "lkh" or method == "lkhu":
                # Note heatmaps / sparse graphs with LKH do not work well
                # Only take each edge once (upper triangular part)
                rng_node = np.arange(heatmaps.shape[-1])
                triu_mask = (rng_node[None, None, :] > rng_node[None, :, None])
                heatmaps = heatmaps * triu_mask

            dataset = [(instance, hm) for instance, hm in zip(raw_dataset, heatmaps)]
        else:
            dataset = [(instance, None) for instance in raw_dataset]

        if method == "lkh" or method == "lkhu":
            executable = get_lkh_executable()

            use_multiprocessing = False
            unlimited_routes = method == "lkhu"

            def run_func(args):
                directory, name, *args = args
                args, heatmap = args
                depot, loc, demand, capacity, *args = args
                grid_size = 1
                if len(args) > 0:
                    depot_types, customer_types, grid_size = args

                return solve_lkh_log(
                    executable,
                    directory, name,
                    depot, loc, demand, capacity, grid_size,
                    mask=heatmap > opts.heatmap_threshold if heatmap is not None else None,
                    runs=runs, unlimited_routes=unlimited_routes,
                    disable_cache=opts.disable_cache,
                    only_cache=opts.only_cache
                )

            # Note: only processing n items is handled by run_all_in_pool
            results, parallelism = run_all_in_pool(
                run_func,
                target_dir, dataset, opts, use_multiprocessing=use_multiprocessing,
            )
        elif method in ('dpdp', 'dpbs'):
            import torch
            use_cuda = torch.cuda.is_available() and not opts.no_cuda
            device_count = torch.cuda.device_count() if use_cuda else 0
            num_cpus = os.cpu_count() if opts.cpus is None else opts.cpus
            assert device_count == 0 or num_cpus % device_count == 0, "Num cpus must be multiple of CUDA device count"

            def run_func(args):
                device = torch.device('cuda:0' if use_cuda else 'cpu')
                if device_count > 1:
                    from multiprocessing import current_process
                    # identity is from 1 to num_cpus
                    p = current_process()
                    # Define device id from worker id
                    device_id = (p._identity[0] - 1) % device_count
                    device = torch.device(f'cuda:{device_id}')

                # device, *args = args
                directory, name, *args = args
                args, heatmap = args
                depot, loc, demand, capacity, *args = args
                grid_size = 1
                if len(args) > 0:
                    depot_types, customer_types, grid_size = args
                collapse = method == 'dpdp'
                evaluate_dp(depot, loc, demand, capacity, heatmap, opts.beam_size, collapse, opts.score_function,
                            opts.heatmap_threshold, opts.knn, opts.verbose, device=device)

            results, parallelism = run_all_in_pool(
                run_func,
                target_dir, dataset, opts, use_multiprocessing=num_cpus > 1,
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
