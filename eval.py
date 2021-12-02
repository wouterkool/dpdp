import psutil
import torch
import os
import argparse
import numpy as np
import itertools
from tqdm import tqdm
from problems import load_problem
from utils.data_utils import save_dataset, load_heatmaps
from utils.functions import move_to, get_durations, compute_batch_costs, accurate_cdist
from torch.utils.data import DataLoader
import time
from dp import BatchGraph, StreamingTopK, SimpleBatchTopK, run_dp
from torch.utils.data import Dataset

# Fix according to https://discuss.pytorch.org/t/
# a-call-to-torch-cuda-is-available-makes-an-unrelated-multi-processing-computation-crash/4075/4
mp = torch.multiprocessing.get_context('spawn')


def evaluate_dp(is_vrp, has_tw, batch, heatmaps, beam_size, collapse, score_function,
                heatmap_threshold, knn, use_weak_version, verbose):

    coords = torch.cat((batch['depot'][:, None], batch['loc']), 1).float() if is_vrp or has_tw else batch
    demands = batch['demand'] if is_vrp else None
    vehicle_capacities = batch['capacity'] if is_vrp else None
    timew = batch['timew'] if has_tw else None
    dist = accurate_cdist(coords, coords)
    quant_c_dt = torch.int32
    if has_tw:
        dist = dist.round()
        assert (dist.max(-1).values.sum(-1) < torch.iinfo(torch.int).max).all()
        assert (timew < torch.iinfo(torch.int).max).all()
        dist = dist.int()
        timew = timew.int()
        quant_c_dt = None  # Don't use quantization since we're using ints already
        batch['dist'] = dist  # For final distance computation

    graph = BatchGraph.get_graph(
        dist, score_function=score_function, heatmap=heatmaps, heatmap_threshold=heatmap_threshold, knn=knn,
        quantize_cost_dtype=quant_c_dt, demand=demands, vehicle_capacity=vehicle_capacities, timew=timew,
        start_node=0, node_score_weight=1.0, node_score_dist_to_start_weight=0.1
    )
    assert graph.batch_size == len(coords)
    add_potentials = graph.edge_weight is not None
    assert add_potentials == ("potential" in score_function.split("_"))

    if False:
        # This implementation is simpler but slower
        candidate_queue = SimpleBatchTopK(beam_size)
    else:
        candidate_queue = StreamingTopK(
            beam_size,
            dtype=graph.score.dtype if graph.score is not None else graph.cost.dtype,
            verbose=verbose,
            payload_dtypes=(torch.int32, torch.int16),  # parent = max 1e9, action = max 2e3 (for VRP with 1000 nodes)
            device=coords.device,
            alloc_size_factor=10. if beam_size * graph.batch_size <= int(1e6) else 2.,  # up to 1M we can easily allocate 10x so 10MB
            kthvalue_method='sort',  # Other methods may increase performance but are experimental / buggy
            batch_size=graph.batch_size
        )
    mincost_dp_qt, solution = run_dp(
        graph, candidate_queue, return_solution=True, collapse=collapse, use_weak_version=use_weak_version,
        beam_device=coords.device, bound_first=True, # Always bound first #is_vrp or beam_size >= int(1e7),
        sort_beam_by='group_idx', trace_device='cpu',
        verbose=verbose, add_potentials=add_potentials
    )
    assert len(mincost_dp_qt) == graph.batch_size
    assert len(solution) == graph.batch_size
    solutions_np = [sol.cpu().numpy() if sol is not None else None for sol in solution]
    cost = graph.dequantize_cost(mincost_dp_qt)
    return solutions_np, cost, graph.batch_size


class HeatmapDataset(Dataset):

    def __init__(self, dataset=None, heatmaps=None):
        super(HeatmapDataset, self).__init__()

        self.dataset = dataset
        self.heatmaps = heatmaps
        assert (len(self.dataset) == len(self.heatmaps)), f"Found {len(self.dataset)} instances but {len(self.heatmaps)} heatmaps"

    def __getitem__(self, item):
        return {
            'data': self.dataset[item],
            'heatmap': self.heatmaps[item]
        }

    def __len__(self):
        return len(self.dataset)


def unpack_heatmaps(batch):
    if isinstance(batch, dict) and 'heatmap' in batch and 'data' in batch:
        return batch['data'], batch['heatmap']
    return batch, None


def pack_heatmaps(dataset, opts, offset=None):
    if opts.heatmap is None:
        return dataset
    offset = offset or opts.offset
    # For TSPTW, use undirected heatmap since problem is undirected because of time windows
    return HeatmapDataset(dataset, load_heatmaps(opts.heatmap, symmetric=opts.problem != 'tsptw')[offset:offset+len(dataset)])


def eval_dataset_mp(args):
    (dataset_path, beam_size, opts, i, device_num, num_processes) = args

    problem = load_problem(opts.problem)
    val_size = opts.val_size // num_processes
    make_dataset_kwargs = {'normalize': False} if opts.decode_strategy[:4] in ('dpbs', 'dpdp') and problem.NAME == 'cvrp' else {}
    dataset = problem.make_dataset(filename=dataset_path, num_samples=val_size, offset=opts.offset + val_size * i, **make_dataset_kwargs)
    dataset = pack_heatmaps(dataset, opts, offset=opts.offset + val_size * i)
    device = torch.device("cuda:{}".format(device_num) if device_num is not None else 'cpu')

    return _eval_dataset(problem, dataset, beam_size, opts, device, no_progress_bar=opts.no_progress_bar or i > 0)  # Disable for other processes


def eval_dataset(dataset_path, beam_size, opts):
    # Even with multiprocessing, we load the model here since it contains the name where to write results

    problem = load_problem(opts.problem)
    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    device_count = torch.cuda.device_count() if use_cuda else 1
    num_processes = opts.num_processes * device_count

    # For logging
    opts.system_info = {
        'used_device_count': device_count,
        'used_num_processes': num_processes,
        'devices': ['cpu'] if not use_cuda else [torch.cuda.get_device_name(i) for i in range(device_count)],
        'cpu_count': os.cpu_count(),
        'memory_gb': psutil.virtual_memory().total / (2 ** 30)
    }

    if num_processes > 1:
        # assert use_cuda, "Can only do multiprocessing with cuda"
        assert opts.val_size % num_processes == 0, f"Dataset size {opts.val_size} must be divisible by {device_count} devices x {opts.num_processes} processes = {num_processes}"

        with mp.Pool(num_processes) as pool:
            results = list(itertools.chain.from_iterable(pool.map(
                eval_dataset_mp,
                [(dataset_path, beam_size, opts, i, i % device_count if use_cuda else None, num_processes) for i in range(num_processes)]
            )))

    else:
        device = torch.device("cuda:0" if use_cuda else "cpu")
        make_dataset_kwargs = {'normalize': False} if opts.decode_strategy[:4] in ('dpbs', 'dpdp') and problem.NAME == 'cvrp' else {}
        dataset = problem.make_dataset(filename=dataset_path, num_samples=opts.val_size, offset=opts.offset, **make_dataset_kwargs)
        dataset = pack_heatmaps(dataset, opts)
        results = _eval_dataset(problem, dataset, beam_size, opts, device, no_progress_bar=opts.no_progress_bar)

    costs, durations, tours = print_statistics(results, opts)

    dataset_basename, ext = os.path.splitext(os.path.split(dataset_path)[-1])
    heatmap_basename = os.path.splitext(os.path.split(opts.heatmap)[-1])[0] if opts.heatmap is not None else ""
    if opts.o is None:
        
        results_dir = os.path.join(opts.results_dir, 'vrp' if problem.NAME == 'cvrp' else problem.NAME, dataset_basename)
        os.makedirs(results_dir, exist_ok=True)

        out_file = os.path.join(results_dir, "{}{}{}-{}-{}{}{}-{}{}{}".format(
            dataset_basename,
            "offs{}".format(opts.offset) if opts.offset is not None else "",
            "n{}".format(opts.val_size) if opts.val_size is not None else "",
            heatmap_basename,
            opts.decode_strategy, beam_size, opts.score_function,
            "th" + str(opts.heatmap_threshold) if opts.heatmap_threshold is not None else "",
            "knn" + str(opts.knn) if opts.knn is not None else "",
            ext
        ))
    else:
        out_file = opts.o

    assert opts.f or not os.path.isfile(
        out_file), "File already exists! Try running with -f option to overwrite."

    print(out_file)
    # Save the options so we can recall everything
    save_dataset((results, opts), out_file)

    return costs, tours, durations


def print_statistics(results, opts):
    num_processes = opts.system_info['used_num_processes']
    device_count = opts.system_info['used_device_count']
    batch_size = opts.batch_size
    assert num_processes % device_count == 0
    num_processes_per_device = num_processes // device_count

    results_stat = [(cost, tour, duration) for (cost, tour, duration) in results if tour is not None]
    if len(results_stat) < len(results):
        failed = [i + opts.offset for i, (cost, tour, duration) in enumerate(results) if tour is None]
        print("*" * 100)
        print("FAILED {} of {} instances, only showing statistics for {} solved instances!".format(
            len(results) - len(results_stat), len(results), len(results_stat)))
        print("Instances failed (showing max 10): ", failed[:10])
        print("*" * 100)
        # results = results_stat
    costs, tours, durations = zip(*results_stat)  # Not really costs since they should be negative
    print("Costs (showing max 10): ", costs[:10])
    if len(tours) == 1:
        print("Tour", tours[0])
    print("Average cost: {:.3f} +- {:.3f}".format(np.mean(costs), 2 * np.std(costs) / np.sqrt(len(costs))))

    avg_serial_duration, avg_parallel_duration, total_duration_parallel, total_duration_single_device, effective_batch_size = get_durations(
        durations, batch_size, num_processes, device_count
    )

    print("Average serial duration (per process per device): {:.3f}".format(avg_serial_duration))
    if batch_size > 1:
        print("Average parallel duration (per process per device), effective batch size {:.2f}): {:.3f}".format(
            effective_batch_size, avg_parallel_duration))
    if device_count > 1:
        print(
            "Calculated total duration for {} instances with {} processes x {} devices (= {} proc) in parallel: {}".format(
                len(durations), num_processes_per_device, device_count, num_processes, total_duration_parallel))
    # On 1 device it takes k times longer than on k devices
    print("Calculated total duration for {} instances with {} processes on 1 device in parallel: {}".format(
        len(durations), num_processes_per_device, total_duration_single_device))
    print("Number of GPUs used:", device_count)
    return costs, durations, tours


def _eval_dataset(problem, dataset, beam_size, opts, device, no_progress_bar=False):

    dataloader = DataLoader(dataset, batch_size=opts.batch_size)

    results = []
    for batch in tqdm(dataloader, disable=no_progress_bar):
        batch = move_to(batch, device)
        batch, heatmaps = unpack_heatmaps(batch)

        start = time.time()
        with torch.no_grad():

            if opts.decode_strategy[:4] in ('dpbs', 'dpdp'):

                assert opts.heatmap_threshold is None or opts.knn is None, "Cannot have both"
                assert problem.NAME in ('cvrp', 'tsp', 'tsptw')
                # Deep policy beam search or deep policy dynamic programming = new style implementation

                batch_size = len(batch) if problem.NAME == 'tsp' else len(batch['loc'])
                try:
                    sequences, costs, batch_size = evaluate_dp(
                        problem.NAME == 'cvrp', problem.NAME == 'tsptw', batch, heatmaps=heatmaps,
                        beam_size=beam_size, collapse=opts.decode_strategy[:4] == 'dpdp', score_function=opts.score_function,
                        heatmap_threshold=opts.heatmap_threshold, knn=opts.knn, use_weak_version=opts.decode_strategy[-1] == '-',
                        verbose=opts.verbose
                    )
                except RuntimeError as e:
                    if 'out of memory' in str(e) and opts.skip_oom:
                        print('| WARNING: ran out of memory, skipping batch')
                        sequences = [None] * batch_size
                        costs = [None] * batch_size
                    else:
                        raise e

                costs = compute_batch_costs(problem, batch, sequences, device=device, check_costs=costs)

        assert len(sequences) == batch_size
        duration = time.time() - start
        # print(sequences, costs)
        for seq, cost in zip(sequences, costs):
            if problem.NAME in ("tsp", "tsptw"):
                if seq is not None:  # tsptw can be infeasible or TSP failed with sparse graph
                    seq = seq.tolist()  # No need to trim as all are same length
            elif problem.NAME == "cvrp":
                if seq is not None:  # Can be failed with sparse graph
                    seq = np.trim_zeros(seq).tolist() + [0]  # Add depot
            else:
                assert False, "Unkown problem: {}".format(problem.NAME)
            # Note VRP only
            results.append((cost, seq, duration))
    assert len(results) == len(dataset)
    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs='+', help="Filename of the dataset(s) to evaluate")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument('--val_size', type=int, default=10000,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--offset', type=int, default=0,
                        help='Offset where to start in dataset (default 0)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Batch size to use during evaluation (per GPU)")
    parser.add_argument('--beam_size', type=int, nargs='+',
                        help='Sizes of beam to use for beam search/DP')
    parser.add_argument('--decode_strategy', type=str,
                        help='Deep Policy Dynamic Programming (dpdp) or Deep Policy Beam Search (dpbs)')
    parser.add_argument('--score_function', type=str, default='model_local',
                        help="Policy/score function to use to select beam: 'cost', 'heatmap' or 'heatmap_potential'")
    parser.add_argument('--problem', type=str)
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--verbose', action='store_true', help='Set to show statistics')
    parser.add_argument('--results_dir', default='results', help="Name of results directory")
    parser.add_argument('--num_processes', type=int, default=1, help='Number of processes to use per device (cpu or gpu).')
    # When providing a heatmap, will sparsify the input
    parser.add_argument('--heatmap', default=None, help="Heatmaps to use")
    parser.add_argument('--heatmap_threshold', type=float, default=None, help="Use sparse graph based on heatmap treshold")
    parser.add_argument('--knn', type=int, default=None, help="Use sparse knn graph")
    parser.add_argument('--kthvalue_method', type=str, default='sort', help="Which kthvalue method to use for dpdp ('auto' = auto determine)")
    parser.add_argument('--skip_oom', action='store_true', help='Skip batch when out of memory')

    opts = parser.parse_args()
    assert opts.o is None or (len(opts.datasets) == 1 and len(opts.beam_size) <= 1), \
        "Cannot specify result filename with more than one dataset or more than one beam_size"
    assert opts.heatmap is None or len(opts.datasets) == 1, "With heatmap can only run one (corresponding) dataset"
    beam_sizes = opts.beam_size if opts.beam_size is not None else [0]

    for beam_size in beam_sizes:
        for dataset_path in opts.datasets:
            eval_dataset(dataset_path, beam_size, opts)
