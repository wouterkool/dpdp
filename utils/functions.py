import math
import os
from datetime import timedelta

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
from torch.utils.data import DataLoader


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def run_all_in_pool(func, directory, dataset, opts, use_multiprocessing=True):
    # # Test
    # res = func((directory, 'test', *dataset[0]))
    # return [res]

    num_cpus = os.cpu_count() if opts.cpus is None else opts.cpus

    w = len(str(len(dataset) - 1))
    offset = getattr(opts, 'offset', None)
    if offset is None:
        offset = 0
    ds = dataset[offset:(offset + opts.n if opts.n is not None else len(dataset))]
    pool_cls = (Pool if use_multiprocessing and num_cpus > 1 else ThreadPool)
    with pool_cls(num_cpus) as pool:
        results = list(tqdm(pool.imap(
            func,
            [
                (
                    directory,
                    str(i + offset).zfill(w),
                    *problem
                )
                for i, problem in enumerate(ds)
            ]
        ), total=len(ds), mininterval=opts.progress_bar_mininterval))

    failed = [str(i + offset) for i, res in enumerate(results) if res is None]

    allow_failure = getattr(opts, 'allow_failure', None)
    if allow_failure:
        if len(failed) > 0:
            print("Warning: some instances failed: {}".format(" ".join(failed)))
    else:
        assert len(failed) == 0, "Some instances failed: {}".format(" ".join(failed))
    return results, num_cpus


def get_durations(durations, batch_size, num_processes, device_count):
    avg_serial_duration = np.mean(durations)
    num_instances_per_process = len(durations) / num_processes
    num_batches = math.ceil(num_instances_per_process / batch_size)
    effective_batch_size = num_instances_per_process / num_batches
    avg_parallel_duration = avg_serial_duration / effective_batch_size
    total_duration_processes = np.sum(durations) / effective_batch_size
    # Round to seconds for nice printing
    total_duration_parallel = timedelta(seconds=int(total_duration_processes / num_processes + .5))
    total_duration_single_device = timedelta(seconds=int(total_duration_processes / (num_processes / max(device_count, 1)) + .5))
    return avg_serial_duration, avg_parallel_duration, total_duration_parallel, total_duration_single_device, effective_batch_size


def ensure_backward_compatibility(opts):
    if not hasattr(opts, 'batch_size'):  # Backwards compatibility
        opts.batch_size = opts.eval_batch_size
        del opts.eval_batch_size
    if not hasattr(opts, 'system_info'):
        opts.system_info = {
            'used_device_count': opts.device_count,
            'used_num_processes': opts.num_processes,  # Total, not per GPU! Param opts.num_processes is changed.
        }
        del opts.device_count
    if 'used_num_processed' in opts.system_info:
        opts.system_info['used_num_processes'] = opts.system_info.pop('used_num_processed')


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def compute_dataset_costs(problem, dataset, sequences, device=None, check_costs=None, batch_size=1000):
    dataloader = DataLoader(dataset, batch_size=batch_size)

    if check_costs is not None:
        check_costs = chunks(check_costs, batch_size)

    costs_of_batches = (
        compute_batch_costs(problem, batch, seq, device=device,
                            check_costs=next(check_costs) if check_costs is not None else None)
        for batch, seq in zip(dataloader, chunks(sequences, batch_size))
    )
    return [c for costs_of_batch in costs_of_batches for c in costs_of_batch]


def compute_batch_costs(problem, batch, sequences, device=None, check_costs=None):
    idx_success = torch.tensor([i for i, seq in enumerate(sequences) if seq is not None], device=device)
    if len(idx_success) == 0:
        return [None] * len(sequences)
    # if device is None:
    #     device = sequences[idx_success[0]].device
    batch_success = batch[idx_success] if torch.is_tensor(batch) else {k: v[idx_success] for k, v in batch.items()}
    maxlen = max([len(seq) for seq in sequences if seq is not None])
    sequences_success = torch.stack(
        [F.pad(torch.tensor(seq), (0, maxlen - len(seq))) for seq in sequences if seq is not None])
    costs_success, _ = problem.get_costs(batch_success, sequences_success.to(device))
    if check_costs is not None:
        check_costs_success = [check_costs[i] for i in idx_success]
        isclose = torch.isclose(costs_success, costs_success.new_tensor(check_costs_success), atol=1e-10)
        if not isclose.all():
            print("Warning check cost {} not exactly equal to {}".format(costs_success, check_costs_success))
        assert torch.allclose(costs_success, costs_success.new_tensor(check_costs_success),
                              atol=1e-4), "Check cost {} not equal to {}".format(costs_success, check_costs_success)
    costs = [None] * len(sequences)
    for i, cost in zip(idx_success, costs_success):
        costs[i] = cost.item()
    return costs
