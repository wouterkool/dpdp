import torch
import math


try:
    import cupy as cp
    from .cupy_utils import ensure_torch_mempool_in_cupy, to_cp, to_pt
    _cupy_import_error = None
except Exception as e:
    _cupy_import_error = e


def kthvalue_via_cupy(vals, k):
    assert _cupy_import_error is None
    ensure_torch_mempool_in_cupy()
    with cp.cuda.Device(vals.device.index):
        vals_partitioned = to_pt(cp.partition(to_cp(vals), k))
    return vals_partitioned[k-1].clone()  # To release the memory of vals_partitioned we clone


def kthvalue(vals, k, dim=-1, outmask=None, return_count_le=False, return_index=False, min_steps='auto', early_stop_max_k=None, max_chunk_size=int(1e7)):
    """
    Custom implementation of kthvalue for large tensors, which uses a binary search.
    """
    assert early_stop_max_k is None or not return_index, "Returning index not compatible with early stop"
    if early_stop_max_k is None:
        early_stop_max_k = k
    needle_size = (early_stop_max_k - k + 1) # How big is the interval we should hit (at least 1)

    # We make the mask a long since this we need to have a long anyway when summing
    mask = outmask if outmask is not None else torch.empty_like(vals, dtype=torch.bool)
    if not torch.is_tensor(k):
        # Otherwise we we will be casting to tensor over and over
        k = torch.tensor(k, device=vals.device)

    assert (0 < k <= vals.size(dim)).all()
    k, _ = torch.broadcast_tensors(k, vals.narrow(dim, start=0, length=1).squeeze(dim))

    # Compute expected minimal number of steps, take some margin since we can be unlucky
    # Device the number of items were searching for the size of the interval we should hit
    # dividing the two gives by how much we should reduce the search space so we need log2(ratio) steps expected
    steps = compute_needed_bits((vals.size(dim) + needle_size - 1) // needle_size)[0] + 2 if min_steps == 'auto' else min_steps
    try:
        MINVAL = vals.new_tensor(torch.iinfo(vals.dtype).min)
        is_integer = True
    except TypeError:
        MINVAL = vals.new_tensor(-math.inf)
        is_integer = False

    if len(vals) > max_chunk_size:
        def compute_sum(val, dim):
            return torch.stack([chunk.sum(dim) for chunk in val.split(max_chunk_size, dim=dim)], dim).sum(dim)
    else:
        compute_sum = torch.sum

    lb_val, _ = vals.min(dim)
    ub_val, ub_ind = vals.max(dim)

    success = False
    mid_val, mid_ind = ub_val, ub_ind  # Initialize with ub_ind so that if lb_val == ub_val this is the corresponding idx
    count_le = vals.size(0)  # Everything is smaller or equal than upper bound
    # Note: even though we have floating point precision, the actual values will always be one of the elements
    # so we can use exact comparison
    while lb_val != ub_val and not (k <= count_le <= early_stop_max_k):

        for i in range(steps):
            # We should round down, so if lb_val and ub_val are very close, then we should never get the ub_val out
            mid_val = ((lb_val + ub_val) // 2) if is_integer else ((lb_val + ub_val) / 2).clamp(max=torch.nextafter(ub_val, lb_val))
            torch.le(vals, mid_val, out=mask)
            count_le = compute_sum(mask, dim)
            # Find largest value <= mid_val, this is the actual value!
            mid_val_exact, mid_ind = torch.where(mask, vals, MINVAL).max(dim)  # Vector with entries > mid set to lb

            # Note: this is a scalar tensor
            success = count_le >= k

            # By doing it in this way, we can do it in batch and pytorch does not synchronize!
            ub_val = torch.where(success, mid_val, ub_val)

            # Note as lower bound, mid val cannot work so we can set mid_val_above as lower bound
            # Which is a bit tighter
            # As new lower bound, set mid_val + 1 (or float equivalent)
            lb_val = torch.where(success, lb_val, mid_val + 1 if is_integer else torch.nextafter(mid_val, ub_val))

        # Typically we will be done, otherwise do some more steps (but half the size)
        steps = 1

    kthval = ub_val
    if not success:
        # Always ub is guaranteed successfull, find index
        _, kthval_idx = torch.eq(vals, kthval, out=mask).max(dim)
        if outmask is not None or return_count_le:
            # Make sure we fill the mask with entries smaller
            torch.le(vals, kthval, out=mask)
            if return_count_le:
                count_le = mask.sum()
    else:
        kthval_idx = mid_ind
    assert (kthval_idx >= 0).all()

    assert count_le >= k or not return_count_le  # May be incorrect if we don't return it
    if return_index:
        return (kthval, kthval_idx, count_le) if return_count_le else (kthval, kthval_idx)
    else:
        return (kthval, count_le) if return_count_le else kthval


def binpack_greedy(items, weights, max_weight):
    current = []
    cum_weight = 0
    for item, weight in zip(items, weights):
        if cum_weight + weight > max_weight:
            yield current, cum_weight
            current = []
            cum_weight = 0
        assert weight <= max_weight, "Item cannot have weight larger than max_weight"
        cum_weight += weight
        current.append(item)
    yield current, cum_weight


def compute_needed_bits(num):
    neededbits = 0
    capacity = 1
    while (capacity < num):
        neededbits += 1
        capacity = capacity << 1
    return neededbits, capacity


MAX_BITS_PACK = 63  # If we pack 64 bits, we may get into trouble with the sign


def pack_keys(chunk):
    assert len(chunk) > 1, "Must be able to pack more than one item"
    key = None

    totalbits = 0
    for (unq, inv), bits in chunk:
        key = inv if key is None else (key << bits).bitwise_or_(inv)
        totalbits += bits
    assert totalbits <= MAX_BITS_PACK
    return key


# Faster unique: implicitly builds a tree and computes unique for each and merges
# Assumes you always want to uniqueify the inner dimension of two since then each row to be sorted is contiguous!
def unique_inverse(list_of_tensors, return_index=False, return_counts=False, device=None):
    """
    Finds unique rows/k-tuples in a n x k matrix represented as k vectors/'columns'.
    Can also accept a matrix directly in which case it must be k x n
    :param list_of_tensors:
    :param return_index:
    :param return_counts:
    :param device:
    :return:
    """

    # TODO: optimizations: we can stop when the count of a group is 1
    # TODO: for first unique calls, maybe check unique consecutive first
    # TODO: we may be able to use a better bin packing heuristic that requires less sorts
    # TODO: instead of bit shifting we may multiply by num_groups to pack even denser at some extra computation

    final = len(list_of_tensors) == 1  # If we have just one row we need to make sure to return correct outputs
    # TODO if the input consists of less than 64 bits we may be able to combine from the start (not unique first)
    queue = [
        unique1d(
            row.to(device),
            return_index=final and return_index,
            return_inverse=True,
            return_counts=final and return_counts
        )
        for row in list_of_tensors
    ]
    del list_of_tensors  # Free up some memory
    while len(queue) > 1:
        needed_bits = [compute_needed_bits(len(unq))[0] for unq, inv in queue]

        chunks = list(binpack_greedy(zip(queue, needed_bits), needed_bits, MAX_BITS_PACK))
        final = len(chunks) == 1
        # There must be at least one chunk that is combinable otherwise we have a problem
        assert any(len(chunk) > 1 for chunk, bits in chunks), "No chunk to be combined"
        # This highlights the parallel nature
        queue = []
        while(len(chunks) > 0):
            chunk, totalbits = chunks.pop(0)  # By doing loop this way, this is the only reference to the chunk

            if len(chunk) == 1:
                res = chunk.pop()  # Get the single item, output of the unique function call
                queue.append(res)  # TODO shouldn't we return res[0], this is what it was before...
                continue

            keys = pack_keys(chunk)
            del chunk  # Free some memory
            res = unique1d(
                keys,
                return_index=final and return_index,
                return_inverse=True,
                return_counts=final and return_counts
            )
            del keys
            queue.append(res)

    # We don't care about the unique
    _, index, *rest = queue.pop(0)
    if return_index:
        inv, *rest = rest
        return (inv, index, *rest)
    return index if len(rest) == 0 else (index, *rest)


def lexsort(a):
    # Performs lexsort similar to numpy
    return torch.argsort(unique_inverse(a.flip(0)))


# def lexsort_via_cupy(a):
#     with cp.cuda.Device(a.device.index):
#         return from_dlpack(cp.lexsort(cp.fromDlpack(to_dlpack(a))).toDlpack())


def diff(a, dim=0, out=None, func=torch.not_equal):
    sz = a.size(dim) - 1
    if out is None:
        out = torch.empty(sz, dtype=torch.bool, device=a.device)
    return func(torch.narrow(a, dim, 1, sz), torch.narrow(a, dim, 0, sz), out=out)


def unique_consecutive_inverse(a, dim=0, out=None):
    if out is None:
        out = torch.empty(a.size(dim), dtype=torch.long, device=a.device)
    out[0] = 0
    diff(a, out=out[1:])
    return torch.cumsum(out, 0, out=out)


def unique_consecutive(a, dim=0, return_vals=True, return_index=False, return_inverse=False, return_counts=False):
    df = torch.empty(a.size(dim), dtype=torch.bool, device=a.device)
    df[0] = True
    diff(a, dim=dim, out=df[1:])

    index = None
    unq = None
    if return_index or return_counts:
        (index,) = df.nonzero(as_tuple=True)
        if return_counts:
            counts = torch.empty(len(index), dtype=torch.long, device=a.device)
            counts[len(index) - 1] = a.size(0) - index[-1]
            diff(index, out=counts[:-1], func=torch.subtract)
        if return_vals:
            unq = torch.index_select(a, dim, index)
    else:
        # Avoid the nonzero call which is slow
        unq = torch.masked_select(a, df) if return_vals else None
    if return_inverse:
        # Important! We must compute inverse after previous calls since we change df
        df[0] = False
        inverse = torch.cumsum(df, 0)
    res = []
    if return_vals:
        res.append(unq)
    if return_index:
        res.append(index)
    if return_inverse:
        res.append(inverse)
    if return_counts:
        res.append(counts)

    return res[0] if len(res) == 1 else tuple(res)


# If this is a small subset, we only have to check that
def unique1d(a, return_index=False, return_inverse=False, return_counts=False):
    # TODO option to first check unique consequtive, only uniqify these and then restore to full unique
    # TODO option for boolean mask which indicates entries that are already unique
    # We need the inverse to return the index
    res = torch.unique(a, return_inverse=return_inverse or return_index, return_counts=return_counts)
    if not return_index:
        return res
    unq, inverse, counts = res if return_counts else (*res, None)

    # From https://github.com/rusty1s/pytorch_unique, but use tmp instead of inverse name
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    tmp, perm = inverse.flip([0]), perm.flip([0])
    index = inverse.new_empty(unq.size(0)).scatter_(0, tmp, perm)
    return (unq, index, inverse, counts) if return_counts else (unq, index, inverse)


def bincount(input, minlength=0):
    # torch.bincount is slow and scatter_sum from pytorch_scatter is not much faster
    unq, counts = torch.unique(input, return_counts=True)
    if minlength == 0:
        minlength = unq.max() + 1
    return counts.new_zeros(minlength).put_(unq.long(), counts)
