import torch
from torch.nn import functional as F

from .sorting_utils import unique_consecutive


def get_max_group_size(coo, max_group_size=None):
    if max_group_size is not None:
        return max_group_size
    counts = unique_consecutive(coo, return_vals=False, return_counts=True)
    return counts.max()


def segment_cummin_coo(vals, coo, max_group_size=None):
    return segment_cummax_coo(vals, coo, max_group_size, max_func=torch.minimum)


def segment_cummax_coo(vals, coo, max_group_size=None, max_func=torch.maximum):
    max_group_size = get_max_group_size(coo, max_group_size)

    step = 1
    max_vals = vals.clone()
    # minval = vals.min()  # Or use min of typeinfo?
    #     minval = 0
    # max_vals[0] = vals[0]
    while (step < max_group_size):
        # For some stupid reason torch.where has no out parameter
        max_vals[step:] = torch.where(coo[step:] == coo[:-step], max_func(max_vals[step:], max_vals[:-step]),
                                      max_vals[step:])

        #         torch.maximum(max_vals[step:], max_vals[:-step].masked_fill(coo[step:] != coo[:-step], minval), out=max_vals[step:])

        # Duplicate step size
        step = step << 1
    return max_vals


def scatter_sort(vals, idx):
    # Note vals must be positive since otherwise bitwise_or_ does not work well
    # assert (vals >= 0).all()
    argsort = (idx.long() << 32).bitwise_or_(vals).argsort()
    return vals.gather(0, argsort), argsort


def segment_sort_coo(vals, coo, max_group_size=None, do_checks=False):
    assert not do_checks or (vals >= 0).all()
    # Note: group size in COO should be smaller than HALF_BLOCK_SIZE
    #     BLOCK_SIZE = 1024
    #     BLOCK_SIZE = 10
    n = len(vals)

    # With very few elements we sort once (2*10^5 was found as break point)
    if n <= 2e5:
        return scatter_sort(vals, coo)

    max_group_size = get_max_group_size(coo, max_group_size)

    # We benchmarked that 128 is the best performing, then 32 (but 128 is always better) and then 1024
    if max_group_size <= 64:
        BLOCK_SIZE = 128
    elif max_group_size <= 512:
        BLOCK_SIZE = 1024
    else:
        # Fall back to method for large groups
        return scatter_sort(vals, coo)

    assert BLOCK_SIZE % 2 == 0
    HALF_BLOCK_SIZE = BLOCK_SIZE // 2

    assert max_group_size <= HALF_BLOCK_SIZE

    num_blocks = (n + (BLOCK_SIZE - 1)) // BLOCK_SIZE

    if do_checks:
        iinfo = torch.iinfo(torch.int)
        assert coo.max() <= iinfo.max
        assert vals.max() <= iinfo.max
        assert coo.min() >= 0
        assert vals.min() >= 0

    # Fill the padded values keeping one half block empty in front and at the end
    vals_pad = torch.empty((num_blocks + 1) * BLOCK_SIZE, dtype=torch.long, device=vals.device)
    key = torch.bitwise_or(coo.long() << 32, vals.int(), out=vals_pad[HALF_BLOCK_SIZE:HALF_BLOCK_SIZE + n])
    linfo = torch.iinfo(torch.long)
    vals_pad[:HALF_BLOCK_SIZE] = linfo.min
    vals_pad[HALF_BLOCK_SIZE + n:] = linfo.max

    # Now sort twice
    vals_sorted, vals_argsort = vals_pad.view(num_blocks + 1, BLOCK_SIZE).sort(-1)

    # Add offsets to make correspond to flat unpadded index
    vals_argsort.add_(
        torch.arange(num_blocks + 1, device=vals_argsort.device).mul_(BLOCK_SIZE).sub_(HALF_BLOCK_SIZE)[:, None])
    vals_sorted2, vals_argsort2 = vals_sorted.view(-1)[HALF_BLOCK_SIZE:-HALF_BLOCK_SIZE].view(num_blocks,
                                                                                              BLOCK_SIZE).sort(-1)
    final_argsort = vals_argsort.view(-1)[HALF_BLOCK_SIZE:-HALF_BLOCK_SIZE].view(num_blocks, BLOCK_SIZE).gather(-1,
                                                                                                                vals_argsort2).view(
        -1)[:n]
    final_vals_sorted = vals.gather(0, final_argsort)

    if do_checks:
        assert (vals_sorted2.view(-1)[:n].int() == final_vals_sorted).all()
        assert ((vals_sorted2.view(-1)[:n] >> 32).int() == coo).all()

    return final_vals_sorted, final_argsort


def csr_to_counts(csr):
    return csr[1:] - csr[:-1]


def counts_to_csr(counts):
    return F.pad(torch.cumsum(counts, 0), (1, 0))


def counts_to_coo(counts):
    return torch.repeat_interleave(torch.arange(len(counts), out=counts.new()), counts)


def csr_to_coo(csr):
    return counts_to_coo(csr_to_counts(csr))


def coo_to_csr(coo, out=None, minlength=0, return_unique=False, filter_unique=False):
    # Note: output will be length + 1
    # Converts [0, 0, 0, 1, 1, 2, 2] to [0, 3, 5, 7] to indicate partitions
    # assert (coo[1:] >= coo[:-1]).all()  # Assumes to be in ascending order
    # [1, 1, 1, 3, 3] converts to [0, 0, 3, 3, 5] if filter_unique == False
    # [1, 1, 1, 3, 3] converts to [0, 3, 5] if filter_unique == True
    # return_unique will always return [1, 3] since filtered unique is simply range(len(unique))
    # If values are 0 to n - 1 then filter_unique has no effect

    # This is somewhat slower it seems
    # unq, counts = torch.unique_consecutive(coo, return_counts=True)
    if return_unique or not filter_unique:
        # We need unq if we don't filter the unique values out
        unq, counts = unique_consecutive(coo, return_counts=True)
    else:
        # Slightly more efficient
        counts = unique_consecutive(coo, return_vals=False, return_counts=True)

    length = max(minlength, len(counts) if filter_unique else unq[-1] + 1)
    if out is None:
        out = torch.zeros(length + 1, dtype=torch.long, device=coo.device)
    else:
        assert out.size() == (minlength + 1,)
        out[0] = 0

    if filter_unique:
        torch.cumsum(counts, -1, out=out[1:])
    else:
        out[1:].scatter_(0, unq.long(), counts)
        torch.cumsum(out, -1, out=out)  # Inplace cumsum works
    return (out, unq) if return_unique else out