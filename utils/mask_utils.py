import numpy as np
import torch

try:
    import cupy as cp
    from utils.cupy_utils import ensure_torch_mempool_in_cupy, to_cp, to_pt
    ensure_torch_mempool_in_cupy()
except ImportError:
    cp = None


def packmask(mask):
    # Packs a bitmask into longs
    # As long as we do it once using numpy (slow) is fine
    bits_np = np.packbits(mask.contiguous().cpu().numpy(), axis=-1, bitorder='little')
    return torch.from_numpy(
        np.pad(bits_np, ((0, 0), (0, ((bits_np.shape[-1] + 7) // 8) * 8 - bits_np.shape[-1]))).view(np.int64)
    ).to(mask.device)


def view_as_uint8(a):
    assert a.is_contiguous()
    if a.device == torch.device('cpu'):
        return torch.from_numpy(a.numpy().view(np.uint8))
    assert a.stride(-1) == 1, "Last stride must be 1 for view_as_uint8 to work"
    with cp.cuda.Device(a.device.index):
        return to_pt(to_cp(a).view(cp.uint8))


def unpackbits(a):
    if a.device == torch.device('cpu'):
        return torch.from_numpy(np.unpackbits(a))
    with cp.cuda.Device(a.device.index):
        return to_pt(cp.unpackbits(to_cp(a)))


def unpack_mask(mask, view_as_binary=False, num_nodes=None, do_check=False):
    num_nodes = num_nodes or mask.size(1) * 64
    assert not view_as_binary or num_nodes == mask.size(1) * 64
    if mask.device != torch.device('cpu'):
        mask = mask.contiguous()
        if mask.size(-1) == 1 and mask.stride(-1) != 1:
            assert mask.stride(0) == 1
            mask = mask.as_strided(mask.size(), (1, 1))
        assert mask.stride(-1) == 1, "mask.stride(-1) should be 1 for unpack_mask to work!!"
        assert mask.dim() == 2
        unpacked = unpackbits(view_as_uint8(mask)[:, :(num_nodes + 7) // 8])
        if view_as_binary:
            # Not sure which nodes to select if we have subset, the first or last n?
            assert num_nodes == mask.size(1) * 64
            # This is default of cupy, bitorder big
            result = unpacked.view(mask.size(0), -1)
        else:
            # Flip bits in the way they are unpacked
            result = unpacked.view(mask.size(0), -1, 8).flip(-1).view(mask.size(0), -1)[:, :num_nodes]
        if do_check:
            checkmask = unpack_mask(mask.cpu(), view_as_binary=view_as_binary, num_nodes=num_nodes).to(mask.device)
            assert (result == checkmask).all()
        return result

    # Note: this function is only for debug purposes
    def to_np(arr):
        return arr.cpu().numpy() if torch.is_tensor(arr) else arr
    mask_np = to_np(mask) if not isinstance(mask, (list, tuple)) else np.column_stack([to_np(col) for col in mask])
    # Bitorder little so that when we print the mask, the leftmost (first digit) corresponds to 1
    unpacked_mask = np.unpackbits(np.ascontiguousarray(mask_np).view(np.uint8), axis=-1, bitorder='little')[:, :num_nodes]
    if view_as_binary:
        unpacked_mask = unpacked_mask[:, ::-1]  # So when digit 1 is set it will be the rightmost when printing
    return torch.from_numpy(unpacked_mask).to(mask.device) if torch.is_tensor(mask) else unpacked_mask