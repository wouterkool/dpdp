import torch

from utils.sorting_utils import kthvalue_via_cupy, kthvalue, unique_inverse, bincount
from utils.scatter_utils import csr_to_counts, counts_to_csr, coo_to_csr
from torch_scatter import scatter_max


class SimpleBatchTopK:
    """
    Simple implementation of BatchTopK for illustrative purposes. Does not support (yet) computation of a bound
    for each entry of the batch, therefore the StreamingTopK implementation is more efficient in combination with DP.
    """

    def __init__(self, capacity):

        self.capacity = capacity
        self._payloads = []
        self._keys = []
        self._batch_ids = []
        self._num_items = 0  # Current number of items in the queue
        self.bound = None
        self.total_items_queued = 0  # Total number of items added to the queue ever
        self._num_items_reduced = None  # Number of items reduced in last reduction
        self._reduced = None

    def get_key(self, device=None, copy=True):
        self._reduce()
        key, payload, batch_id = self._reduced
        return key.to(device)

    def get_payload(self, device=None, copy=True):
        self._reduce()
        key, payload, batch_id = self._reduced
        return [payl.to(device) for payl in payload]

    def reset(self):
        self._payloads = []
        self._keys = []
        self._batch_ids = []
        self._num_items = 0  # Current number of items in the queue
        self.bound = None
        self.total_items_queued = 0  # Total number of items added to the queue ever
        self._num_items_reduced = None  # Number of items reduced in last reduction
        self._reduced = None

    def enqueue(self, key, payload, batch_ids, already_bounded=False):
        self._keys.append(key)
        self._payloads.append(payload)
        self._batch_ids.append(batch_ids)
        self.total_items_queued += len(key)

    def get_num_items_reduced(self):
        num = self._num_items_reduced
        self._num_items_reduced = None
        return num

    def _reduce(self):
        keys = torch.cat(self._keys, 0)
        batch_ids = torch.cat(self._batch_ids, 0)

        inverse, counts = unique_inverse([batch_ids, keys], return_counts=True)
        argsort = torch.argsort(inverse)

        batch_ids_sorted = batch_ids[argsort]
        assert (batch_ids_sorted[1:] >= batch_ids_sorted[:-1]).all()

        csr = coo_to_csr(batch_ids_sorted, filter_unique=True)
        offsets = torch.repeat_interleave(csr[:-1], csr_to_counts(csr))
        idx = argsort[torch.arange(len(batch_ids_sorted), out=counts.new()) < offsets + self.capacity]
        self._num_items_reduced = len(keys)
        self._reduced = (keys[idx], [torch.cat(pl, 0)[idx] for pl in zip(*self._payloads)], batch_ids[idx])


class StreamingTopK:
    """
    Streaming topK implementation that also supports to track a batch of TopK's. This implementation pre-allocates the
    memory buffer which can reduce memory fragmentation when using a large capacity. For solving a single instance,
    supports different (experimental) implementations for topk for very large capacities (more than 10M).
    The methods vary in terms of memory/computation/cupy-dependency and may be unstable, so unless you care about
    time/memory performance for large capacities it is recommended to use the default 'sort'.
    The queue keeps track of the 'top k' with *smallest* keys, and keeps a running upper bound for the k-th value.
    New entries exceeding this value can never be part of the top k and do not need to be added.
    For batch_size > 1, this bound is a tensor and the default is typeinfo.max if there is no bound yet.
    The implementation has different parameters which affect the memory/computation efficiency of the implementation.
    These parameters also affect how often the bound is computed.
    """

    def __init__(self, capacity, dtype=torch.float, device=None, reduce_to_extra_factor=1.1, start_when_extra_factor=1.8, alloc_size_factor=2.0, payload_dtypes=None, kthvalue_method=None, verbose=False, batch_size=1):
        # The tradeoff in the reduce parameters is very subtle.
        # For the topk it seems most efficient to reduce to 50% when we have twice the buffer size
        # although this requires much more memory and makes the bound weaker then if we reduce earlier
        # but that takes extra cost in the binary search

        batch_capacity = batch_size * capacity
        self.memory_size = int(max(batch_capacity * alloc_size_factor, 1000 * min(batch_size, 10000))) # At least 1000 / instance to not over'optimize' small cases
        self._key = torch.empty(self.memory_size, dtype=dtype, device=device)
        self._payload = [torch.empty(self.memory_size, dtype=pl_dt, device=device) for pl_dt in payload_dtypes]
        assert batch_size <= 32000
        self._batch_ids = torch.empty(self.memory_size, dtype=torch.uint8 if batch_size < 256 else torch.int16, device=device)
        self.capacity = capacity
        self.batch_size = batch_size
        self.batch_capacity = batch_capacity
        assert reduce_to_extra_factor <= start_when_extra_factor <= alloc_size_factor
        self.reduce_to_size = int(batch_capacity * reduce_to_extra_factor)
        self.start_when_queue_size = int(batch_capacity * start_when_extra_factor)
        self.device = device
        self.verbose = verbose
        if kthvalue_method is None or kthvalue_method == 'auto':
            self.kthvalue_method = 'sort'
            # Uncommenting below may give you better performance but is somewhat experimental
            # if self.batch_size == 1:
            #     self.kthvalue_method = ('kthvalue' if device == torch.device('cpu') else 'cupy_kthvalue' if batch_capacity <= 1e6 else 'mykthvalue')
        else:
            self.kthvalue_method = kthvalue_method

        # These are properties that need to be reset
        self._num_items = 0  # Current number of items in the queue
        self._batch_num_items_lb = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._infbound = (torch.finfo if dtype.is_floating_point else torch.iinfo)(dtype).max
        self.bound = None if batch_size == 1 else self._key.new_full((self.batch_size, ), self._infbound)
        self.total_items_queued = 0  # Total number of items added to the queue ever
        self._num_items_reduced = None  # Number of items reduced in last reduction

    def get_key(self, device=None, copy=True):
        self._reduce(True)
        assert self._num_items <= self.batch_capacity
        key = self._key[:self._num_items]
        # Always return a copy since we reuse the buffer, unless explicitly asked for!
        return torch.empty_like(key, device=device).copy_(key) if copy else key.to(device)

    def get_payload(self, device=None, copy=True):
        self._reduce(True)
        assert self._num_items <= self.batch_capacity
        # Always return a copy since we reuse the buffer, unless explicitly asked for!
        return [
            torch.empty_like(pl, device=device).copy_(pl) if copy else pl.to(device)
            for pl in
            (pl[:self._num_items] for pl in self._payload)
        ]

    def reset(self):
        self._num_items = 0  # Current number of items in the queue
        self._batch_num_items_lb = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        self.bound = None if self.batch_size == 1 else self._key.new_full((self.batch_size, ), self._infbound)
        self.total_items_queued = 0  # Total number of items added to the queue ever
        self._num_items_reduced = None  # Number of items reduced in last reduction

    def get_num_items_reduced(self):
        num = self._num_items_reduced
        self._num_items_reduced = None
        return num

    def get_remaining_memory_size(self):
        return self.memory_size - self._num_items

    def apply_bound(self, key, payload, batch_ids=None, min_reduction=0.2):
        if self.bound is None:
            return key, payload, batch_ids

        bound = self.bound.gather(0, batch_ids) if batch_ids is not None else self.bound
        msk = key < bound
        assert batch_ids is None or len(self.bound) == self.batch_size
        if msk.sum() > (1 - min_reduction) * self.batch_size:
            return key, payload, batch_ids
        idx = msk.nonzero(as_tuple=False).flatten()
        return (
            key.gather(0, idx),
            [payl.gather(0, idx) for payl in payload],
            batch_ids.gather(0, idx) if batch_ids is not None else None
        )

    def enqueue(self, key, payload, batch_ids=None, already_bounded=False):
        self.total_items_queued += len(key)
        if not already_bounded:
            key, payload, batch_ids = self.apply_bound(key, payload, batch_ids)
        # assert batch_ids is None or (batch_ids == 0).all(), "Batch not supported, use BatchTopK"
        num_add = key.size(0)
        remaining_memory_size = self.get_remaining_memory_size()
        while num_add > remaining_memory_size:
            # We are in bad luck as we cannot enqueue everything and have to sort
            self._add_to_memory(key[:remaining_memory_size], [pl[:remaining_memory_size] for pl in payload], batch_ids[:remaining_memory_size] if self.batch_size > 1 else None)
            key = key[remaining_memory_size:]
            payload = [pl[remaining_memory_size:] for pl in payload]
            batch_ids = batch_ids[remaining_memory_size:] if self.batch_size > 1 else None
            self._reduce()
            # After reduction, bound was updated, so filter remaining
            key, payload, batch_ids = self.apply_bound(key, payload, batch_ids)
            num_add = key.size(0)
            remaining_memory_size = self.get_remaining_memory_size()
            assert remaining_memory_size > 0
        # Now we can add everything to the buffer
        assert num_add <= remaining_memory_size
        self._add_to_memory(key, payload, batch_ids)
        del key, payload, batch_ids  # Free up memory

        # For example, start_when_queue_size = 1.5 * cap while max buffer is 2 * cap, then we reduce when we are > 1.5 cap
        # Hopefully this prevents above situation which is annoying as it requires more memory
        if self._num_items >= self.start_when_queue_size:
            self._reduce()

    def _add_to_memory(self, key, payload, batch_ids=None):
        num_add = key.size(0)
        self._key[self._num_items:self._num_items+num_add] = key
        for (payload, add_pl) in zip(self._payload, payload):
            payload[self._num_items:self._num_items+num_add] = add_pl
        if self.batch_size > 1:
            self._batch_ids[self._num_items:self._num_items+num_add] = batch_ids
        self._num_items += num_add
        if self.batch_size > 1 and (self._batch_num_items_lb < self.capacity).any():
            # Add number of items per batch entry, and see if for some entries we exceed the capacity
            # for the first time so we can define a bound
            new_batch_num_items = self._batch_num_items_lb + bincount(batch_ids, minlength=self.batch_size)
            new_bound = (new_batch_num_items >= self.capacity) & (self._batch_num_items_lb < self.capacity)
            self._batch_num_items_lb = new_batch_num_items
            if new_bound.any():
                # Note: this is not a very tight bound, we can get a tighter bound by taking the max of the first
                # 'capacity' values only per instance, instead of all values (as in the without batch case)
                # Better is to take the k-th largest value, which is not done here (to save computation)
                # but when the queue is actually reduced
                self.bound = torch.where(
                    new_bound,
                    scatter_max(self._key[:self._num_items], self._batch_ids[:self._num_items].long(), dim_size=self.batch_size)[0],
                    self.bound
                )
        elif self._num_items >= self.capacity and self.bound is None and self.batch_size == 1:
            # As soon as we have more than capacity items for the first time, we can define a bound
            # using the first 'capacity' items (gives slightly better bound than using _num_items items)
            self.bound = self._key[:self.capacity].max().cpu()

    def _reduce(self, final=False):
        if final:
            if self.batch_size > 1:
                # Note: we cannot use self._batch_num_items_lb, we must have an upper bound
                if (bincount(self._batch_ids[:self._num_items], minlength=self.batch_size) <= self.capacity).all():
                    return
            else:
                if self._num_items <= self.capacity:
                    return
        elif self._num_items <= self.start_when_queue_size:
            return

            # self.bound = key.max() if len(key) >= max_beam_size else None
        self._num_items_reduced = self._num_items
        key = self._key[:self._num_items]

        if self.verbose:
            print('Try to reduce', self._num_items, 'values')

        if self.batch_size > 1:
            assert self.kthvalue_method == 'sort'

            batch_ids = self._batch_ids[:self._num_items]

            inverse, counts = unique_inverse([batch_ids, key], return_counts=True)
            argsort = torch.argsort(inverse)

            batch_ids_sorted = batch_ids[argsort]
            assert (batch_ids_sorted[1:] >= batch_ids_sorted[:-1]).all()

            # Note: we may have some batch_ids with count == 0 if infeasible
            csr = coo_to_csr(batch_ids_sorted, minlength=self.batch_size)
            counts = csr_to_counts(csr)
            offsets = torch.repeat_interleave(csr[:-1], counts)
            idx = argsort[torch.arange(len(batch_ids_sorted), out=counts.new()) < offsets + self.capacity]

            count_le = len(idx)
            self._num_items_reduced = len(key)

            self._key[:count_le] = key.gather(0, idx)
            for pl in self._payload:
                pl[:count_le] = pl.gather(0, idx)
            self._batch_ids[:count_le] = batch_ids.gather(0, idx)

            # When all are sorted, bound is last entry per batch_id start of next - 1, but only if count >= capacity
            # Otherwise we set a non-effective bound of maxval
            self.bound = None
            if (counts >= self.capacity).any():
                # Also filter the csr representation
                csr = counts_to_csr(counts.clip(max=self.capacity))
                assert csr[-1] == count_le
                maxval = (torch.finfo if torch.is_floating_point(self._key) else torch.iinfo)(self._key.dtype).max
                self.bound = torch.where(counts >= self.capacity, self._key[csr[1:] - 1], self._key.new_tensor(maxval))

        elif self.kthvalue_method == 'sort' or self.kthvalue_method == 'topk':
            vals, inds = torch.sort(key) if self.kthvalue_method == 'sort' else torch.topk(key, self.capacity)
            self._key[:self.capacity] = vals[:self.capacity]
            for pl in self._payload:
                pl[:self.capacity] = pl.gather(0, inds[:self.capacity])

            count_le = self.capacity
            if self.kthvalue_method == 'sort':
                self.bound = self._key[self.capacity-1].cpu()  # To prevent cross device issues
            else:
                # Not sure if we can rely on sorting
                self.bound = self._key[:self.capacity].max().cpu()
        else:
            if (self.kthvalue_method == 'cupy_kthvalue' or final) and key.device != torch.device('cpu'):  # For exactly getting k, it seems always fastest to use cupy
                # kth_val = kthvalue_via_cupy(key.to(torch.device('cuda:0')), self.capacity).to(key.device)
                kth_val = kthvalue_via_cupy(key, self.capacity)
                mask = key <= kth_val
                count_le = mask.sum()
            elif self.kthvalue_method == 'mykthvalue':

                mask = torch.empty_like(key, dtype=torch.bool)
                # We don't need to completely find topk, roughly is ok until final (1.01 if you care about memory) otherwise (1.1)
                kth_val, count_le = kthvalue(key, self.capacity, outmask=mask, return_count_le=True,
                                             min_steps=1, early_stop_max_k=self.reduce_to_size if not final else None)
            elif self.kthvalue_method == 'sort_kthvalue':
                kth_val = torch.sort(key).values[self.capacity]
                mask = key <= kth_val
                count_le = mask.sum()
            else:
                assert self.kthvalue_method == 'kthvalue'
                kth_val, _ = torch.kthvalue(key, self.capacity)
                mask = key <= kth_val
                count_le = mask.sum()

            # We don't care if we get a few more elements unless we desire the final result
            if count_le > self.capacity and final:
                # If desire final, we should have searched for the exact k-th value
                # If we still get more in the count, this means we have duplicate k-th values
                # Find these and remove (count_le - self.capacity) from them by setting them False in the mask
                (ind_eq,) = (key == kth_val).nonzero(as_tuple=True)
                assert len(ind_eq) > count_le - self.capacity
                # mask[ind_eq[:-(count_le - self.capacity)]] = False
                torch.index_fill(mask, 0, ind_eq[:-(count_le - self.capacity)], False)
                del ind_eq  # Free up some memory
                count_le = self.capacity

            (idx_enter_from_topk, ) = mask[count_le:].nonzero(as_tuple=True)
            (idx_enter_to_topk, ) = mask.logical_not_()[:count_le].nonzero(as_tuple=True)
            del mask  # Release some memory

            self._key.scatter_(0, idx_enter_to_topk, self._key[count_le:].gather(0, idx_enter_from_topk))
            for pl in self._payload:
                pl.scatter_(0, idx_enter_to_topk, pl[count_le:].gather(0, idx_enter_from_topk))

            self.bound = self._key[:count_le].max().cpu()  # To prevent cross device issues

        if self.verbose and self.batch_size == 1:
            print("Bound {:.2f}".format(self.bound.item()))
            # Bound is top k of _total_items_queued so this gives the quantile
            # This is the expected fraction of future items that you would expect to be below the bound
            print("Bound quantile: {} / {} = {:.2f}".format(count_le, self.total_items_queued,
                                                            count_le / self.total_items_queued))
        self._num_items = int(count_le)
