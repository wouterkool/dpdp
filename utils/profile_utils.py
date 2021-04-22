import time
import numpy as np
import torch


class Profiler:

    def __init__(self, dummy=False, device=None):
        self.events = []
        self.dummy = dummy
        self.device = device if device != torch.device('cpu') else None
        self.log('start')

    def log(self, name):
        if self.dummy:
            return
        # Optionally synchronize cuda before logging time
        if self.device is not None:
            torch.cuda.synchronize(self.device)
        self.events.append((name, time.time()))

    def print_profile_summary(self, step, detailed=False):
        event_names, event_times = zip(*self.events)
        total_duration = event_times[-1] - event_times[0]
        print(
            "-------------- Step {} total duration: {:.3f} ms -------------------".format(step, total_duration * 1000))
        event_durations = np.diff(event_times)
        event_names = event_names[1:]
        total_generate = sum(d for e, d in zip(event_names, event_durations) if "expansion" in e)
        total_reduce = sum(d for e, d in zip(event_names, event_durations) if "reduced" in e)
        print("Total generate expansions time {:.3f} ms".format(total_generate * 1000))
        print("Total topk selection time      {:.3f} ms".format(total_reduce * 1000))
        print(
            "Total rest time                {:.3f} ms".format((total_duration - total_generate - total_reduce) * 1000))
        maxlen = max(len(en) for en in event_names)

        if detailed:
            for i in np.argsort(-event_durations):  # Sort descending by duration
                print(("{:" + str(maxlen) + "s} {:.3f} ms").format(event_names[i], event_durations[i] * 1000))


def debug_memory(device=None):
    print('*' * 20, "Memory Dump", '*' * 20)
    if device is not None:
        print(torch.cuda.memory_summary(device))
    import gc
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.device, obj.dtype, obj.size())
        except:
            pass