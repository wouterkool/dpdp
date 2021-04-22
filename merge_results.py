import os
import argparse
from utils.data_utils import load_dataset, save_dataset
from utils.functions import ensure_backward_compatibility
from eval import print_statistics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs='+', help="Filenames of the results to combine")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument("--unsafe", action='store_true', help="Force merging results even with different settings")

    opts = parser.parse_args()

    print("Merging", opts.files)
    if opts.o is None:
        opts.o = os.path.commonprefix(opts.files) + "_merged.pkl"

    assert opts.f or not os.path.isfile(
        opts.o), "File already exists! Try running with -f option to overwrite."

    datasets = [load_dataset(filename) for filename in opts.files]

    save_opts = None
    merged_dataset = []
    for dataset, ds_opts in sorted(datasets, key=lambda k: k[1].offset):
        ensure_backwards_compatibility(ds_opts)

        if save_opts is None:
            save_opts = ds_opts
        else:
            assert ds_opts.offset == save_opts.offset + save_opts.val_size, "Offset does not match previous!"
            save_opts.val_size += ds_opts.val_size

        for k in vars(ds_opts):
            if not (k in ('offset', 'val_size', 'o') or getattr(save_opts, k) == getattr(ds_opts, k)):
                print("Warning, different values for option", k)
                assert opts.unsafe, "Not merging results with different options, add --unsafe to disable check"
        if ds_opts.val_size < ds_opts.batch_size:
            print("Warning, inaccurate total time computation if val_size per split is smaller than batch_size")
            assert opts.unsafe, "Not merging results with splits smaller than batch size, add --unsafe to disable check"

        merged_dataset.extend(dataset)

    print_statistics(merged_dataset, save_opts)

    print("Saving", opts.o)
    save_opts.o = opts.o
    save_dataset((merged_dataset, save_opts), save_opts.o)
