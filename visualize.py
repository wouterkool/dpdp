import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils.data_utils import load_dataset, load_heatmaps
from problems.vrp.plot_utils import plot_vrp


parser = argparse.ArgumentParser(description='Visualize solutions')
parser.add_argument('--problem', type=str, default='tsp')
parser.add_argument('--instances', type=str, required=True)
parser.add_argument('--solutions', type=str)
parser.add_argument('--heatmaps', type=str)
parser.add_argument('--heatmap_threshold', type=float, default=1e-5)
parser.add_argument('--heatmap_no_depot', action='store_true')
parser.add_argument('--num_visualize', type=int)
parser.add_argument('--offset', type=int, default=0)
parser.add_argument('--subplots', type=str, default='1x1', help='3x2 to plot 3 rows and 2 cols')
parser.add_argument('--figscale', type=float, default=12)

parser.add_argument('-f', action='store_true', help='Force overwrite existing results')
args = parser.parse_args()

# Read instances
instances = load_dataset(args.instances)

# Read solutions
solutions = None
if args.solutions is not None:
    solutions, extra = load_dataset(args.solutions)

# Read heatmaps
heatmaps = None
heatmaps = load_heatmaps(args.heatmaps)


rows, cols = (int(v) for v in args.subplots.split("x"))


def make_subplots():
    fig, axarr = plt.subplots(rows, cols, squeeze=False, figsize=(args.figscale * rows, args.figscale * cols))
    subplot_idx = 0
    return fig, axarr, subplot_idx


fig, axarr, subplot_idx = make_subplots()

start = args.offset
end = min(len(instances), args.offset + args.num_visualize if args.num_visualize is not None else len(instances))
for i in range(start, end):
    instance = instances[i]
    ax = axarr[subplot_idx // cols, subplot_idx % cols]

    cost, solution, duration = None, None, None
    if solutions is not None:
        if solutions[i] is not None:
            cost, solution, duration = solutions[i]
            print(f"Instance {i}, cost {cost}")
        else:
            print(f"Warning: no solution for instance {i}")

    heatmap, adj = None, None
    if heatmaps is not None:
        heatmap = np.exp(heatmaps[i])
        adj = heatmap > args.heatmap_threshold
        if args.heatmap_no_depot:
            adj[:, 0] = 0
            adj[0, :] = 0

    if args.problem == 'tsp':
        loc = instance
        # TODO
    elif args.problem == 'vrp':

        computed_cost = plot_vrp(ax, instance, solution, heatmap, adj, title=f'Instance {i}')
        if solution is not None:
            assert np.allclose(cost, computed_cost), "Difference between saved cost {} and computed cost {} of solution! Are you using the right data/solutions?".format(cost, computed_cost)

    else:
        assert False, "Unknown problem"

    if i + 1 == end or subplot_idx + 1 == rows * cols:
        # Finalize/show plot
        plt.show(block=True)

        fig, axarr, subplot_idx = make_subplots()
    else:
        subplot_idx += 1
