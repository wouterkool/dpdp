import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.patches import Rectangle

from utils.data_utils import load_dataset, load_heatmaps
from problems.tsptw.problem_tsptw import get_rounded_distance_matrix


# Code inspired by Google OR Tools plot:
# https://github.com/google/or-tools/blob/fb12c5ded7423d524fc6c95656a9bdc290a81d4d/examples/python/cvrptw_plot.py
def plot(ax1, loc, depot=None, demand=None, capacity=None, timew=None, solution=None, heatmap=None, mask=None, hm_symmetric=True,
         dist=None, markersize=5, visualize_demands=False, visualize_timew=None, round_demand=False, title="VRP", grid_size=1, no_legend=False):
    """
    Plot the route(s) on matplotlib axis ax1.
    """
    loc = np.array(loc)

    if depot is not None:
        depot = np.array(depot)
        coords = np.concatenate((depot[None], loc))
    else:
        coords = loc

    if demand is not None:
        demand = np.array(demand)
        min_routes = demand.sum() / capacity

    if dist is None:
        dist = scipy.spatial.distance.cdist(coords, coords)

    visualize_nodes = True
    if visualize_demands:
        assert demand is not None
        d_width, d_height = 0.01 * grid_size, 0.1 * grid_size
        # visualize_nodes = False
        cap_rects = []
        used_rects = []
        dem_rects = []
    if visualize_timew:
        assert timew is not None
        # visualize_nodes = False
        # correct time window of depot
        max_tw = (dist[0, 1:] + timew[1:, 1]).max()
        timew[0, 1] = max_tw
        # Negative width makes them end up to the left side of the node
        tw_width, tw_height = -0.01 * grid_size, 0.1 * grid_size
        span_rects = []
        timew_rects = []
        wait_rects = []

    if depot is not None:
        x_dep, y_dep = depot
        ax1.plot(x_dep, y_dep, 'sk', markersize=markersize * 4)
    ax1.set_xlim(0, grid_size)
    ax1.set_ylim(0, grid_size)

    plot_heatmap(ax1, coords, heatmap, mask, symmetric=hm_symmetric)

    # legend = ax1.legend(loc='upper center')
    total_dist = 0
    if timew is not None:
        current_time = timew[0, 0]

    if solution is None:
        if visualize_demands:
            for (x, y), d in zip(loc, demand):
                cap_rects.append(Rectangle((x, y), d_width, d_height))
                dem_rects.append(Rectangle((x, y), d_width, d_height * d / capacity))
        if visualize_timew:
            for (x, y), (l, u) in zip(coords, timew):
                span_rects.append(Rectangle((x, y), tw_width, tw_height))
                timew_rects.append(Rectangle((x, y + tw_height * l / max_tw), tw_width, tw_height * (u - l) / max_tw))

        if visualize_nodes:
            xs, ys = loc.transpose()
            # color=cmap(0)
            ax1.plot(xs, ys, 'o', mfc='black', markersize=markersize, markeredgewidth=0.0)
        ax1.set_title("{}, min {:.2f} routes".format(title, min_routes) if demand is not None else title)
    else:
        tour = np.array(solution)
        # route is one sequence, separating different routes with 0 (depot)
        if depot is not None:
            routes = [r[r != 0] for r in np.split(tour, np.where(tour == 0)[0]) if (r != 0).any()]
        else:
            routes = [tour]

        cmap = discrete_cmap(len(routes) + 2, 'nipy_spectral')
        qvs = []
        for veh_number, r in enumerate(routes):
            color = cmap(len(routes) - veh_number) #if not no_legend else 'black'  # Invert to have in rainbow order

            route_coords = coords[r, :]
            xs, ys = route_coords.transpose()

            if demand is not None:
                route_demands = demand[r - 1]
                total_route_demand = sum(route_demands)
                assert total_route_demand <= capacity
            if visualize_nodes:
                # Use color of route such that for nodes in an individual route it is clear to which route they belong
                ax1.plot(xs, ys, 'o', mfc=color if len(routes) > 1 else 'black', markersize=markersize, markeredgewidth=0.0)

            r_with_depot = np.concatenate(([0], r)) if depot is not None else r
            route_dist = dist[r_with_depot, np.roll(r_with_depot, -1, 0)].sum()
            total_dist += route_dist

            if visualize_demands:
                cum_demand = 0
                for (x, y), d in zip(route_coords, route_demands):

                    cap_rects.append(Rectangle((x, y), d_width, d_height))
                    used_rects.append(Rectangle((x, y), d_width, d_height * total_route_demand / capacity))
                    dem_rects.append(Rectangle((x, y + d_height * cum_demand / capacity), d_width, d_height * d / capacity))

                    cum_demand += d

            if timew is not None:

                # Does time reset each new route? For now assume one route (TSPTW only)
                prev = 0
                t = current_time
                for (x, y), n in zip(route_coords, r):
                    l, u = timew[n]
                    arr = t + dist[prev, n]
                    t = max(arr, l)
                    assert t <= u, f"Time window violated for node {n}: {t} is not in ({l, u})"

                    if visualize_timew:
                        span_rects.append(Rectangle((x, y), tw_width, tw_height))
                        timew_rects.append(Rectangle((x, y + tw_height * l / max_tw), tw_width, tw_height * (u - l) / max_tw))
                        wait_rects.append(Rectangle((x, y + tw_height * arr / max_tw), tw_width, tw_height * (t - arr) / max_tw))



                    prev = n
                t = t + dist[prev, 0]  # Return to depot
                # For next route, or does it reset?
                current_time = t

            if demand is not None:
                # Assume VRP
                label = 'R{}, # {}, c {} / {}, d {:.2f}{}'.format(
                    veh_number,
                    len(r),
                    int(total_route_demand) if round_demand else total_route_demand,
                    int(capacity) if round_demand else capacity,
                    route_dist,
                    ", t {:.2f}".format(current_time) if timew is not None else ""
                )
            else:
                assert len(routes) == 1
                label = None

            qv = ax1.quiver(
                xs[:-1],
                ys[:-1],
                xs[1:] - xs[:-1],
                ys[1:] - ys[:-1],
                scale_units='xy',
                angles='xy',
                scale=1,
                color=color,
                label=label,
            )

            qvs.append(qv)
        title_makespan = ", makespan {:.2f}".format(current_time) if timew is not None else ""
        if demand is None:
            title = '{}, total distance {:.2f}{}'.format(title, total_dist, title_makespan)
        else:
            title = '{}, {} routes (min {:.2f}), total distance {:.2f}{}'.format(title, len(routes), min_routes, total_dist, title_makespan)
        ax1.set_title(title)
        if label is not None and not no_legend:
            ax1.legend(handles=qvs)

    if visualize_demands:
        if cap_rects is not None:
            ax1.add_collection(PatchCollection(cap_rects, facecolor='whitesmoke', alpha=1.0, edgecolor='lightgray'))
        if used_rects is not None:
            ax1.add_collection(PatchCollection(used_rects, facecolor='lightgray', alpha=1.0, edgecolor='lightgray'))
        if dem_rects is not None:
            ax1.add_collection(PatchCollection(dem_rects, facecolor='black', alpha=1.0, edgecolor='black'))

    if visualize_timew:
        if span_rects is not None:
            ax1.add_collection(PatchCollection(span_rects, facecolor='whitesmoke', alpha=1.0, edgecolor='lightgray'))
        if timew_rects is not None:
            ax1.add_collection(PatchCollection(timew_rects, facecolor='lightgray', alpha=1.0, edgecolor='lightgray'))
        if wait_rects is not None:
            ax1.add_collection(PatchCollection(wait_rects, facecolor='black', alpha=1.0, edgecolor='black'))

    return total_dist


def discrete_cmap(N, base_cmap=None):
    """
      Create an N-bin discrete colormap from the specified input map
      """
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def plot_heatmap(ax1, coords, heatmap, mask, symmetric=True):
    if mask is not None:
        frm, to = (np.triu(mask) if symmetric else mask).nonzero()
        edges_coords = np.stack((coords[frm], coords[to]), -2)

        weights = heatmap[frm, to]
        edge_colors = np.concatenate((np.tile([1, 0, 0], (len(weights), 1)), weights[:, None]), -1)

        lc_edges = LineCollection(edges_coords, colors=edge_colors, linewidths=1, zorder=1)
        ax1.add_collection(lc_edges)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize solutions')
    parser.add_argument('--problem', type=str, default='tsp')
    parser.add_argument('--instances', type=str, required=True)
    parser.add_argument('--solutions', type=str)
    parser.add_argument('--heatmaps', type=str)
    parser.add_argument('--heatmap_threshold', type=float, default=1e-5)
    parser.add_argument('--heatmap_no_depot', action='store_true')
    parser.add_argument('--visualize_demands', action='store_true')
    parser.add_argument('--visualize_timew', action='store_true')
    parser.add_argument('--num_visualize', type=int)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--subplots', type=str, default='1x1', help='3x2 to plot 3 rows and 2 cols')
    parser.add_argument('--figscale', type=float, default=12)
    parser.add_argument('--savefile', type=str)
    parser.add_argument('--no_legend', action='store_true')

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
    # For TSPTW we have a directed heatmap
    heatmaps = load_heatmaps(args.heatmaps, symmetric=args.problem != 'tsptw')

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
                assert args.problem != 'tsp'

                adj[:, 0] = 0
                adj[0, :] = 0

        if args.problem == 'tsp':
            loc = instance
            computed_cost = plot(ax, loc, solution=solution, heatmap=heatmap, mask=adj, title=f'Instance {i}',
                                 grid_size=1, no_legend=args.no_legend)
        elif args.problem == 'tsptw':
            depot, loc, timew, grid_size = instance
            coord = np.concatenate((depot[None], loc), 0)
            dist = get_rounded_distance_matrix(coord)

            computed_cost = plot(ax, loc, depot=depot, timew=timew, solution=solution, dist=dist, hm_symmetric=False,
                                 heatmap=heatmap, mask=adj, title=f'Instance {i}', grid_size=grid_size,
                                 visualize_demands=args.visualize_demands, visualize_timew=args.visualize_timew,
                                 no_legend=args.no_legend)
        elif args.problem == 'vrp':
            depot, loc, demand, capacity, *rest = instance
            grid_size = 1
            if len(rest) > 0:
                depot_types, customer_types, grid_size = rest
            computed_cost = plot(ax, loc, depot=depot, demand=demand, capacity=capacity, solution=solution,
                                 heatmap=heatmap, mask=adj, title=f'Instance {i}', grid_size=grid_size,
                                 visualize_demands=args.visualize_demands, no_legend=args.no_legend)
        else:
            assert False, "Unknown problem"

        if solution is not None:
            assert np.allclose(cost, computed_cost), "Difference between saved cost {} and computed cost {} of solution! Are you using the right data/solutions?".format(cost, computed_cost)

        if i + 1 == end or subplot_idx + 1 == rows * cols:

            if args.savefile is not None:
                plt.savefig(args.savefile, bbox_inches='tight')

            # Finalize/show plot
            plt.show(block=True)

            fig, axarr, subplot_idx = make_subplots()
        else:
            subplot_idx += 1
