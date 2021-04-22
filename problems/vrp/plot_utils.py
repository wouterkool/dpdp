import numpy as np
from matplotlib import pyplot as plt

from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


# Code inspired by Google OR Tools plot:
# https://github.com/google/or-tools/blob/fb12c5ded7423d524fc6c95656a9bdc290a81d4d/examples/python/cvrptw_plot.py

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


def plot_vrp(ax1, instance, solution, heatmap, mask, markersize=5, visualize_demands=False, round_demand=False, title="VRP"):
    """
    Plot the vehicle routes on matplotlib axis ax1.
    """

    depot, loc, demand, capacity, *rest = instance
    grid_size = 1
    if len(rest) > 0:
        depot_types, customer_types, grid_size = rest
    depot = np.array(depot)
    loc = np.array(loc)
    demand = np.array(demand)
    min_routes = demand.sum() / capacity

    x_dep, y_dep = depot
    ax1.plot(x_dep, y_dep, 'sk', markersize=markersize * 4)
    ax1.set_xlim(0, grid_size)
    ax1.set_ylim(0, grid_size)

    if mask is not None:
        frm, to = np.triu(mask).nonzero()
        coords = np.concatenate((depot[None], loc))
        edges_coords = np.stack((coords[frm], coords[to]), -2)

        weights = heatmap[frm, to]
        edge_colors = np.concatenate((np.tile([1, 0, 0], (len(weights), 1)), weights[:, None]), -1)

        lc_edges = LineCollection(edges_coords, colors=edge_colors, linewidths=1)
        ax1.add_collection(lc_edges)


    # legend = ax1.legend(loc='upper center')
    total_dist = 0
    if solution is None:
        if not visualize_demands:
            xs, ys = loc.transpose()
            # color=cmap(0)
            ax1.plot(xs, ys, 'o', mfc='black', markersize=markersize, markeredgewidth=0.0)
        ax1.set_title("{}, min {:.2f} routes".format(title, min_routes))
    else:
        tour = np.array(solution)
        # route is one sequence, separating different routes with 0 (depot)
        routes = [r[r != 0] for r in np.split(tour, np.where(tour == 0)[0]) if (r != 0).any()]

        cmap = discrete_cmap(len(routes) + 2, 'nipy_spectral')
        dem_rects = []
        used_rects = []
        cap_rects = []
        qvs = []
        for veh_number, r in enumerate(routes):
            color = cmap(len(routes) - veh_number)  # Invert to have in rainbow order

            route_demands = demand[r - 1]
            coords = loc[r - 1, :]
            xs, ys = coords.transpose()

            total_route_demand = sum(route_demands)
            assert total_route_demand <= capacity
            if not visualize_demands:
                ax1.plot(xs, ys, 'o', mfc=color, markersize=markersize, markeredgewidth=0.0)

            dist = 0
            x_prev, y_prev = x_dep, y_dep
            cum_demand = 0
            for (x, y), d in zip(coords, route_demands):
                dist += np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2)

                cap_rects.append(Rectangle((x, y), 0.01, 0.1))
                used_rects.append(Rectangle((x, y), 0.01, 0.1 * total_route_demand / capacity))
                dem_rects.append(Rectangle((x, y + 0.1 * cum_demand / capacity), 0.01, 0.1 * d / capacity))

                x_prev, y_prev = x, y
                cum_demand += d

            dist += np.sqrt((x_dep - x_prev) ** 2 + (y_dep - y_prev) ** 2)
            total_dist += dist
            qv = ax1.quiver(
                xs[:-1],
                ys[:-1],
                xs[1:] - xs[:-1],
                ys[1:] - ys[:-1],
                scale_units='xy',
                angles='xy',
                scale=1,
                color=color,
                label='R{}, # {}, c {} / {}, d {:.2f}'.format(
                    veh_number,
                    len(r),
                    int(total_route_demand) if round_demand else total_route_demand,
                    int(capacity) if round_demand else capacity,
                    dist
                )
            )

            qvs.append(qv)
        ax1.set_title('{}, {} routes (min {:.2f}), total distance {:.2f}'.format(title, len(routes), min_routes, total_dist))
        ax1.legend(handles=qvs)

        pc_cap = PatchCollection(cap_rects, facecolor='whitesmoke', alpha=1.0, edgecolor='lightgray')
        pc_used = PatchCollection(used_rects, facecolor='lightgray', alpha=1.0, edgecolor='lightgray')
        pc_dem = PatchCollection(dem_rects, facecolor='black', alpha=1.0, edgecolor='black')

        if visualize_demands:
            ax1.add_collection(pc_cap)
            ax1.add_collection(pc_used)
            ax1.add_collection(pc_dem)

    return total_dist