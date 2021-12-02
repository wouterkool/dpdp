#!/usr/bin/python

# Copyright 2017, Gurobi Optimization, Inc.

# Solve a traveling salesman problem on a set of
# points using lazy constraints.   The base MIP model only includes
# 'degree-2' constraints, requiring each node to have exactly
# two incident edges.  Solutions to this model may contain subtours -
# tours that don't visit every city.  The lazy constraint callback
# adds new constraints to cut them off.

import argparse
import numpy as np
from utils.data_utils import load_dataset, save_dataset
from gurobipy import *


def solve_metric_tsptw(distmat, timew, threads=0):
    """
    Solves the Euclidan TSP problem to optimality using the MIP formulation 
    with lazy subtour elimination constraint generation.
    :param points: list of (x, y) coordinate 
    :return: 
    """

    n = len(distmat)

    M = timew[0, 1]  # large constant

    # Given a tuplelist of edges, find the shortest subtour

    def subtour(edges):
        unvisited = list(range(n))
        cycle = range(n + 1)  # initial length has 1 more city
        while unvisited:  # true if list is non-empty
            thiscycle = []
            neighbors = unvisited
            while neighbors:
                current = neighbors[0]
                thiscycle.append(current)
                unvisited.remove(current)
                neighbors = [j for i, j in edges.select(current, '*') if j in unvisited]
            if len(cycle) > len(thiscycle):
                cycle = thiscycle
        return cycle

    # Dictionary of Euclidean distance between each pair of points

    dist = {(i,j) :
        distmat[i, j]
        for i in range(n) for j in range(n) if i != j}

    m = Model()
    m.Params.outputFlag = False

    # Create variables

    vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='e')

    lb = {i: timew[i, 0] for i in range(n)}
    ub = {i: timew[i, 1] for i in range(n)}
    t_vars = m.addVars(range(n), vtype=GRB.INTEGER, name='t', lb=lb, ub=ub)

    # Add degree-2 constraint

    m.addConstrs(vars.sum(i,'*') == 1 for i in range(n))  # Outgoing 1
    m.addConstrs(vars.sum('*', j) == 1 for j in range(n))  # Incoming 1

    # If i == 0 and j comes after i, then we must use 0 instead of t_i (which is the highest t)
    m.addConstrs(t_vars[j] - (t_vars[i] if i != 0 else 0) >= (d + M) * vars[i, j] - M for (i, j), d in dist.items())

    # Optimize model

    m._vars = vars
    m.Params.threads = threads
    m.optimize()  # We do not need subtour elimination as we have times

    vals = m.getAttr('x', vars)
    selected = tuplelist((i,j) for i,j in vals.keys() if vals[i,j] > 0.5)
    tour = subtour(selected)
    assert len(tour) == n

    return m.objVal, tour
