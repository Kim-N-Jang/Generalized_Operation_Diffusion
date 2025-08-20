import os
import warnings
from multiprocessing import Pool

import numpy as np
import scipy.sparse
import scipy.spatial
import torch
# from utils.cython_merge.cython_merge import merge_cython


def numpy_merge(points, adj_mat):
  dists = np.linalg.norm(points[:, None] - points, axis=-1)

  components = np.zeros((adj_mat.shape[0], 2)).astype(int)
  components[:] = np.arange(adj_mat.shape[0])[..., None]
  real_adj_mat = np.zeros_like(adj_mat)
  merge_iterations = 0
  for edge in (-adj_mat / dists).flatten().argsort():
    merge_iterations += 1
    a, b = edge // adj_mat.shape[0], edge % adj_mat.shape[0]
    if not (a in components and b in components):
      continue
    ca = np.nonzero((components == a).sum(1))[0][0]
    cb = np.nonzero((components == b).sum(1))[0][0]
    if ca == cb:
      continue
    cca = sorted(components[ca], key=lambda x: x == a)
    ccb = sorted(components[cb], key=lambda x: x == b)
    newc = np.array([[cca[0], ccb[0]]])
    m, M = min(ca, cb), max(ca, cb)
    real_adj_mat[a, b] = 1
    components = np.concatenate([components[:m], components[m + 1:M], components[M + 1:], newc], 0)
    if len(components) == 1:
      break
  real_adj_mat[components[0, 1], components[0, 0]] = 1
  real_adj_mat += real_adj_mat.T
  return real_adj_mat, merge_iterations


# def cython_merge(points, adj_mat):
#   with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     real_adj_mat, merge_iterations = merge_cython(points.astype("double"), adj_mat.astype("double"))
#     real_adj_mat = np.asarray(real_adj_mat)
#   return real_adj_mat, merge_iterations


def merge_tours(adj_mat, np_points, edge_index_np, sparse_graph=False, parallel_sampling=1):
  """
  To extract a tour from the inferred adjacency matrix A, we used the following greedy edge insertion
  procedure.
  • Initialize extracted tour with an empty graph with N vertices.
  • Sort all the possible edges (i, j) in decreasing order of Aij/kvi − vjk (i.e., the inverse edge weight,
  multiplied by inferred likelihood). Call the resulting edge list (i1, j1),(i2, j2), . . . .
  • For each edge (i, j) in the list:
    – If inserting (i, j) into the graph results in a complete tour, insert (i, j) and terminate.
    – If inserting (i, j) results in a graph with cycles (of length < N), continue.
    – Otherwise, insert (i, j) into the tour.
  • Return the extracted tour.
  """
  splitted_adj_mat = np.split(adj_mat, parallel_sampling, axis=0)

  if not sparse_graph:
    splitted_adj_mat = [
        adj_mat[0] + adj_mat[0].T for adj_mat in splitted_adj_mat
    ]
  else:
    splitted_adj_mat = [
        scipy.sparse.coo_matrix(
            (adj_mat, (edge_index_np[0], edge_index_np[1])),
        ).toarray() + scipy.sparse.coo_matrix(
            (adj_mat, (edge_index_np[1], edge_index_np[0])),
        ).toarray() for adj_mat in splitted_adj_mat
    ]

  splitted_points = [
      np_points for _ in range(parallel_sampling)
  ]

  if np_points.shape[0] > 1000 and parallel_sampling > 1:
    with Pool(parallel_sampling) as p:
      results = p.starmap(
          numpy_merge,
          zip(splitted_points, splitted_adj_mat),
      )
  else:
    results = [
        numpy_merge(_np_points, _adj_mat) for _np_points, _adj_mat in zip(splitted_points, splitted_adj_mat)
    ]

  splitted_real_adj_mat, splitted_merge_iterations = zip(*results)

  tours = []
  for i in range(parallel_sampling):
    tour = [0]
    while len(tour) < splitted_adj_mat[i].shape[0] + 1:
      n = np.nonzero(splitted_real_adj_mat[i][tour[-1]])[0]
      if len(tour) > 1:
        n = n[n != tour[-2]]
      tour.append(n.max())
    tours.append(tour)

  merge_iterations = np.mean(splitted_merge_iterations)
  return tours, merge_iterations


class TSPEvaluator(object):
  def __init__(self, points):
    self.dist_mat = scipy.spatial.distance_matrix(points, points)

  def evaluate(self, route):
    total_cost = 0
    for i in range(len(route) - 1):
      total_cost += self.dist_mat[route[i], route[i + 1]]
    return total_cost
