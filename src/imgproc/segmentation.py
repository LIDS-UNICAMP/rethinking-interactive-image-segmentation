from __future__ import annotations

import sys
import numpy as np
import numba as nb
import higra as hg

from pyift.shortestpath import seed_competition
from scipy.sparse import csr_matrix

from skimage.segmentation import relabel_sequential

from typing import Tuple


@nb.njit()
def edges_from_mask(mask: np.ndarray, gradient: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    @return: (source, target, weights, reference)
    """
    reference = np.empty(mask.shape, dtype=np.int32)
    count = 0
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x]:
                reference[y, x] = count
                count += 1

    sources = np.empty(count * 2, dtype=np.int32)
    targets = np.empty(count * 2, dtype=np.int32)
    weights = np.empty(count * 2)

    def update_fun(p, q, index):
        sources[index] = reference[p]
        targets[index] = reference[q]
        weights[index] = (gradient[p] + gradient[q]) / 2.0 + 1e-8

    count = 0
    for y in range(mask.shape[0] - 1):
        for x in range(mask.shape[1] - 1):
            cur = (y, x)
            if mask[cur]:
                neigh = (y, x + 1)
                if mask[neigh]:
                    update_fun(cur, neigh, count)
                    count += 1
                neigh = (y + 1, x)
                if mask[neigh]:
                    update_fun(cur, neigh, count)
                    count += 1

    for y in range(mask.shape[0] - 1):
        cur = (y, mask.shape[1] - 1)
        if mask[cur]:
            neigh = (y + 1, mask.shape[1] - 1)
            if mask[neigh]:
                update_fun(cur, neigh, count)
                count += 1

    for x in range(mask.shape[1] - 1):
        cur = (mask.shape[0] - 1, x)
        if mask[cur]:
            neigh = (mask.shape[0] - 1, x + 1)
            if mask[neigh]:
                update_fun(cur, neigh, count)
                count += 1

    return sources[:count], targets[:count], weights[:count]


def graph_from_component(comp: Component):
    sources, targets, weights = edges_from_mask(comp.mask, comp.gradient)
    bool_mask = comp.mask != 0
    graph = hg.UndirectedGraph()
    graph.add_vertices(bool_mask.sum())
    graph.add_edges(sources, targets)
    return graph, weights


def comp_watershed(comp: Component):
    ref_im = comp.ref_image
    graph, edge_weights = graph_from_component(comp)
    tree, altitudes = ref_im.ws_params['hierfun'](graph, edge_weights)
    relevant_nodes = hg.attribute_contour_strength(tree, edge_weights) < ref_im.ws_params['frontier']
    tree, node_map = hg.simplify_tree(tree, relevant_nodes)
    altitudes = altitudes[node_map]
    vertex_labels = hg.labelisation_horizontal_cut_from_threshold(tree, altitudes, ref_im.ws_params['altitude'])
    segments, _, _ = relabel_sequential(vertex_labels)
    labels = np.zeros(comp.mask.shape, dtype=np.int32)
    labels[comp.mask.astype(np.bool)] = segments
    return labels


def seeded_watershed(comp: Component, positives: Tuple[Tuple[int, int]],
                     negatives: Tuple[Tuple[int, int]]) -> np.ndarray:
    graph, edge_weights = graph_from_component(comp)
    real_size = graph.num_vertices()
    reference = np.full(comp.mask.shape, -1, dtype=np.int32)
    reference[comp.mask != 0, ...] = np.arange(0, real_size)

    # adding artificial nodes
    graph.add_vertices(2)
    for coord in positives:
        coord = (coord[0] - comp.bbox[1], coord[1] - comp.bbox[0])
        graph.add_edge(reference[coord], real_size)
    for coord in negatives:
        coord = (coord[0] - comp.bbox[1], coord[1] - comp.bbox[0])
        graph.add_edge(reference[coord], real_size + 1)

    edge_weights = np.concatenate((edge_weights, np.zeros(len(positives) + len(negatives))))
    seeds = np.zeros(graph.num_vertices(), dtype=np.int64)
    seeds[real_size] = 1
    seeds[real_size + 1] = 2

    vertex_labels = hg.labelisation_seeded_watershed(graph, edge_weights, seeds)
    labels = np.zeros_like(comp.mask)
    labels[comp.mask != 0, ...] = vertex_labels[:-2]

    labels[labels == 2] = 0
    return labels


def ift(comp: Component, positives: Tuple[Tuple[int, int]],
        negatives: Tuple[Tuple[int, int]]) -> np.ndarray:
    
    sources, targets, weights = edges_from_mask(comp.mask, comp.gradient)
    bool_mask = comp.mask != 0
    real_size = bool_mask.sum()

    graph = csr_matrix((weights, (sources, targets)), shape=(real_size, real_size)) +\
            csr_matrix((weights, (targets, sources)), shape=(real_size, real_size))

    reference = np.full(comp.mask.shape, -1, dtype=np.int32)
    reference[bool_mask, ...] = np.arange(0, real_size)
    seeds = np.zeros(real_size, dtype=np.int32)
    
    for coord in positives:
        coord = (coord[0] - comp.bbox[1], coord[1] - comp.bbox[0])
        seeds[reference[coord]] = 1
    for coord in negatives:
        coord = (coord[0] - comp.bbox[1], coord[1] - comp.bbox[0])
        seeds[reference[coord]] = 2

    _, _, _, vertex_labels = seed_competition(seeds, graph=graph)
    labels = np.zeros_like(comp.mask)
    labels[bool_mask, ...] = vertex_labels

    labels[labels == 2] = 0
    return labels
