"""Tests the methods in netpatch.py
"""
import numpy as np
import torch
import pytest
from pysyrenn.frontend import Network, ReluLayer, FullyConnectedLayer
try:
    from external.bazel_python.pytest_helper import main
    IN_BAZEL = True
except ImportError:
    IN_BAZEL = False
from prdnn.provable_repair import ProvableRepair

def test_already_good():
    """Point-wise test case where all constraints are already met.

    We want to make sure that it doesn't change any weights unnecessarily.
    """
    network = Network([
        FullyConnectedLayer(np.eye(2), np.zeros(shape=(2,))),
        ReluLayer(),
    ])
    layer_index = 0
    points = [[1.0, 0.5], [2.0, -0.5], [4.0, 5.0]]
    labels = [0, 0, 1]
    patcher = ProvableRepair(network, layer_index, points, labels)
    patched = patcher.compute()
    patched_layer = patched.value_layers[0]
    assert np.allclose(patched_layer.weights.numpy(), np.eye(2))
    assert np.allclose(patched_layer.biases.numpy(), np.zeros(shape=(2,)))

def test_one_point():
    """Point-wise test case with only one constraint.

    This leads to weight bounds that are unbounded-on-one-side.
    """
    # Where the weight is too big.
    network = Network([
        FullyConnectedLayer(np.eye(2), np.zeros(shape=(2,))),
        ReluLayer(),
        FullyConnectedLayer(np.array([[1.0, 0.0], [1.0, 0.0]]),
                            np.array([0.0, 0.0])),
    ])
    layer_index = 0
    points = [[1.0, 0.5]]
    labels = [1]
    patcher = ProvableRepair(network, layer_index, points, labels)
    patched = patcher.compute()
    assert np.argmax(patched.compute(points)) == 1

    # Where the weight is too small.
    network = Network([
        FullyConnectedLayer(np.eye(2), np.array([0.0, 0.0])),
        ReluLayer(),
        FullyConnectedLayer(np.array([[1.0, 0.0], [1.0, 0.0]]), np.array([0.0, 2.0])),
    ])
    layer_index = 0
    points = [[1.0, 0.0]]
    labels = [0]
    patcher = ProvableRepair(network, layer_index, points, labels)
    patched = patcher.compute()
    assert np.argmax(patched.compute(points)) == 0

def test_optimal():
    """Point-wise test case that the greedy algorithm can solve in 1 step.

    All it needs to do is triple the second component.
    """
    network = Network([
        FullyConnectedLayer(np.eye(2), np.zeros(shape=(2,))),
        ReluLayer(),
    ])
    layer_index = 0
    points = [[1.0, 0.5], [2.0, -0.5], [5.0, 4.0]]
    labels = [1, 0, 1]
    patcher = ProvableRepair(network, layer_index, points, labels)
    patched = patcher.compute()
    assert patched.differ_index == 0
    assert np.count_nonzero(
            np.argmax(patched.compute(points), axis=1) == labels) == 3

def test_from_planes():
    """Test case to load key points from a set of labeled 2D polytopes.
    """
    if not Network.has_connection():
        pytest.skip("No server connected.")

    network = Network([
        ReluLayer(),
        FullyConnectedLayer(np.eye(2), np.zeros(shape=(2,)))
    ])
    layer_index = 1
    planes = [
        np.array([[-1.0, -3.0], [-0.5, -3.0], [-0.5, 9.0], [-1.0, 9.0]]),
        np.array([[8.0, -2.0], [16.0, -2.0], [16.0, 6.0], [8.0, 6.0]]),
    ]
    labels = [1, 0]
    patcher = ProvableRepair.from_planes(network, layer_index, planes, labels)
    assert patcher.network is network
    assert patcher.layer_index is layer_index
    assert len(patcher.inputs) == (4 + 2) + (4 + 2)
    true_key_points = list(planes[0])
    true_key_points += [np.array([-1.0, 0.0]), np.array([-0.5, 0.0])]
    true_key_points += list(planes[1])
    true_key_points += [np.array([8.0, 0.0]), np.array([16.0, 0.0])]
    true_labels = ([1] * 6) + ([0] * 6)
    for true_point, true_label in zip(true_key_points, true_labels):
        try:
            i = next(i for i, point in enumerate(patcher.inputs)
                     if np.allclose(point, true_point))
        except StopIteration:
            assert False
        assert true_label == patcher.labels[i]

def test_from_spec():
    """Test case to load key points from a spec function.
    """
    if not Network.has_connection():
        pytest.skip("No server connected.")

    network = Network([
        ReluLayer(),
        FullyConnectedLayer(np.eye(2), np.zeros(shape=(2,)))
    ])
    layer_index = 1
    region_of_interest = np.array([
        [0.5, -3.0],
        [1.0, -3.0],
        [1.0, 9.0],
        [0.5, 9.0],
    ])
    spec_fn = lambda i: np.isclose(i[:, 0], 1.0).astype(np.float32)
    patcher = ProvableRepair.from_spec_function(network, layer_index,
                                            region_of_interest, spec_fn)
    assert patcher.network is network
    assert patcher.layer_index is layer_index
    assert len(patcher.inputs) == (4 + 2)
    true_key_points = list(region_of_interest)
    true_key_points += [np.array([0.5, 0.0]), np.array([1.0, 0.0])]
    true_labels = [0, 1, 1, 0, 0, 1]
    for true_point, true_label in zip(true_key_points, true_labels):
        try:
            i = next(i for i, point in enumerate(patcher.inputs)
                     if np.allclose(point, true_point))
        except StopIteration:
            assert False
        assert true_label == patcher.labels[i]

if IN_BAZEL:
    main(__name__, __file__)
