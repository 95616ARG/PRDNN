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
from prdnn.ft_repair import FTRepair

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
    patcher = FTRepair(network, points, labels)
    patcher.layer = layer_index
    patched = patcher.compute()
    patched_layer = patched.layers[0]
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
    patcher = FTRepair(network, points, labels)
    patcher.epochs = 10000
    patched = patcher.compute()
    assert np.argmax(patched.compute(points)) == 1

if IN_BAZEL:
    main(__name__, __file__)
