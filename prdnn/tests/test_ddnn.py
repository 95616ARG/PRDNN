"""Tests the methods in ddnn.py."""
# pylint: disable=import-error
import numpy as np
import torch
from pysyrenn import ReluLayer, FullyConnectedLayer, ArgMaxLayer
from pysyrenn import HardTanhLayer, MaxPoolLayer, StridedWindowData
try:
    from external.bazel_python.pytest_helper import main
    IN_BAZEL = True
except ImportError:
    IN_BAZEL = False
from prdnn.ddnn import DDNN

def test_compute():
    """Tests that it works for a simple example."""
    activation_layers = [
        FullyConnectedLayer(np.eye(2), np.ones(shape=(2,))),
        ReluLayer(),
        FullyConnectedLayer(2.0 * np.eye(2), np.zeros(shape=(2,))),
        ReluLayer(),
    ]
    value_layers = activation_layers[:2] + [
        FullyConnectedLayer(3.0 * np.eye(2), np.zeros(shape=(2,))),
        ReluLayer(),
    ]
    network = DDNN(activation_layers, value_layers)
    assert network.differ_index == 2
    output = network.compute([[-2.0, 1.0]])
    assert np.allclose(output, [[0.0, 6.0]])
    output = network.compute(torch.tensor([[-2.0, 1.0]])).numpy()
    assert np.allclose(output, [[0.0, 6.0]])

    activation_layers = [
        FullyConnectedLayer(np.eye(2), np.ones(shape=(2,))),
        HardTanhLayer(),
    ]
    value_layers = [
        FullyConnectedLayer(2.0 * np.eye(2), np.zeros(shape=(2,))),
        HardTanhLayer(),
    ]
    network = DDNN(activation_layers, value_layers)
    output = network.compute([[0.5, -0.9]])
    assert np.allclose(output, [[1.0, -1.8]])

    # Test HardTanh
    activation_layers = [
        FullyConnectedLayer(np.eye(2), np.ones(shape=(2,))),
        HardTanhLayer(),
    ]
    value_layers = [
        FullyConnectedLayer(2.0 * np.eye(2), np.zeros(shape=(2,))),
        HardTanhLayer(),
    ]
    network = DDNN(activation_layers, value_layers)
    output = network.compute([[0.5, -0.9]])
    assert np.allclose(output, [[1.0, -1.8]])

    # Test MaxPool
    width, height, channels = 2, 2, 2
    window_data = StridedWindowData((height, width, channels),
                                    (2, 2), (2, 2), (0, 0), channels)
    maxpool_layer = MaxPoolLayer(window_data)
    activation_layers = [
        FullyConnectedLayer(np.eye(8), np.ones(shape=(8,))),
        maxpool_layer,
    ]
    value_layers = [
        FullyConnectedLayer(-1. * np.eye(8), np.zeros(shape=(8,))),
        maxpool_layer,
    ]
    network = DDNN(activation_layers, value_layers)
    output = network.compute([[1.0, 2.0, -1.0, -2.5, 0.0, 0.5, 1.5, -3.5]])
    # NHWC, so the two channels are: [1, -1, 0, 1.5] and [2, -2.5, 0.5, -3.5]
    # So the maxes are 1.5 and 2.0, so the value layer outputs -1.5, -2.0
    assert np.allclose(output, [[-1.5, -2.0]])

def test_compute_representatives():
    """Tests that the linear-region endpoints work."""
    activation_layers = [
        FullyConnectedLayer(np.eye(1), np.zeros(shape=(1,))),
        ReluLayer(),
    ]
    value_layers = [
        FullyConnectedLayer(np.eye(1), np.ones(shape=(1,))),
        ReluLayer(),
    ]
    network = DDNN(activation_layers, value_layers)
    assert network.differ_index == 0
    points = np.array([[0.0], [0.0]])
    representatives = np.array([[1.0], [-1.0]])
    output = network.compute(points, representatives=representatives)
    assert np.array_equal(output, [[1.], [0.]])

def test_nodiffer():
    """Tests the it works if activation and value layers are identical."""
    activation_layers = [
        FullyConnectedLayer(np.eye(2), np.ones(shape=(2,))),
        ReluLayer(),
        FullyConnectedLayer(2.0 * np.eye(2), np.zeros(shape=(2,))),
        ReluLayer(),
    ]
    value_layers = activation_layers
    network = DDNN(activation_layers, value_layers)
    assert network.differ_index == 4
    output = network.compute([[-2.0, 1.0]])
    assert np.allclose(output, [[0.0, 4.0]])

def test_bad_layer():
    """Tests that unspported layers after differ_index fail."""
    # It should work if it's before the differ_index.
    activation_layers = [
        FullyConnectedLayer(np.eye(2), np.ones(shape=(2,))),
        ReluLayer(),
        FullyConnectedLayer(2.0 * np.eye(2), np.zeros(shape=(2,))),
        ArgMaxLayer(),
    ]
    value_layers = activation_layers
    network = DDNN(activation_layers, value_layers)
    assert network.differ_index == 4
    output = network.compute([[-2.0, 1.0]])
    assert np.allclose(output, [[1.0]])
    # But not after the differ_index.
    activation_layers = [
        FullyConnectedLayer(np.eye(2), np.ones(shape=(2,))),
        ReluLayer(),
        FullyConnectedLayer(2.0 * np.eye(2), np.zeros(shape=(2,))),
        ArgMaxLayer(),
    ]
    value_layers = activation_layers[:2] + [
        FullyConnectedLayer(3.0 * np.eye(2), np.zeros(shape=(2,))),
        ReluLayer(),
    ]
    network = DDNN(activation_layers, value_layers)
    assert network.differ_index == 2
    try:
        output = network.compute([[-2.0, 1.0]])
        assert False
    except NotImplementedError:
        pass

def test_serialization():
    """Tests that it correctly (de)serializes."""
    activation_layers = [
        FullyConnectedLayer(np.eye(2), np.ones(shape=(2,))),
        ReluLayer(),
        FullyConnectedLayer(2.0 * np.eye(2), np.zeros(shape=(2,))),
        ReluLayer(),
    ]
    value_layers = activation_layers[:2] + [
        FullyConnectedLayer(3.0 * np.eye(2), np.zeros(shape=(2,))),
        ReluLayer(),
    ]
    network = DDNN(activation_layers, value_layers)
    serialized = network.serialize()
    assert all(serialized == layer.serialize()
               for serialized, layer in zip(serialized.activation_layers,
                                            activation_layers))
    assert all(serialized == layer.serialize()
               for serialized, layer in zip(serialized.value_layers,
                                            value_layers[2:]))
    assert serialized.differ_index == 2

    assert DDNN.deserialize(serialized).serialize() == serialized

if IN_BAZEL:
    main(__name__, __file__)
