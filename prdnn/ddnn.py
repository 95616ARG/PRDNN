"""Methods for describing and executing Decoupled DNNs."""
import numpy as np
from pysyrenn.frontend.network import Network
from pysyrenn.frontend import FullyConnectedLayer, Conv2DLayer
from pysyrenn.frontend import ReluLayer, HardTanhLayer
from pysyrenn.frontend import ConcatLayer, AveragePoolLayer
from pysyrenn.frontend import MaxPoolLayer, NormalizeLayer
import syrenn_proto.syrenn_pb2 as transformer_pb
import torch

# NOTE: We currently only have limited support for concat layers, namely when
# the intermediate layers are all linear.
LINEAR_LAYERS = (FullyConnectedLayer, Conv2DLayer, ConcatLayer,
                 AveragePoolLayer, NormalizeLayer)

class DDNN:
    """Implements a DDNN.

    Currently supports:
    - Arbitrary layers as long as the activation and values parameters are
      equal up to that layer.
    - Once the activation and values parameters differ, only linear (see
      above), ReLU, HardTanh, and MaxPool layers are supported. Support for
      other layer types can be added by modifying the compute(...) method.
    """
    def __init__(self, activation_layers, value_layers):
        """Constructs the new DDNN.

        @activation_layers is a list of layers defining the values of the
            activation vectors.
        @value_layers is a list of layers defining the values of the value
            vectors. Non-linear layers here will be re-interpreted using the
            corresponding decoupled value-network layer.

        Note that the number, types, and output sizes of the layers in
        @activation_layers and @values_layers should match.
        """
        self.activation_layers = activation_layers
        self.value_layers = value_layers

        self.n_layers = len(activation_layers)
        assert self.n_layers == len(value_layers)

        try:
            self.differ_index = next(
                l for l in range(self.n_layers)
                if activation_layers[l] is not value_layers[l])
        except StopIteration:
            self.differ_index = len(value_layers)

    def compute(self, inputs, representatives=None):
        """Computes the output of the Decoupled Network on @inputs.

        @inputs should be a Numpy array of inputs.
        """
        differ_index = self.differ_index
        if representatives is not None:
            differ_index = 0
        # Up to differ_index, the values and activation vectors are the same.
        pre_network = Network(self.activation_layers[:differ_index])
        mid_inputs = pre_network.compute(inputs)
        # Now we have to actually separately handle the masking when
        # activations != values.
        activation_vector = mid_inputs
        if representatives is not None:
            activation_vector = pre_network.compute(representatives)
        value_vector = mid_inputs
        for layer_index in range(differ_index, self.n_layers):
            activation_layer = self.activation_layers[layer_index]
            value_layer = self.value_layers[layer_index]
            if isinstance(activation_layer, LINEAR_LAYERS):
                if isinstance(activation_layer, ConcatLayer):
                    assert not any(
                        isinstance(input_layer, ConcatLayer)
                        for input_layer in activation_layer.input_layers)
                    assert all(
                        isinstance(input_layer, LINEAR_LAYERS)
                        for input_layer in activation_layer.input_layers)
                activation_vector = activation_layer.compute(activation_vector)
                value_vector = value_layer.compute(value_vector)
            elif isinstance(activation_layer, ReluLayer):
                mask = np.maximum(np.sign(activation_vector), 0.0)
                if isinstance(value_vector, np.ndarray):
                    value_vector *= mask
                else:
                    # NOTE: Originally this was torch.tensor(mask,
                    # dtype=torch.float). I changed to this to silence a
                    # warning from Pytorch. I don't think there will be, but it
                    # might be worth testing for a performance regression.
                    value_vector *= mask.clone().detach().float()
                activation_vector *= mask
            elif isinstance(activation_layer, HardTanhLayer):
                mask = np.ones_like(value_vector)
                value_vector[activation_vector >= 1.0] = 1.0
                value_vector[activation_vector <= -1.0] = -1.0
                np.clip(activation_vector, -1.0, 1.0, out=activation_vector)
            elif isinstance(activation_layer, MaxPoolLayer):
                activation_vector, indices = activation_layer.compute(
                    activation_vector, return_indices=True)

                value_vector = value_layer.from_indices(value_vector, indices)
            else:
                raise NotImplementedError
        return value_vector

    def serialize(self):
        """Serializes the DDNN to the Protobuf format.

        Notably, the value_net only includes layers after differ_index.
        """
        serialized = transformer_pb.MaskingNetwork()
        serialized.activation_layers.extend([
            layer.serialize() for layer in self.activation_layers
        ])
        serialized.value_layers.extend([
            layer.serialize()
            for layer in self.value_layers[self.differ_index:]
        ])
        serialized.differ_index = self.differ_index
        return serialized

    @classmethod
    def deserialize(cls, serialized):
        """Deserializes the DDNN from the Protobuf format."""
        activation_layers = serialized.activation_layers
        activation_layers = Network.deserialize_layers(activation_layers)

        value_layers = serialized.value_layers
        value_layers = Network.deserialize_layers(value_layers)

        differ_index = serialized.differ_index
        value_layers = activation_layers[:differ_index] + value_layers
        return cls(activation_layers, value_layers)
