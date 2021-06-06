"""Methods for patching deep neural networks."""
import random
import sys
import os
from timeit import default_timer as timer
import torch
import numpy as np
from scipy import sparse
from tqdm import tqdm
from pysyrenn.frontend import Network, FullyConnectedLayer
from pysyrenn.frontend import Conv2DLayer, ReluLayer
from pysyrenn.frontend import ConcatLayer, HardTanhLayer
from prdnn.ddnn import DDNN, LINEAR_LAYERS
from prdnn.provable_repair import ProvableRepair

class FTRepair(ProvableRepair):
    """Helper for patching a DDNN.
    """
    def __init__(self, network, inputs, labels):
        super().__init__(network, -1, inputs, labels)
        self.epochs = 100
        self.batch_size = 16
        self.lr = 0.01
        self.momentum = 0.9
        self.auto_stop = True
        self.norm_objective = False
        self.layer = None
        self.holdout_set = None
        self.verbose = False

    def maybe_print(self, *messages):
        if self.verbose:
            print(*messages)

    def compute(self):
        network = Network.deserialize(self.network.serialize())

        if self.layer is not None:
            for param in self.get_parameters(network):
                param.requires_grad = False

        parameters = self.get_parameters(network, self.layer)
        for param in parameters:
            param.requires_grad = True

        if self.norm_objective:
            original_parameters = [param.detach().clone() for param in parameters]
            for param in original_parameters:
                # Do not train these, they're just for reference.
                param.requires_grad = False

        start = timer()
        optimizer = torch.optim.SGD(parameters, lr=self.lr, momentum=self.momentum)
        indices = list(range(len(self.inputs)))
        random.seed(24)
        self.epoched_out = None
        holdout_n_correct = self.holdout_n_correct(network)
        for epoch in range(self.epochs):
            # NOTE: In the paper, we checked this _after_ the inner loop. It
            # should only make a difference in the case where the network
            # already met the specification, so should make no difference to
            # the results.
            if self.auto_stop and self.is_done(network):
                self.maybe_print("100% training accuracy!")
                self.epoched_out = False
                break
            random.shuffle(indices)
            losses = []
            for batch_start in range(0, len(self.inputs), self.batch_size):
                batch = slice(batch_start, batch_start + self.batch_size)
                inputs = torch.tensor([self.inputs[i] for i in indices[batch]])
                labels = torch.tensor([self.labels[i] for i in indices[batch]])
                # representatives = [self.representatives[i] for i in indices[batch]]

                optimizer.zero_grad()
                output = network.compute(inputs)
                loss = torch.nn.functional.cross_entropy(output, labels)
                if self.norm_objective:
                    for curr_param, og_param in zip(parameters, original_parameters):
                        delta = (curr_param - og_param).flatten()
                        loss += torch.linalg.norm(delta, ord=2)
                        loss += torch.linalg.norm(delta, ord=float("inf"))
                loss.backward()
                losses.append(loss)
                optimizer.step()
            self.maybe_print("Average Loss:", torch.mean(torch.tensor(losses)))
            if self.holdout_set is not None:
                new_holdout_n_correct = self.holdout_n_correct(network)
                self.maybe_print("New holdout n correct:", new_holdout_n_correct, "/", len(self.holdout_set))
                if new_holdout_n_correct < holdout_n_correct:
                    self.maybe_print("Holdout accuracy dropped, ending!")
                    break
                holdout_n_correct = new_holdout_n_correct
        else:
            self.epoched_out = True
        for param in parameters:
            param.requires_grad = False
        self.timing = dict({
            "total": timer() - start,
        })
        return network

    def is_done(self, network):
        for batch_start in range(0, len(self.inputs), self.batch_size):
            batch = slice(batch_start, batch_start + self.batch_size)
            inputs = torch.tensor(self.inputs[batch])
            labels = torch.tensor(self.labels[batch])
            output = torch.argmax(network.compute(inputs), axis=1)
            if not torch.all(output == labels):
                return False
        return True

    def accuracy_on_repair_set(self, network):
        n_correct = 0
        for batch_start in range(0, len(self.inputs), self.batch_size):
            batch = slice(batch_start, batch_start + self.batch_size)
            inputs = torch.tensor(self.inputs[batch])
            labels = torch.tensor(self.labels[batch])
            output = torch.argmax(network.compute(inputs), axis=1)
            n_correct += torch.sum(output == labels)
        return n_correct / len(self.inputs)

    def holdout_n_correct(self, network):
        if self.holdout_set is None:
            return None
        n_correct = 0
        for batch_start in range(0, len(self.holdout_set), self.batch_size):
            batch = slice(batch_start, batch_start + self.batch_size)
            inputs = torch.tensor(self.holdout_set[batch])
            labels = torch.tensor(self.holdout_labels[batch])
            output = torch.argmax(network.compute(inputs), axis=1)
            n_correct += torch.sum(output == labels)
        return n_correct

    def make_holdout_set(self):
        assert self.holdout_set is None
        indices = list(range(len(self.inputs)))
        random.shuffle(indices)
        holdout_indices = indices[:len(indices)//4]
        self.holdout_set = self.inputs[holdout_indices]
        self.holdout_labels = self.labels[holdout_indices]
        self.inputs = [x for i, x in enumerate(self.inputs)
                       if i not in holdout_indices]
        self.labels = [x for i, x in enumerate(self.labels)
                       if i not in holdout_indices]

    @classmethod
    def from_planes(cls, network, planes, labels,
                    samples_per_plane, label_fn=None):
        """Constructs a ProvableRepair to patch 2D regions.

        @planes should be a list of input 2D planes (Numpy arrays of their
            vertices in counter-clockwise order).
        @labels a list of the corresponding desired labels (integers).
        """
        points = []
        point_labels = []
        if labels is None:
            labels = [0 for i in planes]
        for vertices, label, samples in zip(planes, labels, samples_per_plane):
            coefficients = np.random.uniform(
                0., 1., size=(samples, len(vertices)))
            coefficients = (coefficients.T / np.sum(coefficients, axis=1)).T
            points.extend(list(np.matmul(coefficients, vertices)))
            if not label_fn:
                point_labels.extend(label for _ in range(samples))
        if label_fn:
            point_labels = label_fn(points)
        return cls(network, np.array(points), np.array(point_labels))


    @classmethod
    def from_spec_function(cls, network, region_plane,
                           spec_function, samples_per_plane):
        """Constructs a ProvableRepair for an input region and "Spec Function."

        @region_plane should be a single plane (Numpy array of
            counter-clockwise vertices) that defines the "region of interest"
            to patch over.
        @spec_function should take a set of input points (Numpy array) and
            return the desired corresponding labels (list/Numpy array of ints).
        """
        if len(np.asarray(region_plane).shape) == 2:
            region_plane = [region_plane]
        assert len(np.asarray(region_plane).shape) == 3
        return cls.from_planes(network, region_plane, None,
                               samples_per_plane, label_fn=spec_function)

    @classmethod
    def get_parameters(cls, network, layer=None):
        if layer is not None:
            return cls.get_parameters_layer(network.layers[layer])
        params = []
        for layer in network.layers:
            params.extend(cls.get_parameters_layer(layer))
        return params

    @classmethod
    def get_parameters_layer(cls, layer):
        if isinstance(layer, FullyConnectedLayer):
            return [layer.weights, layer.biases]
        if isinstance(layer, Conv2DLayer):
            return [layer.filter_weights, layer.biases]
        if isinstance(layer, ConcatLayer):
            return [param for in_layer in layer.input_layers
                    for param in cls.get_parameters_layer(in_layer)]
        return []
