"""Repair SqueezeNet on the NAE dataset.
"""
from collections import defaultdict
import random
import numpy as np
# pylint: disable=import-error
from pysyrenn import Network
from pysyrenn import ReluLayer, NormalizeLayer
from pysyrenn import FullyConnectedLayer, Conv2DLayer
from prdnn import ProvableRepair
from experiments.experiment import Experiment
from imagenet_helpers import read_imagenet_images

class SqueezenetRepair(Experiment):
    """Repairs Imagenet with the NAE dataset (Hendrycks et al.)"""
    def run(self):
        """Repair Squeezenet model and record patched versions."""
        network = self.load_network("squeezenet")
        assert not isinstance(network.layers[-1], ReluLayer)
        # Add a normalize layer to the start to take the images to the
        # Squeezenet format.
        normalize = NormalizeLayer(
            means=np.array([0.485, 0.456, 0.406]),
            standard_deviations=np.array([0.229, 0.224, 0.225]))
        network = Network([normalize] + network.layers)

        # Get the trainset and record it.
        train_inputs, train_labels = self.get_train(n_labels=9)

        sorted_labels = sorted(set(train_labels))
        train_labels = list(map(sorted_labels.index, train_labels))

        self.record_artifact(train_inputs, f"train_inputs", "pickle")
        self.record_artifact(sorted_labels, f"sorted_labels", "pickle")
        self.record_artifact(train_labels, f"train_labels", "pickle")

        # Add a final layer which maps it into the subset of classes
        # considered.
        final_weights = np.zeros((1000, len(sorted_labels)))
        final_biases = np.zeros(len(sorted_labels))
        for new_label, old_label in enumerate(sorted_labels):
            final_weights[old_label, new_label] = 1.
        final_layer = FullyConnectedLayer(final_weights, final_biases)
        network = Network(network.layers + [final_layer])

        # Record the network before patching.
        self.record_artifact(network, f"pre_patching", "network")

        # All the layers we can patch.
        patchable = [i for i, layer in enumerate(network.layers)
                     if isinstance(layer, (FullyConnectedLayer, Conv2DLayer))]
        n_rows = int(input("How many rows of Table 1 to generate (1, 2, 3, or 4): "))
        for n_points in [100, 200, 400, 800][:n_rows]:
            print("~~~~", "Points:", n_points, "~~~~")
            for layer in patchable:
                print("::::", "Layer:", layer, "::::")
                key = f"{n_points}_{layer}"

                patcher = ProvableRepair(
                    network, layer,
                    train_inputs[:n_points], train_labels[:n_points])
                patcher.batch_size = 8
                patcher.gurobi_timelimit = (n_points // 10) * 60
                patcher.gurobi_crossover = 0

                patched = patcher.compute()

                self.record_artifact(patcher.timing, f"{key}/timing", "pickle")
                self.record_artifact(
                    patched, f"{key}/patched",
                    "ddnn" if patched is not None else "pickle")

    def analyze(self):
        """Compute drawdown statistics for patched models."""
        print("~~~~ Results ~~~~")
        # Get the datasets and compute pre-patching accuracy.
        network = self.read_artifact("pre_patching")
        train_inputs = self.read_artifact("train_inputs")
        train_labels = self.read_artifact("train_labels")
        sorted_labels = self.read_artifact("sorted_labels")

        test_inputs, test_labels = self.get_test(sorted_labels)

        original_train_accuracy = self.accuracy(
            network, train_inputs, train_labels)
        original_test_accuracy = self.accuracy(
            network, test_inputs, test_labels)
        print("Max size of repair set:", len(train_inputs))
        print("Size of drawdown set:", len(test_inputs))
        print("Buggy network repair set accuracy:", original_train_accuracy)
        print("Buggy network drawdown set accuracy:", original_test_accuracy)

        # Get info about the patch runs.
        by_n_points = defaultdict(list)
        by_layer = defaultdict(list)
        for artifact in self.artifacts:
            artifact = artifact["key"]
            if "timing" not in artifact:
                continue
            key = artifact.split("/")[0]
            n_points, layer = map(int, key.split("_"))
            by_n_points[n_points].append(layer)
            by_layer[layer].append(n_points)

        timing_cols = ["total", "jacobian", "solver", "did_timeout",
                       "efficacy", "drawdown"]
        n_points_csvs = dict({
            n_points:
                self.begin_csv(f"{n_points}_points", ["layer"] + timing_cols)
            for n_points in by_n_points.keys()
        })
        layer_csvs = dict({
            layer: self.begin_csv(f"{layer}_layer", ["points"] + timing_cols)
            for layer in by_layer.keys()
        })
        for n_points in sorted(by_n_points.keys()):
            print("~~~~~", "Points:", min(int(n_points), len(train_inputs)), "~~~~~")
            records_for_row = []
            for layer in sorted(by_n_points[n_points]):
                timing = self.read_artifact(f"{n_points}_{layer}/timing")
                record = timing.copy()

                patched = self.read_artifact(f"{n_points}_{layer}/patched")
                if patched is not None:
                    new_train_accuracy = self.accuracy(patched,
                                                       train_inputs[:n_points],
                                                       train_labels[:n_points])
                    new_test_accuracy = self.accuracy(patched,
                                                      test_inputs,
                                                      test_labels)
                    record["efficacy"] = new_train_accuracy
                    record["drawdown"] = (original_test_accuracy
                                          - new_test_accuracy)
                    records_for_row.append(record)
                else:
                    record["efficacy"] = 0
                    record["drawdown"] = 0

                record["layer"] = layer
                self.write_csv(n_points_csvs[n_points], record)
                del record["layer"]
                record["points"] = n_points
                self.write_csv(layer_csvs[layer], record)
            best_record = min(records_for_row, key=lambda record: record["drawdown"])
            print("\tBest drawdown:", best_record["drawdown"])
            print("\tTotal time for best drawdown (seconds):", best_record["total"])
        return True

    @staticmethod
    def get_train(n_labels=10):
        """Reads the training (patch) set from disk."""
        np.random.seed(24)
        random.seed(24)

        parent = input("ImageNet-A Dataset Path: ")
        images, labels = read_imagenet_images(parent, n_labels=n_labels)

        indices = list(range(len(labels)))
        random.shuffle(indices)
        return images[indices], labels[indices]

    @staticmethod
    def get_test(sorted_labels):
        """Reads the test set from disk.

        @sorted_labels[i] gives the synset label corresponding to the ith
        output of the model. Only returns images belonging to those classes.
        """
        np.random.seed(24)
        random.seed(24)

        parent = input("ImageNet-Val Dataset Path: ")
        images, labels = read_imagenet_images(
            parent, n_labels=None, use_labels=sorted_labels)
        labels = np.array([sorted_labels.index(l) for l in labels])

        indices = list(range(len(labels)))
        random.shuffle(indices)
        return images[indices], labels[indices]

    @staticmethod
    def accuracy(network, inputs, labels):
        """Computes accuracy on a test set."""
        out = np.argmax(network.compute(inputs), axis=1)
        return 100. * np.count_nonzero(np.equal(out, labels)) / len(labels)

if __name__ == "__main__":
    np.random.seed(24)
    random.seed(24)
    SqueezenetRepair("squeezenet_repair").main()
