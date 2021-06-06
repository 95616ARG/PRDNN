"""Experiment to patch an MNIST image-recognition model."""
from collections import defaultdict
import random
from timeit import default_timer as timer
import numpy as np
from pysyrenn import Network
from pysyrenn import ReluLayer
from experiments.experiment import Experiment
from prdnn import ProvableRepair

class MNISTRepair(Experiment):
    """Attempts to patch networks to be resillient to corruptions."""
    corruption = "fog"
    def run(self):
        """Runs the corruption-patching experiment."""
        network = self.load_network("mnist_relu_3_100")

        assert isinstance(network.layers[-1], ReluLayer)
        network = Network(network.layers[:-1])

        self.record_artifact(network, "original", "network")

        n_rows = int(input("How many rows of Table 2 to generate (1, 2, 3, or 4): "))
        for n_lines in [10, 25, 50, 100][:n_rows]:
            print(f"Running with {n_lines} lines")
            self.run_for(network, n_lines)

    def run_for(self, network, n_lines):
        """Runs experiments for a particular # of lines."""
        experiment = f"{n_lines}_lines"

        # Get the training lines. Only use lines where the original image is
        # correctly classified.
        train_lines, train_labels = self.get_corrupted(
            "train", n_lines, only_correct_on=network)
        # Compute SyReNN for each line.
        start = timer()
        train_syrenn = network.exactlines(
            train_lines, compute_preimages=True, include_post=False)
        syrenn_time = timer() - start

        # Record the SyReNNs and the labels.
        self.record_artifact(
            train_syrenn, f"{experiment}/train_syrenn", "pickle")
        self.record_artifact(
            train_labels, f"{experiment}/train_labels", "pickle")

        # Unpack the SyReNNs and associated labels to points for the patcher.
        points, representatives, labels = self.syrenn_to_points(
            train_syrenn, train_labels)

        for layer in [2, 4]:
            print("::::", "Layer:", layer, "::::")
            patcher = ProvableRepair(network, layer, points, labels,
                                 representatives=representatives)
            patcher.constraint_bufer = 0.001
            patcher.gurobi_crossover = 0
            patcher.gurobi_timelimit = 90 * 60
            patched = patcher.compute()
            patcher.timing["syrenn_time"] = syrenn_time
            patcher.timing["total"] += syrenn_time

            self.record_artifact(
                patched, f"{experiment}/patched_{layer}",
                "pickle" if patched is None else "ddnn")
            self.record_artifact(
                patcher.timing, f"{experiment}/timing_{layer}", "pickle")

    def analyze(self):
        """Analyze the patched MNIST networks.

        Reports: Time, Drawdown, and Generalization
        """
        experiments = defaultdict(list)
        for artifact in self.artifacts:
            if "timing" not in artifact["key"]:
                continue
            # 10_lines/timing_2
            n_lines, layer = artifact["key"].split("/")
            experiments[n_lines].append(int(layer.split("_")[1]))

        original_network = self.read_artifact("original")

        test_lines, test_labels = self.get_corrupted("test", None)
        test_images = list(map(np.array, zip(*test_lines)))
        print("Size of drawdown, generalization sets:", len(test_lines))

        timing_cols = ["layer", "total", "syrenn", "jacobian", "solver",
                       "did_timeout", "drawdown", "generalization"]
        for experiment in sorted(experiments.keys(), key=lambda n: int(n.split("_")[0])):
            print(f"~~~~ Analyzing: {experiment} ~~~~")
            # Get the patched data.
            train_syrenn = self.read_artifact(f"{experiment}/train_syrenn")
            train_labels = self.read_artifact(f"{experiment}/train_labels")
            train_images = list(map(
                np.array, zip(*[(l[0], l[-1]) for l in train_syrenn])))
            print("Size of repair set:", len(train_images[0]))
            print("Number of f-hat vertex points:",
                  sum((2*len(l)) - 2 for l in train_syrenn))

            before = self.compute_accuracies(original_network,
                train_images, train_labels, test_images, test_labels)

            results = self.begin_csv(f"{experiment}/analyzed", timing_cols)
            for layer in sorted(experiments[experiment]):
                timing = self.read_artifact(f"{experiment}/timing_{layer}")
                patched = self.read_artifact(f"{experiment}/patched_{layer}")

                record = timing.copy()
                record["layer"] = layer
                record["syrenn"] = record["syrenn_time"]
                del record["syrenn_time"]

                if patched is None:
                    record["drawdown"], record["generalization"] = "", ""
                else:
                    after = self.compute_accuracies(patched,
                        train_images, train_labels, test_images, test_labels)
                    print("Layer:", layer)
                    print("\tTime (seconds):", timing["total"])

                    assert after["train_identity"] == 100.
                    assert after["train_corrupted"] == 100.

                    record["drawdown"] = (before["test_identity"]
                                          - after["test_identity"])
                    record["generalization"] = (after["test_corrupted"]
                                                - before["test_corrupted"])
                    print("\tDrawdown:", record["drawdown"])
                    print("\tGeneralization:", record["generalization"])

                self.write_csv(results, record)
        return True

    def compute_accuracies(self, network, train, train_labels, test,
                           test_labels):
        """Compture train, test accuracy for a network."""
        return dict({
            "train_identity": self.accuracy(network, train[0], train_labels),
            "train_corrupted": self.accuracy(network, train[1], train_labels),
            "test_identity": self.accuracy(network, test[0], test_labels),
            "test_corrupted": self.accuracy(network, test[1], test_labels),
        })

    @staticmethod
    def accuracy(network, inputs, labels):
        """Measures network accuracy."""
        net_labels = np.argmax(network.compute(inputs), axis=1)
        return 100. * (np.count_nonzero(np.equal(net_labels, labels))
                       / len(labels))

    @classmethod
    def syrenn_to_points(cls, syrenn, line_labels):
        """Lists all endpoints in an ExactLine/SyReNN representation.

        Returns (points, representatives, labels). Representatives are
        non-vertex points which should have the same activation pattern in the
        network as the corresponding point.
        """
        points, representatives, labels = [], [], []
        for line, label in zip(syrenn, line_labels):
            for start, end in zip(line, line[1:]):
                points.extend([start, end])
                labels.extend([label, label])
                representative = (start + end) / 2.
                representatives.extend([representative, representative])
        return points, representatives, labels

    @staticmethod
    def get_corrupted(split, max_count, only_correct_on=None, corruption="fog"):
        """Returns the desired dataset."""
        random.seed(24)
        np.random.seed(24)

        all_images = [
            np
            .load(f"external/mnist_c/{corruption}/{split}_images.npy")
            .reshape((-1, 28 * 28))
            for corruption in ("identity", corruption)
        ]
        labels = np.load(f"external/mnist_c/identity/{split}_labels.npy")

        indices = list(range(len(labels)))
        random.shuffle(indices)
        labels = labels[indices]
        all_images = [images[indices] / 255. for images in all_images]

        if only_correct_on is not None:
            outputs = only_correct_on.compute(all_images[0])
            outputs = np.argmax(outputs, axis=1)

            correctly_labelled = (outputs == labels)

            all_images = [images[correctly_labelled] for images in all_images]
            labels = labels[correctly_labelled]

        lines = list(zip(*all_images))
        if max_count is not None:
            lines = lines[:max_count]
            labels = labels[:max_count]
        return lines, labels

if __name__ == "__main__":
    np.random.seed(24)
    random.seed(24)
    MNISTRepair("mnist_repair").main()
