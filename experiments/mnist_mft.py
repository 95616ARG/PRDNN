"""Experiment to Modified Fine-Tune an MNIST image-recognition model."""
from collections import defaultdict
import random
from timeit import default_timer as timer
import numpy as np
from pysyrenn import Network
from pysyrenn import ReluLayer
from experiments.mnist_repair import MNISTRepair
from prdnn import FTRepair

class MNISTMFT(MNISTRepair):
    """Attempts to MFT networks to be resillient to corruptions."""
    def run(self):
        """Runs the corruption-fine-tuning experiment."""
        network = self.load_network("mnist_relu_3_100")

        assert isinstance(network.layers[-1], ReluLayer)
        network = Network(network.layers[:-1])

        self.record_artifact(network, "original", "network")

        self.which_params = int(input("Which fine-tuning params? (1 or 2): "))
        assert self.which_params in {1, 2}
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
            "train", n_lines, only_correct_on=network, corruption=self.corruption)
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
        points, labels = self.sample_like_syrenn(train_syrenn, train_labels)

        self.record_artifact(
            len(points), f"{experiment}/n_repair_points", "pickle")

        for layer in [2, 4]:
            print("::::", "Layer:", layer, "::::")
            patcher = FTRepair(network, points, labels)
            patcher.layer = layer
            patcher.norm_objective = True
            # Don't stop when full repair-set accuracy is reached, only when
            # holdout accuracy gets worse.
            patcher.auto_stop = False
            patcher.make_holdout_set()
            patcher.batch_size = 16
            # This is just a maximum epoch timeout, it will stop once all
            # constraints are met.
            patcher.epochs = 1000
            patcher.momentum = 0.9
            if self.which_params == 1:
                patcher.lr = 0.05
            else:
                patcher.lr = 0.01
            patched = patcher.compute()
            patcher.timing["syrenn_time"] = syrenn_time

            self.record_artifact(
                patcher.epoched_out, f"{experiment}/epoched_out_{layer}", "pickle")
            self.record_artifact(
                patched, f"{experiment}/patched_{layer}",
                "pickle" if patched is None else "network")
            self.record_artifact(
                patcher.timing, f"{experiment}/timing_{layer}", "pickle")
            self.record_artifact(
                patcher.accuracy_on_repair_set(patched),
                f"{experiment}/patched_efficacy_{layer}", "pickle")

    def sample_like_syrenn(self, train_syrenn, train_labels):
        points, labels = [], []
        for line, label in zip(train_syrenn, train_labels):
            start, end = line[0], line[-1]
            points.extend([start, end])
            # We always want to include the start/end
            alphas = np.random.uniform(low=0.0, high=1.0, size=(len(line) - 2))
            interpolated = start + np.outer(alphas, end - start)
            points.extend(interpolated)
            labels.extend(label for _ in range(len(interpolated) + 2))
        return points, labels

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

        test_lines, test_labels = self.get_corrupted("test", None, corruption=self.corruption)
        test_images = list(map(np.array, zip(*test_lines)))
        print("Size of drawdown, generalization sets:", len(test_lines))

        timing_cols = ["layer", "total", "syrenn", "jacobian", "solver",
                       "did_timeout", "drawdown", "generalization"]
        for experiment in sorted(experiments.keys(), key=lambda n: int(n.split("_")[0])):
            print(f"~~~~ Analyzing: {experiment} ~~~~")
            # Get the patched data.
            train_syrenn = self.read_artifact(f"{experiment}/train_syrenn")
            train_labels = self.read_artifact(f"{experiment}/train_labels")
            n_repair_points = self.read_artifact(f"{experiment}/n_repair_points")
            print("Size of repair set:", n_repair_points)
            train_images = list(map(
                np.array, zip(*[(l[0], l[-1]) for l in train_syrenn])))
            print("Size of drawdown, generalization sets:", len(train_images))
            print("Number of f-hat vertex points:",
                  sum((2*len(l)) - 2 for l in train_syrenn))

            before = self.compute_accuracies(original_network,
                train_images, train_labels, test_images, test_labels)

            results = self.begin_csv(f"{experiment}/analyzed", timing_cols)
            for layer in sorted(experiments[experiment]):
                print("Layer:", layer)
                timing = self.read_artifact(f"{experiment}/timing_{layer}")
                patched = self.read_artifact(f"{experiment}/patched_{layer}")
                efficacy = 100 * self.read_artifact(f"{experiment}/patched_efficacy_{layer}")
                epoched_out = self.read_artifact(f"{experiment}/epoched_out_{layer}")

                record = timing.copy()
                record["layer"] = layer
                record["syrenn"] = record["syrenn_time"]
                del record["syrenn_time"]

                after = self.compute_accuracies(patched,
                    train_images, train_labels, test_images, test_labels)
                print("\tTime (seconds):", timing["total"])
                if epoched_out:
                    print("\t(Timed Out)")

                record["drawdown"] = (before["test_identity"]
                                      - after["test_identity"])
                record["generalization"] = (after["test_corrupted"]
                                            - before["test_corrupted"])

                print("\tDrawdown:", record["drawdown"])
                print("\tGeneralization:", record["generalization"])
                print("\tEfficacy:", efficacy, "%")

                self.write_csv(results, record)
        return True

if __name__ == "__main__":
    np.random.seed(24)
    random.seed(24)
    MNISTMFT("mnist_mft").main()
