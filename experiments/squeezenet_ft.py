"""Fine-tuning SqueezeNet on NAE dataset.
"""
from collections import defaultdict
import random
import numpy as np
# pylint: disable=import-error
from pysyrenn import Network
from pysyrenn import ReluLayer, NormalizeLayer
from pysyrenn import FullyConnectedLayer, Conv2DLayer
from prdnn import FTRepair
from experiments.squeezenet_repair import SqueezenetRepair
from imagenet_helpers import read_imagenet_images

class SqueezenetFT(SqueezenetRepair):
    """Fine-tunes SqueezeNet with the NAE dataset (Hendrycks et al.)"""
    def run(self):
        """Fine-tune Squeezenet model and record patched versions."""
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

        which_params = int(input("Which fine-tuning params? (1 or 2): "))
        assert which_params in {1, 2}
        n_rows = int(input("How many rows of Table 1 to generate (1, 2, 3, or 4): "))
        for n_points in [100, 200, 400, 800][:n_rows]:
            print("~~~~", "Points:", n_points, "~~~~")
            key = f"{n_points}_-1"

            patcher = FTRepair(
                network, train_inputs[:n_points], train_labels[:n_points])
            patcher.lr = 0.0001
            patcher.momentum = 0.0
            # This is just a maximum epoch timeout, it will stop once the
            # constraints are met.
            patcher.epochs = 500
            if which_params == 1:
                patcher.batch_size = 2
            else:
                patcher.batch_size = 16

            patched = patcher.compute()

            self.record_artifact(patcher.timing, f"{key}/timing", "pickle")
            self.record_artifact(
                patched, f"{key}/patched",
                "network" if patched is not None else "pickle")

if __name__ == "__main__":
    np.random.seed(24)
    random.seed(24)
    SqueezenetFT("squeezenet_ft").main()
