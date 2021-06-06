"""Methods for patching the ACAS Xu network with SyReNN."""
from timeit import default_timer as timer
import numpy as np
from pysyrenn import FullyConnectedLayer
from prdnn import DDNN, FTRepair
from experiments.acas_repair import ACASRepair

class ACASMFT(ACASRepair):
    """Experiment testing patching performance on an ACAS Xu model."""
    def do_repair(self, train_regions, train_syrenn, syrenn_time):
        n_unique = len(set({tuple(point)
                            for upolytope in train_syrenn
                            for pre_poly in upolytope
                            for point in pre_poly}))
        samples_per_plane = [n_unique // len(train_regions) for _ in range(len(train_regions))]

        _patcher = FTRepair.from_spec_function(
            self.network, train_regions, self.property,
            samples_per_plane=samples_per_plane)
        self.record_artifact(_patcher.inputs, "train_inputs", "pickle")
        self.record_artifact(_patcher.labels, "train_labels", "pickle")

        patchable = [i for i, layer in enumerate(self.network.layers)
                     if isinstance(layer, FullyConnectedLayer)]
        for layer in patchable:
            patcher = FTRepair(self.network, _patcher.inputs.copy(), _patcher.labels.copy())
            patcher.layer = layer
            patcher.norm_objective = True
            patcher.auto_stop = False
            patcher.make_holdout_set()
            patcher.epochs = 1000
            patcher.lr = 0.001
            patcher.momentum = 0.9
            patcher.batch_size = 16
            patched = patcher.compute()
            assert patched is not None
            self.record_artifact(patched, f"{layer}_patched", "network")
            timing = patcher.timing.copy()
            timing["syrenn"] = syrenn_time
            self.record_artifact(timing, f"{layer}_timing", "pickle")

    def analyze(self):
        unpatched = self.read_artifact("unpatched")
        self.network = unpatched

        train_syrenn = self.read_artifact("train_syrenn")
        test_syrenn = self.read_artifact("test_syrenn")
        train_inputs = self.read_artifact("train_inputs")
        train_labels = self.read_artifact("train_labels")

        patchable = [i for i, layer in enumerate(self.network.layers)
                     if isinstance(layer, FullyConnectedLayer)]
        for layer in patchable:
            print("~~~~~ Layer:", layer, "~~~~~")
            patched = self.read_artifact(f"{layer}_patched")
            timing = self.read_artifact(f"{layer}_timing")

            total_points = sum(len(pre_poly) for upolytope in train_syrenn
                               for pre_poly in upolytope)
            print("Size of repair set:", total_points)

            og_train_outputs = np.argmax(unpatched.compute(train_inputs), axis=1)
            print("Number of repair set points originally buggy:",
                  np.sum(og_train_outputs != train_labels))

            gen_set, drawdown_set = self.find_counterexamples(unpatched, test_syrenn)
            print("Size of generalization, drawdown sets:", len(gen_set), len(drawdown_set))

            print("Time (seconds):", timing["total"])

            train_outputs = np.argmax(patched.compute(train_inputs), axis=1)
            print("Number of repair set points buggy after repair:",
                  np.sum(train_outputs != train_labels))

            dd_desired = self.property(drawdown_set)
            dd_outputs = np.argmax(patched.compute(drawdown_set), axis=1)
            print("Drawdown-set counterexamples after repair:", np.sum(dd_desired != dd_outputs))

            gen_desired = self.property(gen_set)
            gen_outputs = np.argmax(patched.compute(gen_set), axis=1)
            print("Generalization-set counterexamples after repair:", np.sum(gen_desired != gen_outputs))

        return True

if __name__ == "__main__":
    ACASMFT("acas_mft").main()
