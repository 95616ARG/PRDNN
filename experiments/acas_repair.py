"""Experiment for repairing the ACAS Xu network with SyReNN."""
from timeit import default_timer as timer
import numpy as np
from prdnn import DDNN, ProvableRepair
from experiments.experiment import Experiment

class ACASRepair(Experiment):
    """Experiment testing Provable Repair on an ACAS Xu model."""
    def run(self):
        """Repair the ACAS Xu network and record results."""
        np.random.seed(24)
        self.network = self.load_network("acas_2_9")
        input_helpers = self.load_input_data("acas")
        process = input_helpers["process"]

        regions = self.find_regions(
            # These are sampled:
            (-0.1, 0.1), (600, 1200), (600, 1200),
            # These are SyReNN'd:
            (0, 60760), (-np.pi, -0.75 * np.pi),
            process, n_samples=20)

        network = self.network
        train_regions = regions[:10]
        test_regions = regions[10:]
        assert len(test_regions) == 12

        self.record_artifact(network, "unpatched", "network")

        # Now that we have the regions, compute the SyReNN.
        syrenn_start = timer()
        train_syrenn = network.transform_planes(
            train_regions, compute_preimages=True, include_post=False)
        syrenn_time = timer() - syrenn_start
        # We don't time this because it's just for evaluation.
        test_syrenn = network.transform_planes(
            test_regions, compute_preimages=True, include_post=False)

        self.record_artifact(train_syrenn, "train_syrenn", "pickle")
        self.record_artifact(test_syrenn, "test_syrenn", "pickle")

        self.do_repair(train_regions, train_syrenn, syrenn_time)

    def do_repair(self, train_regions, train_syrenn, syrenn_time):
        # Then we start the repair.
        patcher = ProvableRepair.from_spec_function(
            self.network, 12, train_regions, self.property,
            use_representatives=True)
        patcher.constraint_type = "linf"
        patcher.soft_constraint_slack_lb = -2.0
        patcher.soft_constraint_slack_ub = 0.
        patcher.soft_constraint_weight = 100.
        patcher.constraint_buffer = 0.
        patcher.gurobi_crossover = 0
        # Batching any more than this doesn't really seem to help.
        patcher.batch_size = 2048
        patched = patcher.compute()
        assert patched is not None
        self.record_artifact(patched, f"patched", "ddnn")
        timing = patcher.timing.copy()
        timing["syrenn"] = syrenn_time
        timing["total"] += syrenn_time
        self.record_artifact(timing, f"timing", "pickle")

    def analyze(self):
        """Analyze the results."""
        unpatched = self.read_artifact("unpatched")
        self.network = unpatched
        patched = self.read_artifact("patched")
        timing = self.read_artifact("timing")
        train_syrenn = self.read_artifact("train_syrenn")
        test_syrenn = self.read_artifact("test_syrenn")

        total_points = sum(len(pre_poly) for upolytope in train_syrenn
                           for pre_poly in upolytope)
        print("Size of repair set:", total_points)

        gen_set, drawdown_set = self.find_counterexamples(unpatched, test_syrenn)
        print("Size of generalization, drawdown sets:", len(gen_set), len(drawdown_set))

        print("Time (seconds):", timing)

        print("Polytope accuracy in the repair set after repair (should be 1.0 = 100%):",
              self.spec_accuracy(patched, train_syrenn)[2])

        dd_desired = self.property(drawdown_set)
        dd_outputs = np.argmax(patched.compute(drawdown_set), axis=1)
        print("Drawdown-set counterexamples after repair:", np.sum(dd_desired != dd_outputs))

        gen_desired = self.property(gen_set)
        gen_outputs = np.argmax(patched.compute(gen_set), axis=1)
        print("Generalization-set counterexamples after repair:", np.sum(gen_desired != gen_outputs))

        return True

    def find_regions(self, *args, **kwargs):
        """Returns 20 input slices which have counterexamples.

        Overall, this is accomplished by randomly sampling regions, applying
        SyReNN, and seeing if they have a counterexample.

        To speed this process up, I have manually run it for enough iterations
        to see which of the first batch of randomly-sampled regions have
        counterexamples. Then I specified their indices in @with_cexs, which
        allows us to skip computing SyReNN for regions without counterexamples
        (there are lots of them!). To verify this, you may set with_cexs=[] to
        force the system to check for counterexamples on all randomly-sampled
        regions explicitly.
        """
        np.random.seed(24)
        n_samples = kwargs["n_samples"]
        with_cexs = [[3, 5, 7, 38, 55],
                     [5, 16, 23, 48, 70, 99],
                     [1, 10, 16, 38, 59, 74, 99],
                     [64, 71, 89, 92]]
        found = []
        iters = -1
        while len(found) < n_samples:
            iters += 1
            kwargs["n_samples"] = 100
            regions = self.compute_regions(*args, **kwargs)
            if iters < len(with_cexs):
                regions = [regions[i] for i in with_cexs[iters]]
            syrenn = self.network.transform_planes(
                regions, compute_preimages=True, include_post=False)
            for i, upolytope in enumerate(syrenn):
                all_points = np.concatenate(upolytope)
                outputs = np.argmax(self.network.compute(all_points), axis=1)
                if np.any(outputs >= 2):
                    found.append(regions[i])
        return found

    def find_counterexamples(self, network, syrenn):
        """Given SyReNN for a region, returns cexs and non-cexs."""
        np.random.seed(24)
        counterexamples = []
        drawdown_pts = []
        for upolytope in syrenn:
            all_polys = np.concatenate(upolytope)
            min_ = np.min(all_polys, axis=0)
            max_ = np.max(all_polys, axis=0)

            all_polys = []
            for pre_poly in upolytope:
                center = np.mean(pre_poly, axis=0)
                for alpha in [0.25, 0.5, 0.75]:
                    poly = pre_poly + (alpha * (pre_poly - center))
                    all_polys.extend(poly)
            desired = self.property(all_polys)
            outputs = np.argmax(network.compute(all_polys), axis=1)
            cex_points = np.asarray(all_polys)[outputs != desired]
            counterexamples.extend(cex_points)

            sample_points = np.random.uniform(min_, max_, size=(5*len(cex_points), 5))
            desired = self.property(sample_points)
            outputs = np.argmax(network.compute(sample_points), axis=1)
            sample_points = sample_points[desired == outputs][:len(cex_points)]
            drawdown_pts.extend(sample_points)
        return counterexamples, drawdown_pts

    def spec_accuracy(self, network, syrenn):
        """Computes statistics about how well the given network meets the spec.

        Note that these are based on SyReNN, by checking the vertices of the
        linear regions. If it states that there are no bad points, then the
        entire region is guaranteed to meet the spec.
        """
        n_good, n_bad, area_bad = 0, 0, 0.
        polytopes = (pre_poly for upolytope in syrenn for pre_poly in upolytope)
        for pre_poly in polytopes:
            desired = self.property(pre_poly)
            if isinstance(network, DDNN):
                representative = np.mean(pre_poly, axis=0)
                representatives = np.array([representative for _ in pre_poly])
                patched_output = network.compute(
                    pre_poly, representatives=representatives)
            else:
                patched_output = network.compute(pre_poly)
            patched_output = np.argmax(patched_output, axis=1)
            is_correct = np.all(patched_output == desired)
            if is_correct:
                n_good += 1
            else:
                n_bad += 1
        return (n_good, n_bad, n_good / (n_good + n_bad), area_bad)

    def property(self, inputs):
        """Property 8 from the Reluplex paper (Katz et al.).

        Note this is a slightly strengthened version of that property, in order
        to make it convex/conjunctive.
        """
        output = self.network.compute(inputs)
        # labels is 0 when it should be COC, 1 when it should be WL.
        label_wl = (output[:, 1] > output[:, 0]).astype(int)
        return label_wl.astype(int)

    def compute_regions(self,
                        intruder_heading, own_velocity, intruder_velocity,
                        rho, phi, process, n_samples):
        """Samples @n_samples 2D slices from the space."""
        regions = []
        for _ in range(n_samples):
            self.intruder_heading = np.random.uniform(*intruder_heading)
            self.own_velocity = np.random.uniform(*own_velocity)
            self.intruder_velocity = np.random.uniform(*intruder_velocity)
            regions.append(process(np.array([
                self.build_input(rho[0], phi[0]),
                self.build_input(rho[1], phi[0]),
                self.build_input(rho[1], phi[1]),
                self.build_input(rho[0], phi[1]),
            ])))
        return regions

    def build_input(self, distance, psi):
        """Returns an (un-processed) input point corresponding to the scenario.
        """
        return np.array([distance, psi, self.intruder_heading,
                         self.own_velocity, self.intruder_velocity])

if __name__ == "__main__":
    ACASRepair("acas_repair").main()
