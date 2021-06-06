"""Methods for patching deep neural networks."""
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
if "GUROBI_HOME" in os.environ:
    PY_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"
    if PY_VERSION == "3.7":
        sys.path.append(os.environ["GUROBI_HOME"] + "/lib/python3.7")
    elif PY_VERSION == "3.8":
        sys.path.append(os.environ["GUROBI_HOME"] + "/lib/python3.8_utf32")
    else:
        # The user can figure it out.
        pass
from gurobipy import Model, GRB
from prdnn.ddnn import DDNN, LINEAR_LAYERS

class ProvableRepair:
    """Helper for patching a DDNN.

    An instance of a ProvableRepair represents patching a specific network on a
    specific pointwise patching specification with specific hyperparameters.
    Polytope patching specifications can be specified using
    ProvableRepair.from_planes or ProvableRepair.from_spec_function, which will
    construct an equivalent pointwise patching specification.
    """
    def __init__(self, network, layer_index, inputs, labels,
                 delta_bounds=(-3., 3.), representatives=None):
        """Initializes a new ProvableRepair.

        By default, we only allow weight deltas to take values between -3. and
        3. to prevent numerical issues. This can be relaxed in some scenarios,
        although I have found (-GRB.INFINITY, GRB.INFINITY) causing actual
        significant issues.

        For raw feasibility, take delta_norm_type="none",
        constraint_type="hard", slack_lb=0, slack_ub=INFTY.

        To do a strict relaxation of feasibility, take delta_norm_type="none",
        constraint_type="linf", slack_lb=0, slack_ub=INFTY. Note that if the
        model is feasible, these settings will find the satisfying model.
        However, the behavior is less predictable if the model is infeasible.
        In that scenario, it may have worst-case behavior.

        To do a better relaxation of feasibility, use constraint_type="l1" and
        play with slack_*b, *_weight.
        """
        self.network = network
        self.layer_index = layer_index
        self.inputs = np.asarray(inputs, dtype=np.float32)
        self.representatives = representatives
        if representatives is None:
            self.representatives = self.inputs
        self.labels = np.asarray(labels, dtype=np.int64)
        self.epsilon = None
        original_layer = network.layers[layer_index]
        # intermediates[i] is the DDNN after i patching steps, so
        # intermediates[0] is the original network.
        self.intermediates = [self.construct_patched(original_layer)]
        # times[i] is the time taken to run the ith iteration.
        self.times = [0.0]

        self.delta_bounds = delta_bounds
        # "hard", "l1", "linf"
        self.constraint_type = "hard"
        # Controls how much to prioritize improving already-met constraints vs.
        # meeting more constraints. Setting slack_lb = 0 means that, once a
        # constraint is met, 'meeting it more' won't improve the score.
        # Changing slack_ub is a bit weirder, basically it controls the maximum
        # 'badness' that an unmet constraint can contribute to the score.
        self.soft_constraint_slack_lb = -0.05
        self.soft_constraint_slack_ub = GRB.INFINITY
        self.soft_constraint_slack_type = GRB.CONTINUOUS
        # Objective weights
        self.delta_l1_weight = 1.
        self.delta_linf_weight = 1.
        self.soft_constraints_weight = 1.
        self.normalize_objective = True
        # We require Ax >= b + cb
        self.constraint_buffer = 0.05
        # Batch size
        self.batch_size = 128
        # Gurobi timeout (in seconds).
        self.gurobi_timelimit = None
        # Crossover parameter in Gurobi.
        self.gurobi_crossover = -1
        self.timing = None

    def compute(self):
        """Performs the Layer patching."""
        patch_start = timer()
        layer = self.network.layers[self.layer_index]
        if isinstance(layer, FullyConnectedLayer):
            weights = layer.weights.numpy().copy()
            biases = layer.biases.numpy().copy()
        elif isinstance(layer, Conv2DLayer):
            weights = layer.filter_weights.numpy().copy()
            biases = layer.biases.numpy().copy()
        else:
            raise NotImplementedError

        model = Model()
        if self.gurobi_timelimit is not None:
            model.Params.TimeLimit = self.gurobi_timelimit
        if self.gurobi_crossover != -1:
            model.Params.Crossover = self.gurobi_crossover
            model.Params.Method = 2

        # Adding variables...
        lb, ub = self.delta_bounds
        weight_deltas = model.addVars(weights.flatten().size, lb=lb, ub=ub).select()
        bias_deltas = model.addVars(biases.size, lb=lb, ub=ub).select()
        all_deltas = weight_deltas + bias_deltas

        if self.constraint_type == "hard":
            soft_constraint_bounds = []
        elif self.constraint_type == "linf":
            soft_constraint_bounds = [model.addVar(
                lb=self.soft_constraint_slack_lb,
                ub=self.soft_constraint_slack_ub,
                vtype=self.soft_constraint_slack_type)]
        elif self.constraint_type == "l1":
            out_dims = self.network.compute(self.inputs[:1]).shape[1]
            soft_constraint_bounds = model.addVars(
                len(self.inputs) * (out_dims - 1),
                lb=self.soft_constraint_slack_lb,
                ub=self.soft_constraint_slack_ub,
                vtype=self.soft_constraint_slack_type).select()
        else:
            raise NotImplementedError

        # Adding constraints...
        jacobian_compute_time = 0.
        for batch_start in tqdm(range(0, len(self.inputs), self.batch_size)):
            batch_slice = slice(batch_start, batch_start + self.batch_size)
            batch_labels = self.labels[batch_slice]

            jacobian_start = timer()
            A_batch, b_batch = self.network_jacobian(batch_slice)
            jacobian_compute_time += (timer() - jacobian_start)

            out_dims = A_batch.shape[1]
            assert out_dims == b_batch.shape[1]

            variables = None
            if self.constraint_type == "l1":
                variables = weight_deltas + bias_deltas + bounds
            weight_softs = 1.
            if self.soft_constraint_slack_type == GRB.BINARY:
                weight_softs = 10.

            full_As, full_bs = [], []
            bounds_slice = slice(batch_slice.start * (out_dims - 1),
                                 batch_slice.stop * (out_dims - 1))
            constraint_bounds_batch = soft_constraint_bounds[bounds_slice]
            for i, (A, b, label) in enumerate(zip(A_batch, b_batch, batch_labels)):
                other_labels = [l for l in range(out_dims) if l != label]
                # A[label]x + b[label] >= A[other]x + b[other]
                # (A[label] - A[other])x >= b[other] - b[label]
                As = np.expand_dims(A[label], 0) - A[other_labels]
                bs = (b[other_labels] - np.expand_dims(b[label], 0))
                bs += self.constraint_buffer

                if self.constraint_type == "linf":
                    As = np.concatenate((As, weight_softs * np.ones((As.shape[0], 1))), axis=1)
                elif self.constraint_type == "l1":
                    bounds = constraint_bounds_batch[
                        i*(out_dims-1):((i+1)*(out_dims-1))]
                    As = np.concatenate((As, weight_softs * np.eye(len(bounds))), axis=1)

                full_As.append(As)
                full_bs.extend(bs)
            model.addMConstr(np.concatenate(tuple(full_As), axis=0),
                             variables, '>', full_bs)

        # Specifying objective...
        objective = 0.
        if self.delta_l1_weight != 0.:
            # Penalize the L_1 norm. To do this, we must add variables which
            # represent the absolute value of each of the deltas.  The approach
            # used here is described at:
            # https://optimization.mccormick.northwestern.edu/index.php/Optimization_with_absolute_values
            n_vars = len(all_deltas)
            abs_ub = max(abs(lb), abs(ub))
            variable_abs = model.addVars(n_vars, lb=0., ub=abs_ub).select()
            n_vars += n_vars

            A = sparse.diags([1., -1.], [0, (n_vars // 2)],
                             shape=(n_vars // 2, n_vars),
                             dtype=np.float, format="lil")
            b = np.zeros(n_vars // 2)
            model.addMConstr(A, all_deltas + variable_abs, '<', b)

            A = sparse.diags([-1., -1.], [0, (n_vars // 2)],
                             shape=(n_vars // 2, n_vars),
                             dtype=np.float, format="lil")
            model.addMConstr(A, all_deltas + variable_abs, '<', b)

            # Then the objective we use is just the L_1 norm. TODO: maybe we
            # should wait until the end to weight this so the coefficients
            # aren't too small?
            if self.normalize_objective:
                weight = self.delta_l1_weight / len(variable_abs)
            else:
                weight = self.delta_l1_weight
            objective += (weight * sum(variable_abs))
        if self.delta_linf_weight != 0.:
            # Penalize the L_infty norm. We use a similar approach, except it
            # only takes one additional variable. For some reason it throws an
            # error if I use just addVar here.
            l_inf = model.addVar(lb=0., ub=max(abs(lb), abs(ub)))
            n_vars = len(weight_deltas) + len(bias_deltas) + 1

            A = sparse.eye(n_vars - 1, n_vars, dtype=np.float, format="lil")
            A[:, -1] = -1.
            b = np.zeros(n_vars - 1)
            model.addMConstr(A, all_deltas + [l_inf], '<', b)

            A = sparse.diags([-1.], shape=(n_vars - 1, n_vars),
                             dtype=np.float, format="lil")
            A[:, -1] = -1.
            model.addMConstr(A, all_deltas + [l_inf], '<', b)

            # L_inf objective.
            objective += (self.delta_linf_weight * l_inf)
        # Soft constraint weight.
        if self.normalize_objective:
            weight = self.soft_constraints_weight / max(len(soft_constraint_bounds), 1)
        else:
            weight = self.soft_constraints_weight
        objective += weight * sum(soft_constraint_bounds)
        model.setObjective(objective, GRB.MINIMIZE)

        # Solving...
        gurobi_start = timer()
        model.update()
        model.optimize()
        gurobi_solve_time = (timer() - gurobi_start)

        self.timing = dict({
            "jacobian": jacobian_compute_time,
            "solver": gurobi_solve_time,
        })
        self.timing["did_timeout"] = (model.status == GRB.TIME_LIMIT)
        if model.status != GRB.OPTIMAL:
            print("Not optimal!")
            print("Model status:", model.status)
            self.timing["total"] = (timer() - patch_start)
            return None

        # Extracting weights...
        weights += np.asarray([d.X for d in weight_deltas]).reshape(weights.shape)
        biases += np.asarray([d.X for d in bias_deltas]).reshape(biases.shape)

        # Returning a patched network!
        if isinstance(layer, FullyConnectedLayer):
            patched_layer = FullyConnectedLayer(weights.copy(), biases.copy())
        else:
            patched_layer = Conv2DLayer(layer.window_data, weights.copy(), biases.copy())

        patched = self.construct_patched(patched_layer)

        self.timing["total"] = (timer() - patch_start)
        return patched

    def network_jacobian(self, batch_slice):
        """Returns A, b st A x delta + b = network output for a given batch.

        Essentially, this method returns the Jacobian of the network output wrt
        the given layer's weights and biases. Unfortunately, Pytorch is *really
        bad* at computing Jacobians. It can compute gradients of a single
        scalar, but not Jacobians of a vector output (or a matrix, as we need).

        This method takes the following approach:
        1. If the layer in question is a fully-connected layer, we can manually
           compute the Jacobian using self.layer_jacobian(...).
        2. Otherwise, we assume the layer is a 2D convolutional layer and use
           Pytorch to compute the Jacobian, which takes m*n Jacobian queries
           where m is the number of points and n is the number of output
           classes.

        TODO(masotoud, good starter project): For case (2), Pytorch recently
        introduced a Jacobian method that does exactly that. We can probably
        simplify this function by using their implementation.

        TODO(masotoud, follow-up project): In fact, there exist a number of
        optimized Jacobian implementations for Pytorch (see, e.g., the
        "backpack" project). However, using them seems to require significant
        modifications to the way we call Pytorch. Long-term, it would be nice
        to move to one of those (or at least re-use their primitives).
        """
        inputs = self.inputs[batch_slice]
        representatives = self.representatives[batch_slice]

        layer = self.network.layers[self.layer_index]

        original_outputs = self.network.compute(inputs)
        n_inputs, out_dims = original_outputs.shape

        if isinstance(layer, FullyConnectedLayer):
            weight_scales, bias_scales = self.layer_jacobian(
                inputs, representatives)
            # weight_scales is (n_inputs, out_dims, in_dims, mid_dims)
            # bias_scales is (n_inputs, out_dims, mid_dims)
        else:
            # This *SHOULD* be possible in general, but I don't know how to
            # express it nicely in Numpy/Pytorch. Trying to run layer_jacobian
            # will create a huge matrix, which is not what we want. The
            # approach below is pretty slow (due primarily to limitations with
            # Jacobian computation in Pytorch) but should work as long as we're
            # not using representatives.
            filters = layer.filter_weights
            biases = layer.biases

            def set_grad(v, g):
                v.requires_grad_(g)
            def maybe_zero(v):
                if v is not None:
                    v.zero_()

            for v in [filters, biases]:
                set_grad(v, True)
            pytorch_x = torch.tensor(inputs, requires_grad=True, dtype=torch.float)
            if self.inputs is self.representatives:
                output = self.network.compute(pytorch_x)
            else:
                masknet = DDNN(self.network.layers, self.network.layers)
                output = masknet.compute(pytorch_x, representatives)
            assert len(output.shape) == 2
            n_inputs, out_dims = output.shape

            weight_scales = np.zeros((n_inputs, out_dims,) + filters.shape)
            bias_scales = np.zeros((n_inputs, out_dims,) + biases.shape)
            for i in range(output.shape[0]):
                for j in range(output.shape[1]):
                    maybe_zero(filters.grad)
                    maybe_zero(biases.grad)
                    output[i, j].backward(retain_graph=True)
                    weight_scales[i, j, :, :, :] = filters.grad.numpy()
                    bias_scales[i, j, :] = biases.grad.numpy()

            for v in [filters, biases]:
                set_grad(v, False)
            weight_scales = np.array(weight_scales)
            bias_scales = np.array(bias_scales)

        # (n_inputs, out_dims, [filter_shape]) -> (out, [filter_size])
        weight_scales = weight_scales.reshape((n_inputs, out_dims, -1))
        # (n_inputs, out_dims, [bias_shape])
        bias_scales = bias_scales.reshape((n_inputs, out_dims, -1))
        return np.concatenate((weight_scales, bias_scales), axis=2), original_outputs

    def layer_jacobian(self, points, representatives):
        """Computes the Jacobian of the FULLY CONNECTED layer parameters.

        Returns (n_points, n_outputs, [weight_shape])

        Basically, we assume WLOG that the layer is the first layer then we get
        something like for each point:
        B_nB_{n-1}...B_1Ax

        For each point we can collapse B_n...B_1 into a matrix B of shape
        (n_outputs, n_A_outputs)
        then we compute the einsum with x to get
        (n_outputs, n_A_outputs, n_A_inputs)
        which is what we want.
        """
        pre_network = Network(self.network.layers[:self.layer_index])
        points = pre_network.compute(points)
        n_points, A_in_dims = points.shape

        representatives = pre_network.compute(representatives)
        representatives = self.network.layers[self.layer_index].compute(representatives)
        n_points, A_out_dims = representatives.shape

        post_network = Network(self.network.layers[(self.layer_index+1):])
        # (n_points, A_out, A_out)
        jacobian = np.repeat([np.eye(A_out_dims)], n_points, axis=0)
        # (A_out, n_points, A_out)
        jacobian = jacobian.transpose((1, 0, 2))
        for layer in post_network.layers:
            if isinstance(layer, LINEAR_LAYERS):
                if isinstance(layer, ConcatLayer):
                    assert not any(isinstance(input_layer, ConcatLayer)
                                   for input_layer in layer.input_layers)
                    assert all(isinstance(input_layer, LINEAR_LAYERS)
                               for input_layer in layer.input_layers)
                representatives = layer.compute(representatives)
                jacobian = jacobian.reshape((A_out_dims * n_points, -1))
                jacobian = layer.compute(jacobian, jacobian=True)
                jacobian = jacobian.reshape((A_out_dims, n_points, -1))
            elif isinstance(layer, ReluLayer):
                # (n_points, running_dims)
                zero_indices = (representatives <= 0)
                representatives[zero_indices] = 0.
                # (A_out, n_points, running_dims)
                jacobian[:, zero_indices] = 0.
            elif isinstance(layer, HardTanhLayer):
                big_indices = (representatives >= 1.)
                small_indices = (representatives <= -1.)
                np.clip(representatives, -1.0, 1.0, out=representatives)
                jacobian[:, big_indices] = 0.
                jacobian[:, small_indices] = 0.
            else:
                raise NotImplementedError
        # (A_out, n_points, n_classes) -> (n_points, n_classes, A_out)
        B = jacobian.transpose((1, 2, 0))
        # (n_points, n_classes, n_A_in, n_A_out)
        C = np.einsum("nco,ni->ncio", B, points)
        return C, B

    def set_points(self, inputs, labels, representatives=None):
        """Change the pointwise patching specification."""
        self.inputs = np.asarray(inputs, dtype=np.float32)
        self.labels = labels
        self.representatives = representatives
        if representatives is None:
            self.representatives = self.inputs

    @classmethod
    def from_planes(cls, network, layer_index, planes, labels, use_representatives=False):
        """Constructs a ProvableRepair to repair 2D regions.

        @planes should be a list of input 2D planes (Numpy arrays of their
            vertices in counter-clockwise order).
        @labels a list of the corresponding desired labels (integers).

        Internally, SyReNN is used to lower the problem to that of finitely
        many points.

        NOTE: This function requires one to have a particularly precise
        representation of the desired network output; in most cases,
        ProvableRepair.from_spec_function is more useful (see below).
        """
        transformed = network.transform_planes(planes,
                                               compute_preimages=True,
                                               include_post=False)
        all_inputs = []
        all_labels = []
        all_representatives = [] if use_representatives else None
        for upolytope, label in zip(transformed, labels):
            # include_post=False so the upolytope is just a list of Numpy
            # arrays.
            if use_representatives:
                points = []
                for vertices in upolytope:
                    all_inputs.extend(vertices)
                    representative = np.mean(vertices, axis=0)
                    all_representatives.extend(representative
                                               for _ in vertices)
                    all_labels.extend(label for _ in vertices)
            else:
                points = []
                for vertices in upolytope:
                    points.extend(vertices)
                # Remove duplicate points.
                points = list(set(map(tuple, points)))
                all_inputs.extend(points)
                all_labels.extend(label for _ in points)
        if not use_representatives:
            all_inputs, indices = np.unique(all_inputs, return_index=True, axis=0)
            all_labels = np.asarray(all_labels)[indices]
        return cls(network, layer_index, all_inputs, all_labels,
                   representatives=all_representatives)

    @classmethod
    def from_spec_function(cls, network, layer_index, region_plane,
                           spec_function, use_representatives=False):
        """Constructs a ProvableRepair for an input region and "Spec Function."

        @region_plane should be a single plane (Numpy array of
            counter-clockwise vertices) that defines the "region of interest"
            to patch over.
        @spec_function should take a set of input points (Numpy array) and
            return the desired corresponding labels (list/Numpy array of ints).

        Here we use a slightly in-exact algorithm; we get all partition
        endpoints using SyReNN, then use those for the ProvableRepair.

        If the @spec_function classifies all points on a linear partition the
        same way, then this exactly encodes the corresponding problem for the
        ProvableRepair (i.e., if the ProvableRepair reports all constraints met
        then the repaired network exactly matches the @spec_function).

        If the @spec_function classifies some points on a linear partition
        differently than others, the encoding may not be exact (i.e.,
        ProvableRepair may report all constraints met even when some input has
        a different output in the patched network than the @spec_function).
        However, in practice, this works *very* well and is significantly more
        efficient than computing the exact encoding and resulting patches.
        """
        if len(np.asarray(region_plane).shape) == 2:
            region_plane = [region_plane]
        assert len(np.asarray(region_plane).shape) == 3
        upolytopes = network.transform_planes(region_plane,
                                              compute_preimages=True,
                                              include_post=False)
        inputs = []
        representatives = [] if use_representatives else None
        for upolytope in upolytopes:
            for polytope in upolytope:
                inputs.extend(list(polytope))
                if use_representatives:
                    representative = np.mean(polytope, axis=0)
                    representatives.extend(representative for _ in polytope)
        if not use_representatives:
            inputs = np.unique(np.asarray(inputs, dtype=np.float32), axis=0)
        labels = spec_function(inputs)
        return cls(network, layer_index, inputs, labels,
                   representatives=representatives)

    def construct_patched(self, patched_layer):
        """Constructs a DDNN given the patched layer."""
        activation_layers = self.network.layers
        value_layers = activation_layers.copy()
        value_layers[self.layer_index] = patched_layer
        return DDNN(activation_layers, value_layers)
