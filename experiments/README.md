# Experiments
These are Python scripts to reproduce experiments from our paper. Experiments
can be run by calling `bazel run experiments:{experiment_name}`. Results are
placed in a ``results`` folder in this directory.

Note that in order to run `experiments:squeezenet_*`, you must have both the
ImageNet-A and ImageNet-Validation datasets extracted on the local machine and
provide their paths when requested. See [../README](../README) for more
information.
