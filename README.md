# PRDNN
PRDNN (pronounced "pardon") is a library for provable repair of Deep Neural
Networks. DNN behavior involving either finitely-many or entire polytopes of
points can be repaired using PRDNN.

The code in this repository is the latest artifact from our paper
***Provable Repair of Deep Neural Networks***, to appear in PLDI 2021 and
currently available [on arXiv](https://arxiv.org/abs/2104.04413).
```
@inproceedings{PLDI2021,
  author = {Sotoudeh, Matthew and Thakur, Aditya V.},
  title = {Provable Repair of Deep Neural Networks},
  booktitle = {42nd {ACM} {SIGPLAN} International Conference on Programming Language Design and Implementation ({PLDI})},
  publisher = {ACM},
  year = {2021},
  note = {To appear}
}
```

# Quickstart
## Prerequisites
#### Using as a package
If you only wish to use PRDNN as a package in your own code, and not run any of
the experiments in `experiments`, then the instructions are:
1. Install [Gurobi](https://www.gurobi.com) (we tested 9.0.1 and 9.0.2)
2. In your project, install the `prdnn` package from PyPI: `python3 -m pip
   install prdnn`.
3. _If you want to do polytope patching_, then in another session clone
   [SyReNN](https://github.com/95616ARG/SyReNN) and run `make start_server`.
   This step is only necessary for polytope patching, not pointwise patching.

#### Reproducing our experiments
On the other hand, if you wish to reproduce the experiments in our paper, you
will need the following prerequisites:
1. Follow the instructions in
   [bazel_python](https://github.com/95616ARG/bazel_python/) to build a
   reproducible version of Python 3.7.4, which will be used by this project. It
   must be built with the relevant OpenSSL libraries installed.
2. Install [Bazel](https://bazel.build) (we have tested on a variety of
   versions, including 4.0.0).
3. Install [Gurobi](https://www.gurobi.com) (we tested 9.0.1 and 9.0.2)
5. If you want to patch the ImageNet model, see `ImageNet` below.
6. In another session, clone [SyReNN](https://github.com/95616ARG/SyReNN) and
   run `make start_server`.
7. Run your desired tests or experiments (see below).

The only supported way to run our experiments is through Bazel and
Bazel-Python, as described above. This ensures a reasonably reproducable
environment.

However, PRDNN is written entirely in Python and it should be possible in many
cases to run the experiments directly.  However, in this scenario you will have
to manage dependencies and downloading data on your own.

#### Hardware Requirements
Most of the experiments can be run on consumer-grade laptop hardware with no
problems. When prompted for the number of rows to produce, note that the 4-row
experiments generally require significantly more memory.

The paper experiments were run using 32 threads and maximum 300 GB of memory.

## Running Tests
To run the library unit tests, use:
```bash
bazel test //prdnn/...
```
NOTE: Bazel does not pass the `$HOME` environment variable into tests. This
means that if your Gurobi license file is stored in `$HOME/gurobi.lic` it will
not be picked up by default, causing a failed test. To resolve this, you should
explicitly set `GRB_LICENSE_FILE` before running the tests, e.g.:
```bash
GRB_LICENSE_FILE=$HOME/gurobi.lic bazel test //...
```
To get the coverage report after running the tests, use
```
bazel run coverage_report
```

## Running Experiments
To run an experiment, use:
```bash
bazel run experiments:{experiment_name}
```
Where `{experiment_name}` is one of:
* `squeezenet_repair`
* `mnist_repair`
* `acas_repair`

The baselines are experiments:
* `squeezenet_ft`, `squeezenet_mft`
* `mnist_ft`, `mnist_mft`
* `acas_ft`, `acas_mft`

Results from the experiment will be printed, with detailed results placed in
`experiments/results/{experiment_name}.exp.tgz`.

## Known Issues
There are currently known issues on macOS. We have tested it successfully on
Ubuntu 16.04, 18.04, and 20.04.

## ImageNet
Currently, Bazel does not support archives with spaces in path names. This
prevents us from using Bazel to manage downloading/unarchiving of the
ImageNet-A and ImageNet datasets.

The below instructions are only necessary to run
`experiments:squeezenet_*`.

For the ImageNet-A dataset, it can be downloaded as below:
```
URL: https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar
SHA256: 3bb3632277e6ba6392ea64c02ddbf4dd2266c9caffd6bc09c9656d28f012589e
```
You should extract it to some place on disk and provide the path when requested
by the ImageNet patching script.

In order to evaluate the patched network, you will also need to download and
extract the original ImageNet validation set somewhere.  AcademicTorrents
should have it. We only need the validation set itself, not the surrounding
devkit/etc.
