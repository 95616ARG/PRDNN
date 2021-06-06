workspace(name = "dnn_patching")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
    name = "bazel_python",
    commit = "f99ab8738dced7257c97dc719457f50a601ed84c",
    remote = "https://github.com/95616ARG/bazel_python.git",
)

load("@bazel_python//:bazel_python.bzl", "bazel_python")

bazel_python()

# See the README in: https://github.com/bazelbuild/rules_foreign_cc
all_content = """filegroup(name = "all", srcs = glob(["**"]), visibility = ["//visibility:public"])"""

# MODELS https://github.com/eth-sri/eran
http_file(
    name = "mnist_relu_3_100_model",
    downloaded_file_path = "model.eran",
    sha256 = "e4151dfced1783360ab8353c8fdedbfd76f712c2c56e4b14799b2f989217229f",
    urls = ["https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_relu_3_100.tf"],
)

http_archive(
    name = "onnx_squeezenet",
    build_file_content = all_content,
    sha256 = "aff6280d73c0b826f088f7289e4495f01f6e84ce75507279e1b2a01590427723",
    strip_prefix = "squeezenet1.1",
    urls = ["https://s3.amazonaws.com/onnx-model-zoo/squeezenet/squeezenet1.1/squeezenet1.1.tar.gz"],
)

# DATASETS
http_archive(
    name = "mnist_c",
    build_file_content = all_content,
    sha256 = "af9ee8c6a815870c7fdde5af84c7bf8db0bcfa1f41056db83871037fba70e493",
    strip_prefix = "mnist_c",
    urls = ["https://zenodo.org/record/3239543/files/mnist_c.zip"],
)
