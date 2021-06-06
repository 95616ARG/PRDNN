py_library(
    name = "experiment",
    srcs = ["experiment.py"],
)

py_binary(
    name = "mnist_repair",
    srcs = ["mnist_repair.py"],
    data = [
        "@mnist_c//:all",
        "@mnist_relu_3_100_model//file",
    ],
    deps = [
        ":experiment",
        "//prdnn",
    ],
)

py_binary(
    name = "mnist_ft",
    srcs = ["mnist_ft.py"],
    data = [
        "@mnist_c//:all",
        "@mnist_relu_3_100_model//file",
    ],
    deps = [
        ":mnist_repair",
        "//prdnn",
    ],
)

py_binary(
    name = "mnist_mft",
    srcs = ["mnist_mft.py"],
    data = [
        "@mnist_c//:all",
        "@mnist_relu_3_100_model//file",
    ],
    deps = [
        ":mnist_repair",
        "//prdnn",
    ],
)

py_binary(
    name = "acas_repair",
    srcs = ["acas_repair.py"],
    data = glob(["models/acas_2_9.eran"]),
    deps = [
        ":experiment",
        "//prdnn",
    ],
)

py_binary(
    name = "acas_ft",
    srcs = ["acas_ft.py"],
    data = glob(["models/acas_2_9.eran"]),
    deps = [
        ":acas_repair",
        "//prdnn",
    ],
)

py_binary(
    name = "acas_mft",
    srcs = ["acas_mft.py"],
    data = glob(["models/acas_2_9.eran"]),
    deps = [
        ":acas_repair",
        "//prdnn",
    ],
)

py_binary(
    name = "squeezenet_repair",
    srcs = ["squeezenet_repair.py"],
    data = ["@onnx_squeezenet//:all"],
    deps = [
        ":experiment",
        ":imagenet_helpers",
        "//prdnn",
    ],
)

py_binary(
    name = "squeezenet_ft",
    srcs = ["squeezenet_ft.py"],
    data = ["@onnx_squeezenet//:all"],
    deps = [
        ":squeezenet_repair",
        "//prdnn",
    ],
)

py_binary(
    name = "squeezenet_mft",
    srcs = ["squeezenet_mft.py"],
    data = ["@onnx_squeezenet//:all"],
    deps = [
        ":squeezenet_repair",
        "//prdnn",
    ],
)

py_binary(
    name = "imagenet_helpers",
    srcs = ["imagenet_helpers.py"],
)
