py_library(
    name = "prdnn",
    srcs = ["__init__.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        ":ddnn",
        ":ft_repair",
        ":provable_repair",
    ],
)

py_library(
    name = "provable_repair",
    srcs = ["provable_repair.py"],
    visibility = ["//visibility:public"],
    deps = [":ddnn"],
)

py_library(
    name = "ft_repair",
    srcs = ["ft_repair.py"],
    visibility = ["//visibility:public"],
    deps = [":provable_repair"],
)

py_library(
    name = "ddnn",
    srcs = ["ddnn.py"],
    visibility = ["//visibility:public"],
)
