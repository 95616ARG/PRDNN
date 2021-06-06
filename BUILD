load("@bazel_python//:bazel_python.bzl", "bazel_python_coverage_report", "bazel_python_interpreter")

bazel_python_interpreter(
    name = "bazel_python_venv",
    python_version = "3.7.4",
    requirements_file = "requirements.txt",
    run_after_pip = """
        pip3 install -i https://pypi.gurobi.com gurobipy || exit 1
    """,
    visibility = ["//:__subpackages__"],
)

bazel_python_coverage_report(
    name = "coverage_report",
    code_paths = ["prdnn/*.py"],
    test_paths = ["prdnn/tests/*"],
)

# For wheel-ifying the Python code.
# Thanks!
# https://hynek.me/articles/sharing-your-labor-of-love-pypi-quick-and-dirty/
genrule(
    name = "wheel",
    srcs = [
        "prdnn",
        "requirements.txt",
        "LICENSE",
        "pip_info/__metadata__.py",
        "pip_info/README.md",
        "pip_info/setup.cfg",
        "pip_info/setup.py",
    ],
    outs = ["prdnn.dist"],
    cmd = """
    PYTHON_VENV=$(location //:bazel_python_venv)
    pushd $$PYTHON_VENV/..
    source bazel_python_venv_installed/bin/activate
    popd
    cp pip_info/* .
    python3 setup.py sdist bdist_wheel
    cp -r dist $@
    """,
    tools = [
        "//:bazel_python_venv",
    ],
)
