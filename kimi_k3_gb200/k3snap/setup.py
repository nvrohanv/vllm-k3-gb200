from setuptools import setup

setup(
    name="k3snap",
    version="0.1",
    py_modules=["k3snap"],
    entry_points={"vllm.general_plugins": ["k3snap = k3snap:register"]},
)
