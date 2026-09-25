from setuptools import setup

setup(
    name="k3opt",
    version="0.1",
    py_modules=["k3opt", "tailattn_patch", "mla_patch", "l2pf_patch", "gemv_patch"],
    entry_points={"vllm.general_plugins": ["k3opt = k3opt:register"]},
)
