from setuptools import setup

setup(
    name="k3opt",
    version="0.1",
    py_modules=["k3opt", "tailattn_patch", "mla_patch", "l2pf_patch", "gemv_patch", "kda_fb_patch", "oproj_patch", "moeblock_patch", "plans_patch", "step_patch", "stepov_patch", "attnfront_patch", "upentry_patch", "upentry_skinny_patch"],
    entry_points={"vllm.general_plugins": ["k3opt = k3opt:register"]},
)
