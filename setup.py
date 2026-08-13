from setuptools import find_packages, setup

setup(
    name="simpler_env",
    version="0.0.1",
    author="Xuanlin Li",
    # ``simpler_protocol`` deliberately has no simulator or model-framework
    # dependencies.  Policy servers can install it without importing SAPIEN.
    packages=find_packages(include=["simpler_env*", "simpler_protocol*", "policy_servers*"]),
    python_requires=">=3.10",
)
