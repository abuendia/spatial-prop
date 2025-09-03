from setuptools import find_packages, setup


VERSION = "0.1.0"

setup(
    name="spatial-gnn",
    version=VERSION,
    description="GNN models for spatial transcriptomics",
    namespace_packages=[],
    package_dir={"": "src"},
    packages=find_packages(where="src", exclude=["test*"]),
    include_package_data=True,
    python_requires=">=3.10,<4",
)