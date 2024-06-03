from setuptools import setup, find_packages

setup(
    name="TEST",
    version="1.0",
    description="End-to-end fluorescence to spike inference procedure",
    author="Gerard Joey Broussard",
    author_email="",
    packages=find_packages(),
    python_requires="==3.8",
    install_requires=[
        "numpy==1.19.5",
        "scipy",
        "matplotlib==3.6.0",
        "tensorflow==2.4.0",  # pip install CPU and GPU tensorflow
        "h5py",,
        "ruamel.yaml",
    ],
)