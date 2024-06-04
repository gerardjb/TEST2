# Setup on Della

These are della specific instructions for setting up the project. We need to retrieve
some of the dependencies from vcpkg and we need to setup a python environment with
tensorflow with GPU support. Finally, we need to install the project in editable mode
with automatic rebuilds enabled in sckit-build-core.

## Setup vcpkg
```bash
cd /scratch/gpfs/<username>/TEST2
git clone https://github.com/microsoft/vcpkg
./vcpkg/bootstrap-vcpkg.sh
./vcpkg/vcpkg integrate install gsl Armadillo jsoncpp boost-circular-buffer
./vcpkg/vcpkg integrate install
```

## Setup python environment

We use anaconda to get a python environment with tensorflow with GPU support on della. 
This is the recommened way from the research computing website. Alternatively, we could
use pip but we would need to load the appropriate modules. See: 
https://researchcomputing.princeton.edu/support/knowledge-base/tensorflow

```bash
module load anaconda3/2024.2
conda create --name spike_find tensorflow-gpu --channel conda-forge
conda activate spike_find
```

## Setup python build dependencies 

We only need to do this on della because we are doing development and we don't want to 
download the dependencies every time we build the code.

```bash
pip install scikit-build-core setuptools_scm pybind11
```

## Install the project

As per the instructions from scikit-build-core and [editable instals](https://scikit-build-core.readthedocs.io/en/latest/configuration.html#editable-installs), which are still a bit experimental, we need to use the following command to install the project in editable mode. This will enable automatic rebuilds when the source code changes and the package is imported.

```bash
pip install --no-build-isolation --config-settings=editable.rebuild=true -Cbuild-dir=build -ve.
```

