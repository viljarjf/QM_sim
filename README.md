# QM_sim

Python library for simulation of quantum mechanical systems. The documentation is available on [GitHub pages](https://viljarjf.github.io/QM_sim/).

[![Build docs](https://github.com/viljarjf/QM_sim/actions/workflows/build_docs.yml/badge.svg)](https://github.com/viljarjf/QM_sim/actions/workflows/build_docs.yml)

[//]: # (This is a comment. This comment should be on line 7. If this changes, also change the hard-coded line number for the start-line for the mdinclude at the top of docs/source/index.rst )

## Features 
- 1D, 2D, and 3D systems
- Choice of finite difference scheme
- Zero and periodic boundary conditions
- Stationary and temporal solutions
- Plots

## Planned features
- Transfer matrix for transmission ect.
- Proper testing

[//]: # (This is a comment. This comment should be on line 20. If this changes, also change the hard-coded line number for the end-line for the mdinclude at the top of docs/source/index.rst )

## Installation

`pip install qm-sim`

To be able to use the [PyTorch](https://pytorch.org/) backend for eigenvalue calculations, run the following command: 

`pip install qm-sim[torch]`

This will install the cpu-version of the package. To run GPU calculations, install the version for your system at the [PyTorch website](https://pytorch.org/get-started/locally/) instead.

## Usage

Examples are provided in the `examples/`-folder.
These are enumerated with increasing level of simulation complexity.

## Contribution

To contribute, please open a pull request to the `dev`-branch on [GitHub](https://www.github.com/viljarjf/QM_sim/pulls).

### Linting

When opening a PR, a linting check is performed.
To ensure your contribution passes the checks, you can run the following

~~~bash
$ pip install .[linting]
$ black src/qm_sim
$ isort src -m 3 --trailing-comma
$ pylint src --fail-under=7
~~~

### Setup

The following is an example of how to set up VS Code for development, adapt to your IDE of choice.

TL;DR: 
- `pip install -e .` to install in an editable state

**Requirements**
- VS Code
    - Python extension
- Python 3.10 or above

**Steps**
1. Clone the repo recursively and open the repo in VS Code. If not cloned recursively, initialize the submodules with `git submodule update --init`
2. Press f1, and run `Python: Create Environment`. Select `.venv`
3. Open a new terminal, which should automatically use the virtual environment. If not, run `.venv\Scripts\activate` on Windows, or `source .venv/bin/activate` on Unix
4. In the same terminal, run `pip install -e .[test]` to install the current directory in an editable state, and the testing utility Pytest
5. To run tests, press f1 and run `Python: Configure Tests`. Choose `pytest`. Run tests in the testing menu
