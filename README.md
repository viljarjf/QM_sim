# QM_sim

Python library for simulation of quantum mechanical systems.

## Features 
- 1D and 2D systems
- Choice of finite difference scheme
- Stationary and temporal solutions

## Planned features
- Boundary conditions
- 3D systems
- Transfer matrix for transmission ect.
- Testing

## Installation

`pip install qm-sim`

## Usage

Examples are provided in the `examples/`-folder.
These are enumerated with increasing level of simulation complexity.

## Contribution

To contribute, please open a pull request to the `dev`-branch on [GitHub](https://www.github.com/viljarjf/QM_sim/pulls).

The following is an example of how to set up VS Code for development, adapt to your IDE of choice.

TL;DR: 
- `pip install -e .` to install in an editable state
- `pip install .` to (re)compile the C++ (subsequent python file edits will not be recognized before another reinstall)

### Requirements
- VS Code
    - Python extension
- Python 3.10 or above

### Setup
1. Clone the repo recursively and open the repo in VS Code. If not cloned recursively, initialize the submodules with `git submodule update --init`
2. Press f1, and run `Python: Create Environment`. Select `.venv`
3. Open a new terminal, which should automatically use the virtual environment. If not, run `.venv\Scripts\activate` on Windows, or `source .venv/bin/activate` on Unix
4. In the same terminal, run `pip install -e .[test]` to install the current directory in an editable state, and the testing utility Pytest
5. To run tests, press f1 and run `Python: Configure Tests`. Choose `pytest`. Run tests in the testing menu
