# QM_sim

Python library for simulation of quantum mechanical systems.

## Features 
- 1D and 2D systems
- Choice of finite difference scheme
- Stationary solutions

## Planned features
- Boundary conditions
- 3D systems
- Temporal simulation
- Time-variant potentials
- Transfer matrix for transmission ect.
- Testing

## Installation

`pip install qm-sim`

## Usage

Examples are provided in the `examples/`-folder

## Contribution

To contribute, please open a pull request to the `dev`-branch on [GitHub](https://www.github.com/viljarjf/QM_sim/pulls).

The following is an example of how to set up VS Code for development, adapt to your IDE of choice.

### Requirements
- VS Code
    - Python extension
- Python 3.10 or above

### Setup
1. Clone and open the repo in VS Code
2. Press f1, and run `Python: Create Environment`. Select `.venv`
3. In the terminal, run `.venv\Scripts\activate` on Windows, or `source .venv/bin/activate` on Unix
4. In the same terminal, run `pip install -e .` to install the current directory in an editable state
5. To run tests, run `pip install pytest` in the terminal, press f1 and run `Python: Configure Tests`. Choose `pytest`
