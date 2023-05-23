"""

Script to generate an __init__.py-file for the C++ module

"""

import pathlib
import sys


def main(module_name: str):
    path = pathlib.Path(__file__).parent.parent
    with open(path / "__init__.py", "w") as file:
        file.write(
            f"""try:
    from .{module_name} import *
except ImportError:
    print("Warning: {module_name} C++ module not found")
"""
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception("No module name supplied. Please enter a module name.")
    main(sys.argv[1])
