try:
    from .eigen_wrapper import *
except ImportError:
    print("Warning: Eigen C++ module not found")
