try:
    from .eigen_wrapper import *
except ImportError:
    print("Warning: eigen_wrapper C++ module not found")
