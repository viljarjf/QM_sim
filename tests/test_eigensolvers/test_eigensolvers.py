from qm_sim.cpp import eigen as s
from matplotlib import pyplot as plt
import numpy as np
from qm_sim.hamiltonian import Hamiltonian
from qm_sim.nature_constants import m_e, e_0
import time
import functools

from .scipy_backend import TestHam, eigs, eigsh

# Decorator to print the time a function took
# Optionally give an amount of calls to the function
def timer(n: int = 1):
    def decorator(func):
        @functools.wraps(func)
        def _timer(*args, **kwargs):
            start = time.perf_counter()
            for _ in range(n):
                out = func(*args, **kwargs)
            print(f"{func.__name__}\n\ttook {time.perf_counter() - start :#.5g}s")
            return out
        return _timer
    return decorator

def test_eigensolvers():

    N = [1000,]
    k = 4
    L = [10.0,]
    stencil = [-2.0, 1.0]
    n_iter = 10

    V = [0.5 for _ in range(N[0])]
    for i in range(N[0]//4, 3*N[0]//4):
        V[i] = 0.0
    
    H = s.Hamiltonian(N, L, stencil, 1.0)
    H.set_potential(V)

    H2 = TestHam(N, L, stencil, 1.0)
    H2.set_potential(V)
    H2_matop = H2.matop()

    # NOTE: the Hamiltonian class does NOT use reduced units (yet)
    H3 = Hamiltonian(N, [1e-9*i for i in L], m_e)
    H3.V = np.array(V) * e_0

    H4 = Hamiltonian(N, [1e-9*i for i in L], m_e, eigensolver="pytorch")
    H4.V = np.array(V) * e_0

    @timer(n_iter)
    def spectra():
        return s.eigen(N, L, stencil, k, 1.0)

    @timer(n_iter)
    def scipy_cpp_op():
        return eigsh(H.as_operator, N[0], a1.dtype, which="SA", k=k)

    @timer(n_iter)
    def scipy_numba_op():
        return eigsh(H2_matop, N[0], a1.dtype, which="SA", k=k)

    @timer(n_iter)
    def scipy_numba_op_nonsymmetric():
        return eigs(H2_matop, N[0], a1.dtype, which="SR", k=k)
    
    @timer(n_iter)
    def scipy_matrix_op():
        return eigsh(H3.mat._mul_vector, N[0], a1.dtype, which="SA", k=k)

    @timer(n_iter)
    def scipy_matrix_op_nonsymmetric():
        return eigs(H3.mat._mul_vector, N[0], a1.dtype, which="SR", k=k)

    @timer(n_iter)
    def qm_sim():
        return H3.eigen(k)
    
    @timer(n_iter)
    def qm_sim_gpu():
        return H4.eigen(k)

    a1 = spectra()[:, ::-1]
    w, a2 = scipy_cpp_op()
    w, a3 = scipy_numba_op()
    _, a5 = scipy_matrix_op()
    scipy_matrix_op_nonsymmetric()
    scipy_numba_op_nonsymmetric()
    _, a4 = qm_sim()
    _, a6 = qm_sim_gpu()


    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(a1[:, 1])
    plt.plot(a2[:, 1])
    plt.plot(a3[:, 1])
    plt.plot(a5[:, 1])
    # plt.plot(a4[1, :]) # normalised => not on the same scale as the rest
    plt.legend(["spectra", "scipy cpp", "scipy numba", "scipy python", "qm_sim",])
    plt.subplot(1, 2, 2)
    plt.plot(a1[:, 3])
    plt.plot(a2[:, 3])
    plt.plot(a3[:, 3])
    plt.plot(a5[:, 3])
    # plt.plot(a4[3, :]) # normalised => not on the same scale as the rest
    plt.legend(["spectra", "scipy cpp", "scipy numba", "scipy python", "qm_sim",])
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(a2)
    plt.legend(["cpp-scipy"])
    plt.subplot(1, 2, 2)
    plt.plot(a3)
    plt.legend(["numba-scipy"])
    plt.show()

    plt.figure()
    plt.plot(w[2]*a3[:, 2])
    plt.plot(H2.matop()(a3[:, 2]))
    plt.title("Totally an eigenvector guys cmon")
    plt.legend(["$\lambda v$", "$Av$"])
    plt.show()

def test_implemented_eigensolvers():

    from qm_sim.hamiltonian import Hamiltonian
    N = (200, 200)
    k = 4
    L = (10e-9, 10e-9)
    n_iter = 1

    V = e_0 * np.ones(N)

    H1 = Hamiltonian(N, L, m_e, eigensolver="scipy")
    H1.V = V

    H2 = Hamiltonian(N, L, m_e, eigensolver="pytorch")
    H2.V = V

    @timer(n_iter)
    def scipy_backend():
        return H1.eigen(k)
    
    @timer(n_iter)
    def pytorch_backend():
        return H2.eigen(k)
    
    scipy_backend()
    pytorch_backend()
