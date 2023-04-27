
import numpy as np
from scipy import sparse as sp


def nabla(N: tuple[int], L: tuple[float], order: int = 2, dtype: type = np.float64) -> sp.dia_matrix:
    """
    Finite difference derivative in cartesian coordinates.
    Uses central stencil.

    Args:
        N (tuple): 
            Discretization count along each axis
        L (tuple): 
            System size along each axis
        order (int, optional): 
            Numerical order of the differentiation scheme.
            Options are:
                - 2
                - 4
                - 6
                - 8
            Defaults to 2.
        dtype (type, optional):
            datatype of matrix. 
            Defaults to float64

    Example:

        Approximate the derivative of sin(x).

        >>> from qm_sim.spatial_derivative.cartesian import nabla
        >>> import numpy as np
        >>> N = (1000,)
        >>> L = (2*np.pi,)
        >>> n = nabla( N, L )
        >>> x = np.linspace( 0, L[0], N[0] )
        >>> y = np.sin(x)
        
        Now, the following estimates the second derivative of sin(x)
        >>> n @ y

    """

    # Lookup for first half of stencil. 
    # A 0 is appended for the central element, 
    # and the rest is mirrored with a sign change later
    match order:
        case 2:
            stencil = [-1/2]
        case 4:
            stencil = [1/12, -2/3]
        case 6:
            stencil = [-1/60, 3/20, -3/4]
        case 8:
            stencil = [1/280, -4/105, 1/5, -4/5]
        case _:
            raise NotImplementedError(f"Finite difference scheme not found for {order = }")
    stencil += [0]
    indices = _mirror_sign_list([-i for i in range(len(stencil))][::-1])
    stencil = _mirror_sign_list(stencil)
    mat = _matrix_from_stencil(stencil, indices, 1, N, L, dtype)


def laplacian(N: tuple[int], L: tuple[float], order: int = 2, dtype: type = np.float64) -> sp.dia_matrix:
    """
    Finite difference double derivative in cartesian coordinates.
    Uses central stencil

    Args:
        N (tuple): 
            Discretization count along each axis
        L (tuple): 
            System size along each axis
        order (int, optional): 
            Numerical order of the differentiation scheme.
            Options are:
                - 2
                - 4
                - 6
                - 8
            Defaults to 2.
        dtype (type, optional):
            datatype of matrix. 
            Defaults to float64

    Example:

        Approximate the second derivative of sin(x).

        >>> from qm_sim.spatial_derivative.cartesian import laplacian
        >>> import numpy as np
        >>> N = (1000,)
        >>> L = (2*np.pi,)
        >>> n = laplacian( N, L )
        >>> x = np.linspace( 0, L[0], N[0] )
        >>> y = np.sin(x)
        
        Now, the following estimates the second derivative of sin(x)
        >>> n @ y
        
    """
    # lookup the first half of the stencil, to be mirrored later
    match order:
        case 2:
            stencil = [1, -2]
        case 4:
            stencil = [-1/12, 4/3, -5/2]
        case 6:
            stencil = [1/90, -3/20, 3/2, -49/18]
        case 8:
            stencil = [-1/560, 8/315, -1/5, 8/5, -205/72]
        case _:
            raise NotImplementedError(f"Finite difference scheme not found for {order = }")
    
    indices = _mirror_sign_list([-i for i in range(len(stencil))][::-1])
    stencil = _mirror_list(stencil)
    return _matrix_from_stencil(stencil, indices, 2, N, L, dtype)
    

def _matrix_from_stencil(stencil: list[float], indices: list[int], 
    power: int, N: tuple[int], L: tuple[float], dtype: type) -> sp.dia_matrix:
    """
    Creates a full matrix from a stencil and its corresponding indices.
    """

    # iteration setup
    mat = 0
    prev_N = 1
    indices = np.array(indices)

    for L, N in zip(L, N):
        h = L / N
        # create regular finite difference matrix for each iteration dimension
        # but push the indices of the stencil out to account for the current dimension
        next_mat = 1/h**power * sp.diags(
            stencil,
            indices * prev_N,
            shape=(N * prev_N, N * prev_N),
            dtype=dtype,
            format="dia"
            )
        
        # Expand the previous iteration with the new one
        mat = next_mat + sp.kron(sp.eye(N), mat, format="dia")
        prev_N *= N

    return mat


def _mirror_list(l: list) -> list:
    """
    Mirror a list around the final element.

    Example: [a, b, c] -> [a, b, c, b, a]
    """
    return l + l[-2::-1]


def _mirror_sign_list(l: list) -> list:
    """
    Mirror a list around the final element,
    and flip the sign of each new element.

    Example: [-a, b, c] -> [-a, b, c, -b, a]
    """
    return l + [-i for i in l[-2::-1]]
