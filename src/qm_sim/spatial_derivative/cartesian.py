
import numpy as np
from scipy import sparse as sp


def nabla(N: tuple[int], L: tuple[float], order: int = 2, dtype: type = np.float64, 
          boundary_condition: str = "zero") -> sp.dia_matrix:
    """Finite difference derivative in cartesian coordinates.
    Uses central stencil.

    Example: approximate the derivative of sin(x).

    >>> from qm_sim.spatial_derivative.cartesian import nabla
    >>> import numpy as np
    >>> N = (1000,)
    >>> L = (2*np.pi,)
    >>> n = nabla( N, L, boundary_condition="periodic")
    >>> x = np.linspace( 0, L[0], N[0], endpoint=False )
    >>> y = np.sin(x)
    >>> np.allclose(n @ y, np.cos(x)) # The analytical solution is cos(x)
    True

    :param N: Discretization count along each axis
    :type N: tuple[int]
    :param L: System size along each axis
    :type L: tuple[float]
    :param order: Numerical order of the differentiation scheme.
        Options are:

            - 2
            - 4
            - 6
            - 8

        Defaults to 2.
    :type order: int, optional
    :param dtype: datatype of matrix. 
        Defaults to np.float64
    :type dtype: type, optional
    :raises NotImplementedError: if requested order is not available
    :return: Discretized derivative matrix
    :rtype: sp.dia_matrix
    :param boundary_condition: Which boundary condition to apply.
            Options are:

            - zero
            - periodic

            Defaults to "zero"
    :type boundary_condition: str, optional
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

    stencil = _mirror_sign_list(stencil)
    return _matrix_from_central_stencil(stencil, 1, N, L, dtype, boundary_condition)


def laplacian(N: tuple[int], L: tuple[float], order: int = 2, dtype: type = np.float64, 
              boundary_condition: str = "zero") -> sp.dia_matrix:
    """Finite difference double derivative in cartesian coordinates.
    Uses central stencil

    Example: approximate the second derivative of sin(x).

    >>> from qm_sim.spatial_derivative.cartesian import laplacian
    >>> import numpy as np
    >>> N = (1000,)
    >>> L = (2*np.pi,)
    >>> n = laplacian( N, L, boundary_condition="periodic" )
    >>> x = np.linspace( 0, L[0], N[0], endpoint=False)
    >>> y = np.sin(x)
    >>> np.allclose(n @ y, -np.sin(x)) # The analytical solution is -sin(x)
    True

    :param N: Discretization count along each axis
    :type N: tuple[int]
    :param L: System size along each axis
    :type L: tuple[float]
    :param order: Numerical order of the differentiation scheme.
        Options are:

            - 2
            - 4
            - 6
            - 8

        Defaults to 2.
    :type order: int, optional
    :param dtype: datatype of matrix. 
        Defaults to np.float64
    :type dtype: type, optional
    :raises NotImplementedError: if requested order is not available
    :return: Discretized derivative matrix
    :rtype: sp.dia_matrix
    :param boundary_condition: Which boundary condition to apply.
            Options are:

            - zero
            - periodic

            Defaults to "zero"
    :type boundary_condition: str, optional
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
    
    stencil = _mirror_list(stencil)
    return _matrix_from_central_stencil(stencil, 2, N, L, dtype, boundary_condition)
    

def _matrix_from_central_stencil(stencil: list[float], power: int, 
                         N: tuple[int], L: tuple[float], dtype: type, 
                         boundary_condition: str = "zero") -> sp.dia_matrix:
    """
    Creates a full matrix from a central stencil. Determines indices from stencil
    """

    available_boundary_conditions = ["zero", "periodic"]
    if boundary_condition not in available_boundary_conditions:
        raise ValueError(f"Invalid boundary condition: {boundary_condition}. Options are: " 
                         + ", ".join(available_boundary_conditions))

    if boundary_condition == "zero":
        indices = np.arange(len(stencil)) - len(stencil) // 2
        axis_indices = [indices for _ in N]

    elif boundary_condition == "periodic":
        indices = np.arange(len(stencil)) - len(stencil) // 2
        # There might be different sizes for each axis.
        # Therefore, we keep the updated indices for each axis
        periodic_indices = [list() for _ in N]
        periodic_stencil = []
        for i, ind in enumerate(indices):
            # Lower diagonals
            if ind < 0:
                for i, n in enumerate(N):
                    periodic_indices[i].append(-n - ind)
                periodic_stencil.append(stencil[i])
            # Upper diagonals
            elif ind > 0:
                for i, n in enumerate(N):
                    periodic_indices[i].append(n - ind)
                periodic_stencil.append(stencil[i])
            # Also keep the central diagonals
            else:
                for i in range(len(N)):
                    periodic_indices[i] += list(indices)
                periodic_stencil += stencil
        axis_indices = np.array(periodic_indices)
        stencil = np.array(periodic_stencil)

    mat = np.zeros((1,1))    
    for l, n, indices in zip(L, N, axis_indices):
        h = l / n
        next_mat = 1/h**power * sp.diags(
            stencil,
            indices,
            shape=(n, n),
            dtype=dtype,
            format="dia"
            )
        mat = sp.kronsum(mat, next_mat, format="dia")

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
