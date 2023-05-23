"""
PyTorch backend for finding eigenstates
"""

import numpy as np
import torch
from scipy import sparse as sp

PYTORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def torch_get_eigen(mat: sp.dia_matrix, n: int, shape: tuple[int], **kwargs):
    """Calculate :code:`n` eigenvalues of :code:`mat`. Reshape output to :code:`shape`

    :param mat: Matrix to calculate eigenvectors and -values for
    :type mat: sp.dia_matrix
    :param n: Amount of eigenvectors and -values to calculate
    :type n: int
    :param shape: Output shape for eigenvectors
    :type shape: tuple[int]
    :return: eigenvalues, eigenvectors
    :rtype: tuple[np.ndarray(shape = (:code:`n`)), np.ndarray(shape = (:code:`n`, :code:`shape`)]
    """
    H = spmatrix_to_tensor(mat)
    v, w = torch.lobpcg(H, k=n, largest=False)
    w = w.cpu().numpy()
    w = np.array([w[:, i].reshape(shape, order="F") for i in range(n)])
    return v, w


def spmatrix_to_tensor(mat: sp.spmatrix) -> torch.Tensor:
    """
    Convert a sparse diagonal scipy matrix to a sparse pytorch tensor.
    """

    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.from_numpy(indices)
    v = torch.from_numpy(values)
    shape = coo.shape

    return torch.sparse_coo_tensor(i, v, shape, device=PYTORCH_DEVICE)
