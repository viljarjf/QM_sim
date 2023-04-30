"""
PyTorch backend for finding eigenstates
"""
from scipy import sparse as sp
import numpy as np
import torch


PYTORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_eigen(mat: sp.dia_matrix, n: int, shape: tuple[int], **kwargs):
    """Calculate `n` eigenvalues of `mat`. Reshape output to `shape`

    Args:
        mat (sp.dia_matrix): Matrix to calculate eigenvectors and -values
        n (int): Amount of eigenvectors and -values to find
        shape (tuple[int]): shape of eivenvectors

    Returns:
        np.ndarray: eigenvalues, shape (n,)
        np.ndarray: eigenvectors, shape (n, *`shape`)
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
