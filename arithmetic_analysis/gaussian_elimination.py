import numpy as np
from numpy import float64
from numpy.typing import NDArray

def retroactive_resolution(
    coefficients: NDArray[float64], vector: NDArray[float64]
) -> NDArray[float64]:
    """
    This function performs a retroactive linear system resolution
    for a triangular matrix.
    """
    rows, columns = np.shape(coefficients)

    x: NDArray[float64] = np.zeros((rows, 1), dtype=float)
    for row in reversed(range(rows)):
        total = np.dot(coefficients[row, row + 1 :], x[row + 1 :])
        x[row, 0] = (vector[row] - total) / coefficients[row, row]

    return x

def gaussian_elimination(
    coefficients: NDArray[float64], vector: NDArray[float64]
) -> NDArray[float64]:
    """
    This function performs Gaussian elimination method with partial pivoting.
    """
    rows, columns = np.shape(coefficients)

    if rows != columns:
        raise ValueError("Matrix must be square for Gaussian elimination.")

    augmented_mat = np.column_stack((coefficients, vector)).astype(float64)

    for row in range(rows - 1):
        # Partial pivoting
        max_row = np.argmax(np.abs(augmented_mat[row:, row])) + row
        augmented_mat[[row, max_row]] = augmented_mat[[max_row, row]]

        pivot = augmented_mat[row, row]
        for col in range(row + 1, columns):
            factor = augmented_mat[col, row] / pivot
            augmented_mat[col, :] -= factor * augmented_mat[row, :]

    x = retroactive_resolution(augmented_mat[:, 0:columns], augmented_mat[:, columns:columns + 1])

    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        raise ValueError("Matrix is singular or ill-conditioned.")

    return x

if __name__ == "__main__":
    import doctest

    doctest.testmod()
    
