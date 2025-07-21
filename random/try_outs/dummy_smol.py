import numpy as np
def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
    if len(a[0]) != len(b):
        return -1
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    m = a.shape[0]
    c = np.arange(m, dtype=float)
    for idx in range(len(a)):
        c[idx] = a[ idx, : ] @ b.T
    return c

print(matrix_dot_vector([[1.5, 2.5], [3.0, 4.0]], [2, 1]))