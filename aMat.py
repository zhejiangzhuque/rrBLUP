import time

import numpy as np

# Define the add_mat function as previously described with corrections
def add_mat_numpy(mark):
    p = np.mean(0.5 * mark, axis=0)  # Ensure p is an ndarray for element-wise operations
    t = np.ones((mark.shape[0], 1))
    p_reshape = p.reshape(1,-1)
    x = np.dot(t, p.reshape(1, -1))
    m = mark - 2 * np.dot(t, p.reshape(1, -1))  # Corrected matrix multiplication
    rel = np.dot(m, m.T)
    q = 1 - p
    sum_val = np.sum(2 * p * q)  # Element-wise multiplication
    relf = rel / sum_val  # Element-wise division
    return relf

# Create fixed fake data as specified
mark_numpy = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # First individual with all 0s
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Second individual with all 1s
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],  # Third individual with all 2s
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # Fourth individual with all 0s
])

start_time = time.time()
relf_numpy = add_mat_numpy(mark_numpy)
print("Additive genetic relationship matrix:\n", relf_numpy)
end_time = time.time()
print("NumPy version execution time:", end_time - start_time, "seconds")


# Create large test data for NumPy version
num_individuals = 20000
num_markers = 32000
# Each row alternates between 0, 1, and 2
mark_numpy_large = np.tile(np.array([0, 1, 2]), (num_individuals, num_markers // 3 + 1))[:, :num_markers]

# Time the execution with large data for NumPy version
start_time = time.time()
relf_numpy_large = add_mat_numpy(mark_numpy_large)
end_time = time.time()
print("NumPy version execution time:", end_time - start_time, "seconds")
