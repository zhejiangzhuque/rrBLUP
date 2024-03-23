import jax.numpy as jnp
from jax import jit
import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
@jit
def add_mat_jax(mark):
    """
        mark use 0,1,2 encoding
    Args:
        mark:  shape is n,m

    Returns:
    """

    p = jnp.mean(0.5 * mark, axis=0)
    t = jnp.ones((mark.shape[0], 1))
    m = mark - 2 * jnp.dot(t, p.reshape(1, -1))
    rel = jnp.dot(m, m.T)
    q = 1 - p
    sum_val = jnp.sum(2 * p * q)
    relf = rel / sum_val
    return relf


def small_test():
    global _, start_time, end_time
    # Fixed fake data for JAX
    mark_jax = jnp.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    # Warm-up call to include JIT compilation time separately
    _ = add_mat_jax(mark_jax).block_until_ready()
    # Measure execution time after compilation
    start_time = time.time()
    relf_jax = add_mat_jax(mark_jax).block_until_ready()
    print("Additive genetic relationship matrix:\n", relf_jax)
    end_time = time.time()
    print("JAX version execution time (after JIT compilation):", end_time - start_time, "seconds")
    # Perform PCA
    pca = PCA(n_components=3)  # Adjust n_components as needed
    principal_components = pca.fit_transform(relf_jax)

    # Plot the first two principal components
    plt.figure(figsize=(8, 6))
    plt.scatter(principal_components[:, 0], principal_components[:, 1])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Relationship Matrix')
    plt.show()


def large_test():
    global _, start_time, end_time
    num_individuals = 20000
    num_markers = 40000
    mark_large = jnp.tile(jnp.array([0, 1, 2]), (num_individuals, num_markers // 3 + 1))[:, :num_markers]
    # Warm-up JIT compilation with large data
    _ = add_mat_jax(mark_large).block_until_ready()
    # Time the execution with large data
    start_time = time.time()
    relf_large = add_mat_jax(mark_large).block_until_ready()
    end_time = time.time()
    print("JAX version execution time (after JIT compilation):", end_time - start_time, "seconds")

    pca = PCA(n_components=3)  # Adjust n_components as needed
    principal_components = pca.fit_transform(relf_large)


