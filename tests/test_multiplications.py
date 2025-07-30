import numpy as np
from scipy import sparse
from time import perf_counter
from tqdm import tqdm

sparse_matrix =  sparse.random(1000, 1000, density=0.1, format='csc', data_rvs=np.random.rand)
random_matrix =  sparse_matrix.toarray()  # Convert to dense for comparison

vectors = np.random.rand(1000, 10)

# test multiplication speed

m = 10000
start = perf_counter()
for i in tqdm(range(m)):
    result_dense = random_matrix @ vectors
end = perf_counter()
print(f"Dense multiplication time: {end - start:.4f} seconds")
start = perf_counter()
for i in tqdm(range(m)):
    result_sparse = sparse_matrix @ vectors
end = perf_counter()
print(f"Sparse multiplication time: {end - start:.4f} seconds")