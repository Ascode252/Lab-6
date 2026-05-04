import numpy as np
import time

def simulate_broadcast(source_data, n_workers):
    return [source_data.copy() for _ in range(n_workers)]

def simulate_reduce(worker_data_list, reduce_fn=np.add):
    result = worker_data_list[0].copy()
    for arr in worker_data_list[1:]:
        result = reduce_fn(result, arr)
    return result

def simulate_allreduce_naive(worker_data_list):
    reduced = simulate_reduce(worker_data_list)
    return simulate_broadcast(reduced, len(worker_data_list))

def simulate_allreduce_ring(worker_data_list):
    n = len(worker_data_list)
    total = np.zeros_like(worker_data_list[0])
    for arr in worker_data_list:
        total += arr
    return [total.copy() for _ in range(n)]

if __name__ == "__main__":
    np.random.seed(0)
    N_WORKERS = 4
    VECTOR_SIZE = 8

    worker_data = [
        np.random.randint(1, 10, size=VECTOR_SIZE).astype(float)
        for _ in range(N_WORKERS)
    ]

    print("Worker gradients:")
    for i, d in enumerate(worker_data):
        print(f"Worker {i}: {d}")

    expected_sum = sum(worker_data)
    print(f"\nExpected global sum: {expected_sum}")

    result_naive = simulate_allreduce_naive(worker_data)
    result_ring = simulate_allreduce_ring(worker_data)

    print(f"\nNaive all-reduce Worker 0: {result_naive[0]}")
    print(f"Ring all-reduce Worker 0: {result_ring[0]}")

    assert np.allclose(result_naive[0], expected_sum), "Naive failed!"
    assert np.allclose(result_ring[0], expected_sum), "Ring failed!"
    print("\nAll assertions passed.")
