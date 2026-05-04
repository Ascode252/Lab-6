import numpy as np
import sys
import time

sizes = [10_000, 100_000, 500_000, 1_000_000, 5_000_000]
n_features = 50

print(f"{'Rows':>12} {'Features':>10} {'Matrix MB':>12} {'sys bytes':>15} {'Gen Time (s)':>14}")
print("-" * 70)

for n in sizes:
    t0 = time.time()
    X = np.random.randn(n, n_features).astype(np.float64)    
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int32)
    elapsed = time.time() - t0

    mb = X.nbytes / (1024 ** 2)
    sys_bytes = sys.getsizeof(X)

    print(f"{n:>12,} {n_features:>10} {mb:>12.2f} {sys_bytes:>15} {elapsed:>14.4f}")
    del X, y
