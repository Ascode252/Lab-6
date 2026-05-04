import numpy as np

serial_fractions = [0.05, 0.10, 0.20, 0.50]
workers = [1, 2, 4, 8, 16, 32, 64]

print(f"{'Workers':>8}", end="")
for fs in serial_fractions:
    print(f" fs={fs:.0%} ", end="")
print()
print("-" * (8 + 8 * len(serial_fractions)))

for p in workers:
    print(f"{p:>8}", end="")
    for fs in serial_fractions:
        speedup = 1 / (fs + (1 - fs) / p)
        print(f" {speedup:>6.2f}", end="")
    print()
