# Benchmark Results

| | |
|---|---|
| **Date** | 2026-02-22 20:33:34 |
| **Commit** | `114cbff` (master) |

## Charts

![Latency](charts/latency.svg)

![Throughput](charts/throughput.svg)

## Summary

| Benchmark | N | Mean | 95% CI | Std dev | Throughput |
|-----------|--:|-----:|--------|--------:|------------|
| fft |    64 |  248.90 µs | [246.19 µs … 252.39 µs] |   16.01 µs | 257.13 Kelem/s |
| fft |   256 |  396.39 µs | [392.64 µs … 400.75 µs] |   21.18 µs | 645.83 Kelem/s |
| fft |  1024 |  417.01 µs | [406.94 µs … 429.06 µs] |   57.15 µs | 2.46 Melem/s |
| fft |  4096 |  486.75 µs | [473.38 µs … 502.78 µs] |   76.03 µs | 8.42 Melem/s |
| fft | 16384 |  579.30 µs | [575.54 µs … 583.78 µs] |   21.14 µs | 28.28 Melem/s |
| fft | 65536 |  975.68 µs | [971.91 µs … 979.77 µs] |   20.10 µs | 67.17 Melem/s |
| | | | | | |
| ifft |    64 |  250.18 µs | [248.12 µs … 252.34 µs] |   10.74 µs | 255.82 Kelem/s |
| ifft |   256 |  395.41 µs | [392.25 µs … 399.52 µs] |   18.69 µs | 647.44 Kelem/s |
| ifft |  1024 |  401.58 µs | [398.59 µs … 405.02 µs] |   16.51 µs | 2.55 Melem/s |
| ifft |  4096 |  473.71 µs | [471.19 µs … 476.48 µs] |   13.64 µs | 8.65 Melem/s |
| ifft | 16384 |  656.57 µs | [654.00 µs … 659.49 µs] |   14.04 µs | 24.95 Melem/s |
| ifft | 65536 |    1.14 ms | [1.14 ms … 1.14 ms] |   23.20 µs | 57.49 Melem/s |
| | | | | | |
| roundtrip |    64 |  504.64 µs | [499.85 µs … 509.68 µs] |   25.18 µs | 126.82 Kelem/s |
| roundtrip |   256 |  792.37 µs | [787.85 µs … 797.21 µs] |   23.91 µs | 323.08 Kelem/s |
| roundtrip |  1024 |  803.36 µs | [794.24 µs … 818.30 µs] |   64.47 µs | 1.27 Melem/s |
| roundtrip |  4096 |  942.26 µs | [936.78 µs … 948.52 µs] |   30.12 µs | 4.35 Melem/s |
| roundtrip | 16384 |    1.17 ms | [1.16 ms … 1.18 ms] |   47.67 µs | 13.97 Melem/s |
| roundtrip | 65536 |    2.14 ms | [2.12 ms … 2.15 ms] |   79.48 µs | 30.68 Melem/s |
