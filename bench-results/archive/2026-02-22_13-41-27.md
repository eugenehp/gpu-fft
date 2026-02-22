# Benchmark Results

| | |
|---|---|
| **Date** | 2026-02-22 13:43:12 |
| **Commit** | `7d73d3c` (master) |

## Summary

| Benchmark | N | Mean | 95% CI | Std dev | Throughput |
|-----------|--:|-----:|--------|--------:|------------|
| fft |    64 |  248.90 µs | [246.19 µs … 252.39 µs] |   16.01 µs | 257.13 Kelem/s |
| fft |   256 |  354.67 µs | [352.61 µs … 356.89 µs] |   10.91 µs | 721.79 Kelem/s |
| fft |  1024 |  741.46 µs | [725.90 µs … 756.71 µs] |   79.17 µs | 1.38 Melem/s |
| fft |  4096 |    1.25 ms | [1.24 ms … 1.25 ms] |   18.63 µs | 3.29 Melem/s |
| | | | | | |
| ifft |    64 |  250.18 µs | [248.12 µs … 252.34 µs] |   10.74 µs | 255.82 Kelem/s |
| ifft |   256 |  377.19 µs | [373.06 µs … 381.35 µs] |   21.20 µs | 678.70 Kelem/s |
| ifft |  1024 |  712.71 µs | [702.92 µs … 722.82 µs] |   51.03 µs | 1.44 Melem/s |
| ifft |  4096 |    1.02 ms | [1.02 ms … 1.02 ms] |   21.37 µs | 4.02 Melem/s |
| | | | | | |
| roundtrip |    64 |  504.64 µs | [499.85 µs … 509.68 µs] |   25.18 µs | 126.82 Kelem/s |
| roundtrip |   256 |  697.18 µs | [690.93 µs … 703.77 µs] |   32.80 µs | 367.20 Kelem/s |
| roundtrip |  1024 |    1.37 ms | [1.35 ms … 1.40 ms] |  126.21 µs | 746.99 Kelem/s |
| roundtrip |  4096 |    2.31 ms | [2.30 ms … 2.32 ms] |   52.06 µs | 1.77 Melem/s |

## Raw Output

<details>
<summary>expand</summary>

```
     Running benches/fft_bench.rs (target/release/deps/fft_bench-d2f659c578aa9e12)
Benchmarking fft/64
Benchmarking fft/64: Warming up for 2.0000 s
Benchmarking fft/64: Collecting 100 samples in estimated 5.0550 s (20k iterations)
Benchmarking fft/64: Analyzing
fft/64                  time:   [244.56 µs 246.66 µs 249.06 µs]
                        thrpt:  [256.97 Kelem/s 259.46 Kelem/s 261.69 Kelem/s]
Found 14 outliers among 100 measurements (14.00%)
  11 (11.00%) high mild
  3 (3.00%) high severe
Benchmarking fft/256
Benchmarking fft/256: Warming up for 2.0000 s
Benchmarking fft/256: Collecting 100 samples in estimated 5.4631 s (15k iterations)
Benchmarking fft/256: Analyzing
fft/256                 time:   [352.30 µs 354.43 µs 356.72 µs]
                        thrpt:  [717.65 Kelem/s 722.28 Kelem/s 726.65 Kelem/s]
Found 7 outliers among 100 measurements (7.00%)
  1 (1.00%) low mild
  2 (2.00%) high mild
  4 (4.00%) high severe
Benchmarking fft/1024
Benchmarking fft/1024: Warming up for 2.0000 s
Benchmarking fft/1024: Collecting 100 samples in estimated 7.1122 s (10k iterations)
Benchmarking fft/1024: Analyzing
fft/1024                time:   [725.71 µs 742.42 µs 757.93 µs]
                        thrpt:  [1.3510 Melem/s 1.3793 Melem/s 1.4110 Melem/s]
Benchmarking fft/4096
Benchmarking fft/4096: Warming up for 2.0000 s

Warning: Unable to complete 100 samples in 5.0s. You may wish to increase target time to 6.3s, enable flat sampling, or reduce sample count to 60.
Benchmarking fft/4096: Collecting 100 samples in estimated 6.2872 s (5050 iterations)
Benchmarking fft/4096: Analyzing
fft/4096                time:   [1.2436 ms 1.2486 ms 1.2540 ms]
                        thrpt:  [3.2662 Melem/s 3.2806 Melem/s 3.2936 Melem/s]
Found 12 outliers among 100 measurements (12.00%)
  9 (9.00%) high mild
  3 (3.00%) high severe

Benchmarking ifft/64
Benchmarking ifft/64: Warming up for 2.0000 s
Benchmarking ifft/64: Collecting 100 samples in estimated 6.1856 s (25k iterations)
Benchmarking ifft/64: Analyzing
ifft/64                 time:   [247.73 µs 249.85 µs 252.08 µs]
                        thrpt:  [253.89 Kelem/s 256.15 Kelem/s 258.35 Kelem/s]
Found 1 outliers among 100 measurements (1.00%)
  1 (1.00%) high mild
Benchmarking ifft/256
Benchmarking ifft/256: Warming up for 2.0000 s
Benchmarking ifft/256: Collecting 100 samples in estimated 5.3479 s (15k iterations)
Benchmarking ifft/256: Analyzing
ifft/256                time:   [373.65 µs 381.19 µs 388.30 µs]
                        thrpt:  [659.28 Kelem/s 671.58 Kelem/s 685.14 Kelem/s]
Found 20 outliers among 100 measurements (20.00%)
  5 (5.00%) low severe
  3 (3.00%) low mild
  7 (7.00%) high mild
  5 (5.00%) high severe
Benchmarking ifft/1024
Benchmarking ifft/1024: Warming up for 2.0000 s
Benchmarking ifft/1024: Collecting 100 samples in estimated 6.7376 s (10k iterations)
Benchmarking ifft/1024: Analyzing
ifft/1024               time:   [699.07 µs 715.27 µs 731.66 µs]
                        thrpt:  [1.3995 Melem/s 1.4316 Melem/s 1.4648 Melem/s]
Found 5 outliers among 100 measurements (5.00%)
  1 (1.00%) low mild
  3 (3.00%) high mild
  1 (1.00%) high severe
Benchmarking ifft/4096
Benchmarking ifft/4096: Warming up for 2.0000 s

Warning: Unable to complete 100 samples in 5.0s. You may wish to increase target time to 5.3s, enable flat sampling, or reduce sample count to 60.
Benchmarking ifft/4096: Collecting 100 samples in estimated 5.2779 s (5050 iterations)
Benchmarking ifft/4096: Analyzing
ifft/4096               time:   [1.0181 ms 1.0231 ms 1.0280 ms]
                        thrpt:  [3.9845 Melem/s 4.0035 Melem/s 4.0231 Melem/s]

Benchmarking roundtrip/64
Benchmarking roundtrip/64: Warming up for 2.0000 s
Benchmarking roundtrip/64: Collecting 100 samples in estimated 5.2298 s (10k iterations)
Benchmarking roundtrip/64: Analyzing
roundtrip/64            time:   [496.41 µs 501.42 µs 506.88 µs]
                        thrpt:  [126.26 Kelem/s 127.64 Kelem/s 128.93 Kelem/s]
Found 2 outliers among 100 measurements (2.00%)
  2 (2.00%) high mild
Benchmarking roundtrip/256
Benchmarking roundtrip/256: Warming up for 2.0000 s
Benchmarking roundtrip/256: Collecting 100 samples in estimated 6.9595 s (10k iterations)
Benchmarking roundtrip/256: Analyzing
roundtrip/256           time:   [679.95 µs 685.82 µs 692.10 µs]
                        thrpt:  [369.89 Kelem/s 373.28 Kelem/s 376.50 Kelem/s]
Found 1 outliers among 100 measurements (1.00%)
  1 (1.00%) high mild
Benchmarking roundtrip/1024
Benchmarking roundtrip/1024: Warming up for 2.0000 s

Warning: Unable to complete 100 samples in 5.0s. You may wish to increase target time to 6.8s, enable flat sampling, or reduce sample count to 60.
Benchmarking roundtrip/1024: Collecting 100 samples in estimated 6.8057 s (5050 iterations)
Benchmarking roundtrip/1024: Analyzing
roundtrip/1024          time:   [1.3360 ms 1.3661 ms 1.3952 ms]
                        thrpt:  [733.96 Kelem/s 749.56 Kelem/s 766.45 Kelem/s]
Found 5 outliers among 100 measurements (5.00%)
  3 (3.00%) low mild
  2 (2.00%) high mild
Benchmarking roundtrip/4096
Benchmarking roundtrip/4096: Warming up for 2.0000 s
Benchmarking roundtrip/4096: Collecting 100 samples in estimated 5.1635 s (2200 iterations)
Benchmarking roundtrip/4096: Analyzing
roundtrip/4096          time:   [2.3020 ms 2.3121 ms 2.3223 ms]
                        thrpt:  [1.7638 Melem/s 1.7716 Melem/s 1.7793 Melem/s]
Found 2 outliers among 100 measurements (2.00%)
  2 (2.00%) high mild

```

</details>
